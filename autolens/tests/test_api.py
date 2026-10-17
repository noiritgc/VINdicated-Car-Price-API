import math
from fastapi.testclient import TestClient
from app.main import app, STATE


def test_inference_and_schema():
    with TestClient(app) as c:
        assert c.get('/api/v1/health').json()['status'] == 'ready'
        case = c.get('/api/v1/cases?seed=42').json()['cases'][0]
        result = c.post('/api/v1/predict', json=case['vehicle'])
        assert result.status_code == 200
        r = result.json()
        assert math.isfinite(r['estimated_price'])
        assert 0 < r['prediction_interval']['lower'] < r['estimated_price'] < r['prediction_interval']['upper']
        assert len(r['influences']) == 5
        assert r['support']['same_make_model_training_rows'] >= len(r['comparables'])
        assert result.headers['X-Request-ID']
        assert c.get('/').status_code == 200
        assert c.get('/openapi.json').status_code == 200


def test_batch_matches_single_and_is_deterministic():
    with TestClient(app) as c:
        examples = c.get('/api/v1/cases?count=3&seed=7').json()
        assert examples == c.get('/api/v1/cases?count=3&seed=7').json()
        vehicles = [r['vehicle'] for r in examples['cases']]
        batch = c.post('/api/v1/predict/batch', json={'vehicles': vehicles}).json()
        assert batch['count'] == 3
        for v, p in zip(vehicles, batch['predictions']):
            single = c.post('/api/v1/predict', json=v).json()
            assert single['estimated_price'] == p['estimated_price']
            assert single['prediction_interval'] == p['prediction_interval']


def test_unknown_history_and_unseen_categories():
    with TestClient(app) as c:
        v = {'year': 2022, 'make': 'NeverSeen', 'model': 'Prototype', 'mileage': 35000}
        r = c.post('/api/v1/predict', json=v)
        assert r.status_code == 200
        assert r.json()['support']['level'] == 'limited'
        assert r.json()['comparables'] == []
        assert any('No training examples' in w for w in r.json()['warnings'])
        assert any('Unspecified' in w for w in r.json()['warnings'])


def test_validation_rejects_bad_inputs_and_oversized_batch():
    with TestClient(app) as c:
        v = {'year': 2022, 'make': 'Toyota', 'model': 'Camry', 'mileage': 35000}
        invalid = [{**v, 'mileage': -1}, {**v, 'year': 3000}, {**v, 'make': ' '},
                   {**v, 'askingPrice': 0}, {**v, 'carfax_one_price': 1},
                   {**v, 'ownerCount': 0}, {**v, 'sellerState': 'California'}]
        for example in invalid:
            assert c.post('/api/v1/predict', json=example).status_code == 422
        assert c.post('/api/v1/predict', content='not json', headers={'Content-Type': 'application/json'}).status_code == 422
        assert c.post('/api/v1/predict/batch', json={'vehicles': []}).status_code == 422
        assert c.post('/api/v1/predict/batch', json={'vehicles': [v]*101}).status_code == 422
        assert c.get('/api/v1/cases?count=21').status_code == 422


def test_asking_price_and_normalization():
    with TestClient(app) as c:
        v = {'year': 2022, 'make': 'Toyota', 'model': 'Camry', 'mileage': 35000,
             'drivetrain': 'FWD', 'askingPrice': 1}
        r = c.post('/api/v1/predict', json=v).json()
        assert r['asking_price_comparison']['position'] == 'below_estimate'
        v.update(make=' toyota ', model='CAMRY', drivetrain='Front-Wheel Drive')
        normalized = c.post('/api/v1/predict', json=v).json()
        assert normalized['estimated_price'] == r['estimated_price']


def test_all_demo_cases_are_valid_and_train_test_disjoint():
    from app.main import Vehicle, as_vehicle
    with TestClient(app):
        for row in STATE['cases']:
            Vehicle(**as_vehicle(row))
        # Vehicle-ID disjointness is asserted in train.py; lookalike vehicles can share features.
        assert STATE['metrics']['train_rows'] + STATE['metrics']['test_rows'] + STATE['metrics']['calibration_rows'] == STATE['metrics']['eligible_rows']

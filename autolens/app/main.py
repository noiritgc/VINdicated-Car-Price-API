"""AutoLens API. Run: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from app.features import FEATURES, CATEGORICAL, features, normalize

ROOT = Path(__file__).resolve().parents[1]
STATE = {}
log = logging.getLogger('autolens')
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]

class Vehicle(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, json_schema_extra={'examples': [{'year': 2022, 'make': 'Toyota', 'model': 'Camry', 'trim': 'SE', 'mileage': 42000, 'ownerCount': 1, 'accidentCount': 0, 'sellerState': 'CA'}]})
    year: int = Field(ge=1980, le=2027)
    make: Text
    model: Text
    mileage: float = Field(ge=0, le=1_000_000)
    trim: Text | None = None
    bodyStyle: Text | None = None
    drivetrain: Text | None = None
    fuelType: Text | None = None
    engine: Text | None = None
    sellerState: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r'^[A-Za-z]{2}$')] | None = None
    accidentCount: int | None = Field(default=None, ge=0, le=50)
    accidentDetail: Text | None = None
    ownerCount: int | None = Field(default=None, ge=1, le=50)
    askingPrice: float | None = Field(default=None, gt=0, le=10_000_000)

class Batch(BaseModel):
    vehicles: list[Vehicle] = Field(min_length=1, max_length=100)

class PriceInterval(BaseModel):
    lower: float
    upper: float
    nominal_coverage: float
    method: str

class TrainingSupport(BaseModel):
    same_make_model_training_rows: int
    level: str

class Influence(BaseModel):
    feature: str
    value: str | float | int | None
    log_price_contribution: float
    direction: str

class Comparable(BaseModel):
    year: int
    make: str
    model: str
    trim: str | None
    mileage: int
    reference_price: float
    sellerState: str

class AskingComparison(BaseModel):
    asking_price: float
    difference_usd: float
    difference_percent: float
    position: str

class Valuation(BaseModel):
    estimated_price: float
    currency: str
    prediction_interval: PriceInterval
    support: TrainingSupport
    influences: list[Influence]
    explanation_note: str
    comparables: list[Comparable]
    warnings: list[str]
    model_version: str
    target: str
    reference_year: int
    asking_price_comparison: AskingComparison | None = None
    batch_processing_ms: float

class BatchValuation(BaseModel):
    count: int
    predictions: list[Valuation]

@asynccontextmanager
async def lifespan(app):
    model = CatBoostRegressor()
    model.load_model(str(ROOT / 'artifacts/valuation.cbm'))
    STATE.update(model=model, metrics=json.loads((ROOT / 'artifacts/metrics.json').read_text()),
                 comparables=pd.read_json(ROOT / 'artifacts/comparables.json'),
                 cases=json.loads((ROOT / 'artifacts/cases.json').read_text()))
    STATE['xtrain'] = features(STATE['comparables'])
    yield
    STATE.clear()

app = FastAPI(title='AutoLens Vehicle Valuation API', version='1.0.0', lifespan=lifespan,
              description='Predict the supplied CARFAX reference-price target. Includes calibrated ranges, training comparables, batch inference, and held-out demo cases.')
app.mount('/static', StaticFiles(directory=ROOT / 'app/static'), name='static')

@app.middleware('http')
async def request_metadata(request: Request, call_next):
    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception('Unhandled API exception request_id=%s', request_id)
        response = JSONResponse(status_code=500, content={'error': 'Internal server error', 'request_id': request_id})
    response.headers['X-Request-ID'] = request_id
    response.headers['X-Response-Time-Ms'] = f'{(time.perf_counter()-start)*1000:.1f}'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT / 'app/static/index.html')

@app.get('/api/v1/health')
def health():
    return {'status': 'ready', 'model_version': STATE['metrics']['model_version']}

@app.get('/api/v1/metrics')
def metrics():
    return STATE['metrics']

@app.get('/api/v1/catalog')
def catalog(make: str | None = None, model: str | None = None):
    df = STATE['comparables']
    makes = sorted(df.make.unique().tolist())
    if make:
        df = df[df.make.map(normalize) == normalize(make)]
    models = sorted(df.model.unique().tolist()) if make else []
    if model:
        df = df[df.model.map(normalize) == normalize(model)]
    return {'makes': makes, 'models': models, 'trims': sorted(df.trim.dropna().unique().tolist()) if model else [],
            'reference_year': 2026}

def as_vehicle(row):
    return {k: row.get(k) for k in FEATURES if row.get(k) is not None}

@app.get('/api/v1/cases')
def cases(count: int = Query(1, ge=1, le=20), seed: int | None = Query(None, ge=0, le=2**32-1)):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(STATE['cases']), size=count, replace=False)
    return {'source': 'Held-out test vehicles, not synthetic records', 'cases': [
        {'vehicle': as_vehicle(STATE['cases'][int(i)]),
         'reference_price': STATE['cases'][int(i)]['carfax_one_price'],
         'context': {k: STATE['cases'][int(i)].get(k) for k in ['sellerCity', 'exteriorColor']}}
        for i in chosen]}

def predict_many(vehicles):
    start = time.perf_counter()
    data = [v.model_dump(exclude={'askingPrice'}) for v in vehicles]
    x = features(data)
    estimates = np.exp(STATE['model'].predict(x, thread_count=2))
    # SHAP values explain model outputs in log-price space, not causal dollar changes.
    shap = STATE['model'].get_feature_importance(Pool(x, cat_features=CATEGORICAL), type='ShapValues', thread_count=2)
    q = STATE['metrics']['interval']['log_radius']
    results = []
    train = STATE['comparables']
    xt = STATE['xtrain']
    for i, (v, estimate) in enumerate(zip(vehicles, estimates)):
        warnings = []
        mask = (xt.make == x.iloc[i]['make']) & (xt.model == x.iloc[i]['model'])
        same = train[mask]
        if not len(same):
            warnings.append('No training examples for this make/model. Estimate may extrapolate; range coverage is not guaranteed.')
        elif len(same) < 30:
            warnings.append(f'Limited training support: {len(same)} examples for this make/model.')
        if v.year < int(train.year.min()) or v.year > int(train.year.max()) or v.mileage > float(train.mileage.max()):
            warnings.append('Year or mileage is outside the training range. Treat this estimate cautiously.')
        missing = [k for k in FEATURES if k not in ('year', 'make', 'model', 'mileage') and data[i].get(k) is None]
        if missing:
            warnings.append('Unspecified features are treated as unknown: ' + ', '.join(missing))
        if v.accidentCount is not None and v.accidentCount > 1:
            warnings.append('Training accidentCount is limited to 0/1. Counts above 1 are outside training support.')
        if len(same) and v.trim and normalize(v.trim) not in set(same.trim.map(normalize)):
            warnings.append('Trim was not observed for this make/model in training.')
        peers = same.copy()
        if len(peers):
            # Similarity ranks by age, mileage and trim; reference price is never used to rank.
            peers['distance'] = abs(peers.year-v.year) * 2 + abs(peers.mileage-v.mileage) / 15000
            if v.trim:
                peers['distance'] += (peers.trim.map(normalize) != normalize(v.trim)).astype(int)*2
            peers = peers.nsmallest(4, 'distance')
        comparable_rows = []
        for _, p in peers.iterrows():
            comparable_rows.append({'year': int(p.year), 'make': p.make, 'model': p.model,
                                    'trim': None if pd.isna(p.trim) else p.trim, 'mileage': int(p.mileage),
                                    'reference_price': float(p.carfax_one_price), 'sellerState': p.sellerState})
        top = np.argsort(np.abs(shap[i, :-1]))[::-1][:5]
        result = {'estimated_price': round(float(estimate), 2), 'currency': 'USD',
                  'prediction_interval': {'lower': round(float(estimate*np.exp(-q)), 2),
                                          'upper': round(float(estimate*np.exp(q)), 2),
                                          'nominal_coverage': .90, 'method': 'split conformal (log-price)'},
                  'support': {'same_make_model_training_rows': len(same), 'level': 'strong' if len(same) >= 100 else 'moderate' if len(same) >= 30 else 'limited'},
                  'influences': [{'feature': FEATURES[int(j)], 'value': data[i].get(FEATURES[int(j)]),
                                  'log_price_contribution': round(float(shap[i, j]), 4),
                                  'direction': 'up' if shap[i, j] > 0 else 'down'} for j in top],
                  'explanation_note': 'SHAP contributions are relative to the model baseline in log-price space; they are not causal price effects.',
                  'comparables': comparable_rows, 'warnings': warnings, 'model_version': STATE['metrics']['model_version'],
                  'target': 'CARFAX reference price; not a sale-price appraisal', 'reference_year': 2026}
        if v.askingPrice is not None:
            diff = float(estimate)-v.askingPrice
            result['asking_price_comparison'] = {'asking_price': v.askingPrice, 'difference_usd': round(diff, 2),
                                                 'difference_percent': round(diff/estimate*100, 2),
                                                 'position': 'below_estimate' if diff > 0 else 'above_estimate' if diff < 0 else 'at_estimate'}
        results.append(result)
    elapsed = round((time.perf_counter()-start)*1000, 1)
    for result in results:
        result['batch_processing_ms'] = elapsed
    return results

@app.post('/api/v1/predict', response_model=Valuation, response_model_exclude_none=True)
def predict(vehicle: Vehicle):
    return predict_many([vehicle])[0]

@app.post('/api/v1/predict/batch', response_model=BatchValuation, response_model_exclude_none=True)
def batch_predict(batch: Batch):
    return {'count': len(batch.vehicles), 'predictions': predict_many(batch.vehicles)}

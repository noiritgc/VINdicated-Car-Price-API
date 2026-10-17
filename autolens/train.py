"""Reproduce model, calibration, independent test evaluation, and serving artifacts."""
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.features import FEATURES, CATEGORICAL, features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)

def score(y, prediction):
    return {'mae_usd': round(float(mean_absolute_error(y, prediction)), 2),
            'rmse_usd': round(float(math.sqrt(mean_squared_error(y, prediction))), 2),
            'r2': round(float(r2_score(y, prediction)), 4),
            'median_absolute_percentage_error': round(float(np.median(np.abs(y - prediction) / y) * 100), 2)}

def main():
    source = ROOT / 'data/training_data.csv'
    raw = pd.read_csv(source)
    df = raw[(raw.carfax_one_price > 0) & raw.year.between(1900, 2027) & (raw.mileage >= 0)].copy()
    # The key begins with a 17-character vehicle identifier; repeated listings must not cross splits.
    df['vehicle_group'] = df.carfax_primary_key.str[:17]
    df = df.drop_duplicates('vehicle_group', keep='first').reset_index(drop=True)
    train, other = train_test_split(df, test_size=.30, random_state=42)
    calibration, test = train_test_split(other, test_size=.50, random_state=43)
    assert not set(train.vehicle_group) & set(other.vehicle_group)
    assert not set(calibration.vehicle_group) & set(test.vehicle_group)
    model = CatBoostRegressor(iterations=1000, depth=7, learning_rate=.055,
                             loss_function='RMSE', random_seed=42, thread_count=4,
                             l2_leaf_reg=5, verbose=200, allow_writing_files=False)
    # Fixed hyperparameters: neither calibration nor test is used for model selection.
    model.fit(features(train), np.log(train.carfax_one_price), cat_features=CATEGORICAL)
    model.save_model(str(OUT / 'valuation.cbm'))
    cal_pred = model.predict(features(calibration))
    residual = np.abs(np.log(calibration.carfax_one_price.to_numpy()) - cal_pred)
    # Split conformal finite-sample corrected quantile in log-price space.
    rank = min(len(residual), math.ceil((len(residual) + 1) * .90))
    q = float(np.sort(residual)[rank - 1])
    pred = np.exp(model.predict(features(test)))
    actual = test.carfax_one_price.to_numpy()
    baseline = np.full(len(test), train.carfax_one_price.median())
    # Stronger baseline using training-only make/model medians.
    group_medians = train.groupby(['make', 'model']).carfax_one_price.median()
    grouped = np.array([group_medians.get((r.make, r.model), train.carfax_one_price.median()) for r in test.itertuples()])
    coverage = float(np.mean((actual >= pred * np.exp(-q)) & (actual <= pred * np.exp(q))))
    metrics = {'model': 'CatBoost log-price regressor', 'model_version': '1.0.0',
               'target': 'carfax_one_price (supplied CARFAX reference price, not completed sale price)',
               'input_rows': len(raw), 'eligible_rows': len(df), 'duplicate_or_invalid_rows_removed': len(raw)-len(df),
               'train_rows': len(train), 'calibration_rows': len(calibration), 'test_rows': len(test),
               'split': '70/15/15 random, deduplicated on first 17 key characters; seeds 42/43',
               'reference_year': 2026, 'features': FEATURES, 'test': score(actual, pred),
               'median_baseline_test': score(actual, baseline), 'make_model_baseline_test': score(actual, grouped),
               'interval': {'nominal_coverage': .90, 'test_coverage': round(coverage, 4), 'log_radius': q,
                            'method': 'Split conformal absolute log residual, calibration set only'},
               'training_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'runtime': {'python': platform.python_version(), 'catboost': catboost.__version__},
               'feature_importance': dict(sorted(zip(FEATURES, map(float, model.feature_importances_)), key=lambda p: -p[1])),
               'segments': {}}
    for body, part in test.assign(prediction=pred).groupby('bodyStyle'):
        if len(part) >= 30:
            metrics['segments'][str(body)] = {'rows': len(part), **score(part.carfax_one_price, part.prediction)}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    # Comparable prices are exclusively training examples; demo cases exclusively test examples.
    context = pd.read_csv(ROOT / 'data/listing_context.csv').drop_duplicates('_primaryKey')
    for name, split in [('comparables', train), ('cases', test)]:
        rows = split.merge(context, left_on='carfax_primary_key', right_on='_primaryKey', how='left')
        cols = FEATURES + ['carfax_one_price', 'sellerCity', 'exteriorColor', 'serviceRecordCount']
        rows[cols].to_json(OUT / f'{name}.json', orient='records')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

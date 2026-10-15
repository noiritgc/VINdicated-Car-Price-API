"""
Train the used-car price model.

Target  : carfax_one_price (the CARFAX "one price" asking price, USD)
Model   : LightGBM gradient-boosted trees on log(price), native categorical handling
Validation: 5-fold GroupKFold by VIN (so the same car never sits in train and test)

Run:  python training/train.py  [--data data/training_data.csv]
Outputs backend/artifacts/{model.txt, meta.json} (the backend loads them on startup)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..", "backend")
sys.path.insert(0, BACKEND)  # features.py lives with the backend so it ships with the API
from features import (CATEGORICAL, NUMERIC, FEATURES, REFERENCE_YEAR,  # noqa: E402
                      build_features)

ART = os.path.join(BACKEND, "artifacts")

PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=10,
    feature_fraction=0.8,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    cat_smooth=10,
    cat_l2=10,
    max_cat_to_onehot=8,
    verbose=-1,
    seed=42,
    # Sanity constraints: more miles or accidents can only lower the price
    monotone_constraints=[{"mileage": -1, "log_mileage": -1,
                           "accidentCount": -1, "accident_severity": -1}.get(f, 0) for f in FEATURES],
    monotone_constraints_method="advanced",
)


def metrics(y_true, y_pred):
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "training_data.csv"))
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    raw = raw[(raw.carfax_one_price >= 1000) & raw.mileage.notna()].reset_index(drop=True)
    y = raw.carfax_one_price.values
    groups = raw.carfax_primary_key.str[:17].values

    # Category vocabularies come from the full data so codes are stable.
    vocab = {c: sorted(raw[c].dropna().astype(str).unique().tolist()) for c in
             ["make", "model", "trim", "bodyStyle", "drivetrain", "fuelType", "engine", "sellerState"]}
    X = build_features(raw, vocab)

    # ---- cross-validation -------------------------------------------------
    oof = np.zeros(len(raw))
    best_iters = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[tr], np.log(y[tr]), categorical_feature=CATEGORICAL)
        dte = lgb.Dataset(X.iloc[te], np.log(y[te]), categorical_feature=CATEGORICAL)
        m = lgb.train(PARAMS, dtr, 5000, valid_sets=[dte],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        best_iters.append(m.best_iteration)
        oof[te] = np.exp(m.predict(X.iloc[te], num_iteration=m.best_iteration))
        print(f"fold {fold}: iters={m.best_iteration}  MAPE={metrics(y[te], oof[te])['mape']:.3%}")

    cv = metrics(y, oof)
    print("CV:", json.dumps(cv, indent=2))

    # Baseline for context: median price of same make+model+year (else make, else global)
    base = raw.assign(p=y).groupby(["make", "model", "year"]).p.transform("median")
    print("Baseline (median by make/model/year, in-sample) MAPE:",
          f"{metrics(y, base.values)['mape']:.3%}")

    # Residual spread (log space) -> used for the price range shown in the UI
    log_resid = np.log(y) - np.log(oof)
    q_lo, q_hi = np.quantile(log_resid, [0.10, 0.90])

    # ---- final model on all data ----------------------------------------
    n_iter = int(np.mean(best_iters) * 1.1)
    final = lgb.train(PARAMS, lgb.Dataset(X, np.log(y), categorical_feature=CATEGORICAL), n_iter)
    os.makedirs(ART, exist_ok=True)
    final.save_model(os.path.join(ART, "model.txt"))

    imp = dict(zip(FEATURES, final.feature_importance("gain").tolist()))
    tot = sum(imp.values())
    imp = {k: round(v / tot, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}

    # Lookup tables so the UI can offer valid dropdowns (make -> model -> trims)
    catalog = {}
    for (mk, md), g in raw.groupby(["make", "model"]):
        catalog.setdefault(mk, {})[md] = {
            "trims": sorted(g.trim.dropna().astype(str).unique().tolist()),
            "bodyStyles": g.bodyStyle.value_counts().index.tolist(),
            "drivetrains": g.drivetrain.dropna().value_counts().index.tolist(),
            "fuelTypes": g.fuelType.value_counts().index.tolist(),
            "engines": g.engine.value_counts().index.tolist(),
            "years": [int(g.year.min()), int(g.year.max())],
            "count": int(len(g)),
        }

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(raw)),
        "target": "carfax_one_price",
        "reference_year": REFERENCE_YEAR,
        "n_iterations": n_iter,
        "cv_metrics": cv,
        "interval_log_quantiles": {"p10": float(q_lo), "p90": float(q_hi)},
        "feature_importance": imp,
        "vocab": vocab,
        "catalog": catalog,
        "accident_severities": ["none", "minor", "moderate", "severe", "airbag"],
        "ranges": {
            "year": [int(raw.year.min()), int(raw.year.max())],
            "mileage": [0, int(raw.mileage.max())],
            "ownerCount": [1, int(raw.ownerCount.max())],
        },
    }
    with open(os.path.join(ART, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved model ({n_iter} trees) and meta to {ART}")


if __name__ == "__main__":
    main()

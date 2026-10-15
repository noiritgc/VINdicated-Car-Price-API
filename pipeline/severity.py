"""Step 2b: measure how much each accident severity level lowers a car's value.

The price model alone can't tell severity levels apart (too few severe cars), so we measure
them directly with cross-fitting:
  1. Split the non-test cars into K folds. For each fold, train on the others and price the
     held-out cars *as if they had no accident* (no car is priced by a model that saw it).
  2. effect(level) = median log(actual value / no-accident price) for that level, minus the
     same median for clean cars (which should be ~0).
Fair price for a car with an accident = no-accident model price x exp(effect(level)).

Usage: python -m pipeline.severity      (after train; writes severity_effects.json)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from pipeline import config
from pipeline.common import log
from pipeline.train import make_pool, predict_price

STEP = "severity"
NO_ACCIDENT = "none"


def no_accident(df: pd.DataFrame) -> pd.DataFrame:
    """Same cars with a clean history (unknown history stays unknown)."""
    return df.assign(accidentCount=df["accidentCount"].where(df["accidentCount"].isna(), 0.0))


def estimate_effects(df: pd.DataFrame, seed: int = config.SEED) -> dict[str, dict]:
    """{level: {"effect": log effect, "pct": %, "n": cars, "ci90": [lo%, hi%]}} from out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, config.SEVERITY_FOLDS, len(df))
    pred = pd.Series(np.nan, index=df.index)
    params = {**config.CATBOOST_PARAMS, **config.SEVERITY_CATBOOST_OVERRIDES}
    for k in range(config.SEVERITY_FOLDS):
        held = df[fold == k]
        model = CatBoostRegressor(**params).fit(make_pool(df[fold != k]))
        pred[held.index] = predict_price(model, no_accident(held))
        log(STEP, f"fold {k + 1}/{config.SEVERITY_FOLDS} done")

    log_ratio = np.log(df["price"] / pred)
    level = df["accidentSeverity"]
    baseline = float(log_ratio[level == NO_ACCIDENT].median())
    effects = {NO_ACCIDENT: {"effect": 0.0, "pct": 0.0, "n": int((level == NO_ACCIDENT).sum()), "ci90": [0.0, 0.0]}}
    for name in sorted(set(level.dropna()) - {NO_ACCIDENT}):
        r = (log_ratio[level == name] - baseline).to_numpy()
        if len(r) < config.SEVERITY_MIN_CARS:
            continue
        boots = [np.median(rng.choice(r, len(r))) for _ in range(config.SEVERITY_BOOTSTRAPS)]
        effect = min(float(np.median(r)), 0.0)          # an accident never adds value
        effects[name] = {"effect": effect, "pct": round(float(np.expm1(effect)) * 100, 1), "n": len(r),
                         "ci90": [round(float(np.expm1(np.percentile(boots, q))) * 100, 1) for q in (5, 95)]}
    return effects


def load_effects() -> dict[str, float]:
    """{level: log effect}. Empty if severity hasn't been measured (then accidents use the model as-is)."""
    if not config.SEVERITY_PATH.exists():
        return {}
    return {k: v["effect"] for k, v in json.loads(config.SEVERITY_PATH.read_text()).items()}


def accident_effect(row: pd.Series, effects: dict[str, float]) -> float | None:
    """Log effect for this car's accident level, or None to use the model's own accident handling."""
    if not effects or pd.isna(row.get("accidentSeverity")):
        return None
    return effects.get(row["accidentSeverity"])


def final_prices(model: CatBoostRegressor, df: pd.DataFrame, effects: dict[str, float]) -> np.ndarray:
    """Fair prices: no-accident model price x measured severity effect (model price if not measured)."""
    model_price = predict_price(model, df)
    if not effects:
        return model_price
    clean_price = predict_price(model, no_accident(df))
    eff = df["accidentSeverity"].map(effects)
    return np.where(eff.notna(), clean_price * np.exp(eff.fillna(0.0)), model_price)


def main() -> None:
    df = pd.read_parquet(config.CLEAN_PATH).merge(pd.read_parquet(config.SPLIT_PATH), on="listingId")
    dev = df[df["split"] != "test"].reset_index(drop=True)      # test cars stay untouched for evaluate.py
    effects = estimate_effects(dev)
    config.SEVERITY_PATH.write_text(json.dumps(effects, indent=1))
    for name, e in effects.items():
        log(STEP, f"{name:12} n={e['n']:5}  effect {e['pct']:+.1f}%  90% CI {e['ci90']}")


if __name__ == "__main__":
    main()

"""Synthetic prices for --dev mode, used while the free sample has prices redacted.

The rules mimic real used-car pricing (depreciation, mileage, accidents, owners,
region) so the model, SHAP breakdowns and rounds behave realistically. The real
`priceDifferential` (asking price minus CarGurus' estimate) is kept, so a car
CarGurus rated OVERPRICED is still overpriced against our fair price.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from pipeline import config


def _stable_unit(text: str) -> float:
    """Deterministic number in [-1, 1] for a string (same model -> same value)."""
    digest = hashlib.md5(text.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF * 2 - 1


def _trim_adjustment(trim: object) -> float:
    if not isinstance(trim, str):
        return 0.0
    words = trim.split()
    adj = sum(v for k, v in config.DEV_TRIM_PREMIUM_WORDS.items() if k in trim)
    adj += sum(v for k, v in config.DEV_TRIM_BASE_WORDS.items() if k in words)
    return adj


def new_price(row: pd.Series) -> float:
    base = config.DEV_BODY_BASE.get(row["bodyStyle"], config.DEV_BODY_BASE_DEFAULT)
    tier = config.DEV_MAKE_TIER.get(row["make"], config.DEV_MAKE_TIER_DEFAULT)
    model_var = 1 + config.DEV_MODEL_SPREAD * _stable_unit(f"{row['make']}|{row['model']}")
    extras = (
        _trim_adjustment(row["trim"])
        + config.DEV_DRIVETRAIN_ADJ.get(row["drivetrain"], 0.0)
        + config.DEV_FUEL_ADJ.get(row["fuelType"], 0.0)
    )
    return base * tier * model_var * (1 + extras)


def value_share(age: float) -> float:
    """Share of the new price a car keeps at a given age."""
    if age <= 0:
        return 1.0
    share = config.DEV_FIRST_YEAR_KEEP * config.DEV_YEARLY_KEEP ** (age - 1)
    return max(share, config.DEV_MIN_VALUE_SHARE)


def expected_price(row: pd.Series) -> float:
    """Noise-free 'fair' price for one listing."""
    price = new_price(row) * value_share(row["age"])
    expected_miles = config.DEV_MILES_PER_YEAR * max(row["age"], 1)
    mileage = row["mileage"] if pd.notna(row["mileage"]) else expected_miles
    lo, hi = config.DEV_MILEAGE_CLIP
    price *= float(np.clip(np.exp(-config.DEV_MILEAGE_COEF * (mileage - expected_miles)), lo, hi))
    if pd.notna(row["accidentCount"]):
        price *= config.DEV_ACCIDENT_KEEP ** row["accidentCount"]
    if pd.notna(row["ownerCount"]):
        extra_owners = max(row["ownerCount"] - 1, 0)
        price *= 1 - min(config.DEV_OWNER_PENALTY * extra_owners, config.DEV_OWNER_PENALTY_CAP)
    price *= 1 + config.DEV_STATE_ADJ.get(row["sellerState"], 0.0)
    return price


def fill_synthetic_prices(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Fill expectedPrice and price. Needs an `age` column."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    fair = out.apply(expected_price, axis=1)
    fair = fair * np.exp(rng.normal(0, config.DEV_NOISE_SIGMA, len(out)))
    # priceDifferential = expected - asking (positive = good deal). The dataset's schema.json
    # says the opposite, but GREAT_PRICE rows have positive values and OVERPRICED negative.
    diff = pd.to_numeric(out["priceDifferential"], errors="coerce").fillna(0.0)
    asking = np.maximum(fair - diff, fair * config.DEV_MIN_PRICE_SHARE_OF_EXPECTED)
    out["expectedPrice"] = fair.round(0)
    out["price"] = asking.round(0)
    return out

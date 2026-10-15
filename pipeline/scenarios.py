"""Scenario adjustments applied after the model: branded titles and private-party sales.

Both use real data when the optional files exist in data/raw, else config fallbacks:
  - ForCar title effects CSV with columns: make, model, title_type (salvage|rebuilt), pct_change
  - Craigslist vehicles CSV (Kaggle austinreese) with columns: url, price, year, manufacturer, model, odometer
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from pipeline import config
from pipeline.common import log

STEP = "scenarios"
TITLE_TYPES = {"Salvage title": "salvage", "Rebuilt title": "rebuilt"}


@lru_cache(maxsize=1)
def _forcar() -> pd.DataFrame | None:
    if not config.FORCAR_TITLE_PATH.exists():
        log(STEP, f"no {config.FORCAR_TITLE_PATH.name}; using config title discounts")
        return None
    df = pd.read_csv(config.FORCAR_TITLE_PATH)
    df["make"] = df["make"].str.lower()
    df["model"] = df["model"].str.lower()
    df["title_type"] = df["title_type"].str.lower()
    log(STEP, f"loaded {len(df)} ForCar title rows")
    return df


def title_discount(make: str, model: str, title: str) -> float:
    """Fractional price change for a branded title (e.g. -0.3)."""
    fc = _forcar()
    if fc is not None:
        kind = fc[fc["title_type"] == TITLE_TYPES[title]]
        exact = kind[(kind["make"] == make.lower()) & (kind["model"] == model.lower())]
        if len(exact):
            return float(exact["pct_change"].median())
        if len(kind):
            return float(kind["pct_change"].median())
    return config.TITLE_DISCOUNTS[title]


def _bucket(series: pd.Series, size: int) -> pd.Series:
    return (series // size).astype("Int64")


@lru_cache(maxsize=1)
def _private_ratios() -> tuple[dict[tuple, float], float] | None:
    if not config.CRAIGSLIST_PATH.exists():
        log(STEP, f"no {config.CRAIGSLIST_PATH.name}; using config private discount")
        return None
    cl = pd.read_csv(config.CRAIGSLIST_PATH, usecols=["url", "price", "year", "manufacturer", "model", "odometer"])
    cl = cl.dropna()
    cl = cl[cl["price"].between(config.MIN_PRICE, config.MAX_PRICE)]
    cl["seller"] = cl["url"].str.extract(r"/(cto|ctd)/", expand=False)
    cl = cl.dropna(subset=["seller"])
    cl["key_make"] = cl["manufacturer"].str.lower()
    cl["key_model"] = cl["model"].str.lower().str.split().str[0]
    cl["yb"] = _bucket(cl["year"], config.YEAR_BUCKET_SIZE)
    cl["mb"] = _bucket(cl["odometer"], config.MILEAGE_BUCKET_SIZE)
    keys = ["key_make", "key_model", "yb", "mb"]
    med = cl.groupby(keys + ["seller"])["price"].agg(["median", "size"]).unstack("seller")
    med = med.dropna()
    med = med[(med[("size", "cto")] >= config.CRAIGSLIST_MIN_PAIRS) & (med[("size", "ctd")] >= config.CRAIGSLIST_MIN_PAIRS)]
    ratios = (med[("median", "cto")] / med[("median", "ctd")] - 1).to_dict()
    overall = float(pd.Series(ratios).median()) if ratios else config.PRIVATE_PARTY_DISCOUNT
    log(STEP, f"Craigslist: {len(cl)} rows, {len(ratios)} matched groups, overall private discount {overall:.1%}")
    return ratios, overall


def private_discount(make: str, model: str, year: int, mileage: float) -> float:
    """Fractional price change for buying from a private seller instead of a dealer."""
    pr = _private_ratios()
    if pr is None:
        return config.PRIVATE_PARTY_DISCOUNT
    ratios, overall = pr
    key = (make.lower(), model.lower().split()[0], year // config.YEAR_BUCKET_SIZE,
           int(mileage // config.MILEAGE_BUCKET_SIZE))
    return float(ratios.get(key, overall))

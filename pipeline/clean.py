"""Step 1: load raw listings (Carfax or CarGurus), clean them, write clean.parquet.

Usage:
    python -m pipeline.clean                        # Carfax (default): real CARFAX One Price values
    python -m pipeline.clean --source cargurus      # CarGurus: real prices required (paid)
    python -m pipeline.clean --source cargurus --dev  # CarGurus free sample with synthetic prices
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.common import log
from pipeline.sources import load_listings
from pipeline.synthetic import fill_synthetic_prices

STEP = "clean"
NUMERIC_COLUMNS = ["price", "expectedPrice", "priceDifferential", "mileage", "year",
                   "accidentCount", "ownerCount"]


class Counter:
    """Tracks row counts after each step for the log."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.prev = len(df)
        log(STEP, f"loaded {self.prev} rows")

    def __call__(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        log(STEP, f"{label}: {len(df)} rows (-{self.prev - len(df)})")
        self.prev = len(df)
        return df


def unredact(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace(config.REDACTED_TOKEN, np.nan)
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["_firstSeenAt"] = pd.to_datetime(out["_firstSeenAt"], utc=True, errors="coerce")
    out["listingId"] = out["listingId"].astype(str)
    return out


def derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["listing_year"] = out["_firstSeenAt"].dt.year
    out["age"] = (out["listing_year"] - out["year"]).clip(lower=0)
    out["miles_per_year"] = out["mileage"] / out["age"].clip(lower=1)
    return out


def clean(raw: pd.DataFrame, dev: bool) -> pd.DataFrame:
    count = Counter(raw)
    df = unredact(raw)
    df = count(df[df["vehicleCondition"].isin(config.KEEP_CONDITIONS)], "keep USED")
    df = count(df[~df["dealRatingKey"].isin(config.DROP_DEAL_RATINGS)], "drop OUTLIER")
    df = df.sort_values("_firstSeenAt").drop_duplicates("listingId", keep="last")
    df = count(df, "dedupe listingId")
    df = count(df[df["year"] >= config.MIN_YEAR], f"year >= {config.MIN_YEAR}")
    df = count(df[df["mileage"].notna() & (df["mileage"] <= config.MAX_MILEAGE)],
               f"mileage present and <= {config.MAX_MILEAGE}")
    df = derive(df)

    real_prices = df["price"].notna().sum()
    if dev:
        log(STEP, f"--dev: filling synthetic prices ({real_prices} rows had real prices; all replaced)")
        df = fill_synthetic_prices(df)
    elif real_prices == 0:
        raise SystemExit("All prices are redacted. Use --dev, or a source with real prices (e.g. --source carfax).")

    df = count(df[df["price"].notna()], "price present")
    df = count(df[df["price"].between(config.MIN_PRICE, config.MAX_PRICE)],
               f"price in ${config.MIN_PRICE:,}-${config.MAX_PRICE:,}")
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=config.SOURCES, default=config.DEFAULT_SOURCE)
    parser.add_argument("--dev", action="store_true", help="fill synthetic prices")
    args = parser.parse_args()
    log(STEP, f"source={args.source}")
    df = clean(load_listings(args.source), dev=args.dev)
    df["is_dev"] = args.dev
    df["source"] = args.source
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CLEAN_PATH, index=False)
    log(STEP, f"wrote {len(df)} rows to {config.CLEAN_PATH.name} (dev={args.dev})")


if __name__ == "__main__":
    main()

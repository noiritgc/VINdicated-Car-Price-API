"""Independent sanity check: our model vs Kelley Blue Book fair price ranges.

Uses Rebrowser's free AutoTrader sample (pipeline/data/raw/autotrader/), whose
kbbFairPriceLow/High are real. Accident/owner history isn't in that data, so the
model predicts with those unknown. Appends a section to report.md.

Usage: python scripts/compare_kbb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402
from pipeline.common import log  # noqa: E402
from pipeline.sources import CARFAX_BODY_STYLES, CARFAX_DRIVETRAINS, read_files  # noqa: E402
from pipeline.train import predict_price  # noqa: E402

STEP = "compare_kbb"
LISTING_YEAR = 2026
HEADER = "## Independent check: Kelley Blue Book"


def autotrader_features(raw: pd.DataFrame) -> pd.DataFrame:
    lo = pd.to_numeric(raw["kbbFairPriceLow"], errors="coerce")
    hi = pd.to_numeric(raw["kbbFairPriceHigh"], errors="coerce")
    df = pd.DataFrame({
        "make": raw["makeName"], "model": raw["modelName"],
        "trim": raw["trim"],
        "bodyStyle": raw["bodyStyle"].map(CARFAX_BODY_STYLES).fillna(raw["bodyStyle"].astype(str).str.lower()),
        "drivetrain": raw["drivetrain"].map(CARFAX_DRIVETRAINS).fillna(raw["drivetrain"]),
        "fuelType": raw["fuelType"],
        "age": (LISTING_YEAR - pd.to_numeric(raw["year"], errors="coerce")).clip(lower=0),
        "mileage": pd.to_numeric(raw["mileage"], errors="coerce"),
        "accidentCount": float("nan"), "ownerCount": float("nan"),
        "sellerState": raw["sellerState"],
        "listingType": raw["listingType"], "kbb_low": lo, "kbb_high": hi,
    })
    return df[(df["listingType"] == "Used") & (lo > 0) & (hi > 0) & df["mileage"].notna()]


def main() -> None:
    at = autotrader_features(read_files(config.KBB_DIR))
    covered = set(json.loads(config.COVERAGE_PATH.read_text())["covered"])
    at = at[(at["make"] + "|" + at["model"]).isin(covered)].copy()
    if at.empty:
        sys.exit("No AutoTrader cars match the model's covered make/models.")
    model = CatBoostRegressor()
    model.load_model(str(config.MODEL_PATH))
    at["pred"] = predict_price(model, at)
    mid = (at["kbb_low"] + at["kbb_high"]) / 2
    at["ratio"] = at["pred"] / mid
    within = at["pred"].between(at["kbb_low"], at["kbb_high"])
    near = at["pred"].between(at["kbb_low"] * 0.9, at["kbb_high"] * 1.1)
    by_make = (at.groupby("make")["ratio"].agg(["size", "median"]).query("size >= 30")
               .sort_values("median").round(3).rename(columns={"size": "cars", "median": "our / KBB mid"}))

    lines = [
        HEADER, "",
        f"{len(at):,} used AutoTrader cars in models we cover (history unknown, so this is a rough check).", "",
        "| metric | value |", "|---|---|",
        f"| our price inside KBB fair range | {within.mean():.1%} |",
        f"| within KBB range ±10% | {near.mean():.1%} |",
        f"| median our / KBB midpoint | {at['ratio'].median():.3f} |",
        f"| median abs. gap to KBB midpoint | {(at['ratio'] - 1).abs().median():.1%} |",
        "", "By make (≥30 cars):", "", by_make.to_markdown(), "",
    ]
    report = config.REPORT_PATH.read_text().split(HEADER)[0].rstrip() + "\n\n" + "\n".join(lines)
    config.REPORT_PATH.write_text(report)
    log(STEP, f"{len(at)} cars: inside KBB range {within.mean():.1%}, ±10% {near.mean():.1%}, "
              f"median ratio {at['ratio'].median():.3f}")


if __name__ == "__main__":
    main()

"""Turn each raw data source into the pipeline's internal listing format (config.USED_COLUMNS).

carfax   (default) Rebrowser Carfax sample. price = CARFAX "One Price" value estimate (free, real).
cargurus           Rebrowser CarGurus listings. price = asking price (premium field; --dev fakes it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config

CARFAX_BODY_STYLES = {
    "SUV": "suv", "Sedan": "sedan", "Pickup": "pickup_truck", "Coupe": "coupe", "Van": "van",
    "Minivan": "minivan", "Hatchback": "hatchback", "Wagon": "wagon", "Convertible": "convertible",
}
CARFAX_DRIVETRAINS = {
    "FWD": "Front-Wheel Drive", "AWD": "All-Wheel Drive", "4WD": "Four-Wheel Drive", "RWD": "Rear-Wheel Drive",
}
UNSPECIFIED = "Unspecified"


def read_files(raw_dir: Path) -> pd.DataFrame:
    """All .parquet/.csv files in a folder (parquet wins over a CSV with the same name)."""
    parquet = {p.stem: p for p in raw_dir.glob("*.parquet")}
    csv = {p.stem: p for p in raw_dir.glob("*.csv") if p.stem not in parquet}
    files = sorted({**csv, **parquet}.values())
    if not files:
        raise FileNotFoundError(f"No .parquet or .csv files in {raw_dir}")
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False) for p in files]
    return pd.concat(frames, ignore_index=True)


def _require(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source} data is missing columns: {missing}")


# ---------------------------------------------------------------- cargurus
def from_cargurus(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in config.USED_COLUMNS if c != "accidentDetail"]
    _require(raw, cols, "CarGurus")
    out = raw[cols].copy()
    out["accidentDetail"] = np.nan
    return out


# ---------------------------------------------------------------- carfax
def _owner_count(history: object) -> float:
    """Number of owners in the Carfax owner history (NaN when empty/unknown)."""
    if history is None or (isinstance(history, float) and np.isnan(history)):
        return np.nan
    if isinstance(history, str):
        history = json.loads(history) if history.startswith("[") else [history]
    n = len(history)
    return float(n) if n > 0 else np.nan


def damage_label(text: object) -> str | float:
    """'Accident Reported: Minor to Moderate Damage' -> 'Accident reported (minor to moderate damage)'."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    parts = [p.strip() for p in text.split(":")]
    head = parts[0]
    if head.lower().startswith("accident"):
        detail = parts[1].lower() if len(parts) > 1 else ""
        return f"Accident reported ({detail})" if detail else "Accident reported"
    return f"Damage reported ({head.lower().replace(' damage', '')})"


def _accidents(no_accidents: pd.Series, damage: pd.Series) -> pd.Series:
    """0 = Carfax says no accidents, 1 = accident/damage reported, NaN = unknown."""
    out = pd.Series(np.nan, index=no_accidents.index)
    out[no_accidents.map(lambda v: v is True or v == "True")] = 0.0
    out[damage.notna()] = 1.0
    return out


def from_carfax(raw: pd.DataFrame) -> pd.DataFrame:
    _require(raw, ["_primaryKey", "_firstSeenAt", "year", "make", "model", "trim", "bodyStyle", "drivetrain",
                   "fuelType", "engine", "mileage", "onePrice", "vehicleCondition", "noAccidents",
                   "ownerHistory", "sellerState", "snapshotDamageSeverity"], "Carfax")
    trim = raw["trim"].where(raw["trim"] != UNSPECIFIED)
    sub = raw.get("subTrim")
    if sub is not None:
        trim = trim.where(sub.isna() | trim.isna(), trim + " " + sub.astype(str))
    price = pd.to_numeric(raw["onePrice"].replace(config.REDACTED_TOKEN, np.nan), errors="coerce")
    price = price.where(price > 0)
    return pd.DataFrame({
        "listingId": raw["_primaryKey"].astype(str),
        "price": price,
        "expectedPrice": np.nan,        # no independent benchmark inside Carfax
        "priceDifferential": np.nan,
        "dealRatingKey": "NA",
        "mileage": raw["mileage"],
        "vehicleCondition": raw["vehicleCondition"].astype(str).str.upper(),
        "year": raw["year"],
        "make": raw["make"],
        "model": raw["model"],
        "trim": trim,
        "bodyStyle": raw["bodyStyle"].map(CARFAX_BODY_STYLES).fillna(raw["bodyStyle"].str.lower()),
        "drivetrain": raw["drivetrain"].map(CARFAX_DRIVETRAINS),
        "fuelType": raw["fuelType"],
        "engine": raw["engine"],
        "accidentCount": _accidents(raw["noAccidents"], raw["snapshotDamageSeverity"]),
        "ownerCount": raw["ownerHistory"].map(_owner_count),
        "sellerState": raw["sellerState"],
        "_firstSeenAt": raw["_firstSeenAt"],
        "accidentDetail": raw["snapshotDamageSeverity"].map(damage_label),
    })


ADAPTERS = {"carfax": from_carfax, "cargurus": from_cargurus}


def load_listings(source: str) -> pd.DataFrame:
    if source not in ADAPTERS:
        raise ValueError(f"Unknown source {source!r}; choose from {list(ADAPTERS)}")
    return ADAPTERS[source](read_files(config.RAW_DIR / source))

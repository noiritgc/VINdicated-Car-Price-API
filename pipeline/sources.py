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
    cols = [c for c in config.USED_COLUMNS if c not in ("accidentDetail", "accidentSeverity")]
    _require(raw, cols, "CarGurus")
    out = raw[cols].copy()
    out["accidentDetail"] = np.nan
    n = pd.to_numeric(out["accidentCount"].replace(config.REDACTED_TOKEN, np.nan), errors="coerce")
    out["accidentSeverity"] = np.select([n.isna(), n == 0, n == 1], [None, "none", "1 accident"], "2+ accidents")
    out["accidentSeverity"] = out["accidentSeverity"].where(n.notna())
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
    """Readable tag from Carfax's damage text, e.g.
    'Accident Reported: Moderate Damage' -> 'Accident reported (moderate damage)'
    'Minor to Moderate Damage'          -> 'Damage reported (minor to moderate)'
    'Hail Damage Reported: Moderate Damage' -> 'Hail damage reported (moderate)'
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan
    head, _, detail = (part.strip() for part in text.partition(":"))
    if not head.lower().endswith("reported"):          # severity only, e.g. 'Moderate Damage'
        head, detail = "Damage Reported", head
    label = head[0].upper() + head[1:].lower()
    detail = detail.lower()
    if "damage" in head.lower():                       # 'Damage reported (moderate)', not '(moderate damage)'
        detail = detail.removesuffix(" damage")
    return f"{label} ({detail})" if detail else label


SEVERITY_LEVELS = ("none", "minor", "moderate", "severe", "unspecified")


def severity_level(accident_count: float, damage_text: object) -> str | float:
    """Accident severity from Carfax's damage text: none / minor (minor-to-moderate) / moderate /
    severe (moderate-to-severe or airbag deployed) / unspecified (reported, no detail). NaN = unknown."""
    if pd.isna(accident_count):
        return np.nan
    if accident_count == 0:
        return "none"
    text = str(damage_text).lower() if isinstance(damage_text, str) else ""
    if "airbag" in text or "severe" in text:
        return "severe"
    if "minor to moderate" in text:
        return "minor"
    if "moderate" in text:
        return "moderate"
    return "unspecified"


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
    }).pipe(lambda df: df.assign(accidentSeverity=[
        severity_level(n, t) for n, t in zip(df["accidentCount"], raw["snapshotDamageSeverity"])]))


ADAPTERS = {"carfax": from_carfax, "cargurus": from_cargurus}


def load_listings(source: str) -> pd.DataFrame:
    if source not in ADAPTERS:
        raise ValueError(f"Unknown source {source!r}; choose from {list(ADAPTERS)}")
    return ADAPTERS[source](read_files(config.RAW_DIR / source))

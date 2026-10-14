"""Turn model SHAP values into a player-facing dollar breakdown that sums to the fair price.

SHAP values are in log-price space: log(pred) = E + sum(phi). We group features into
buckets (base, mileage, accidents, owners, location) and convert sequentially:
    running = exp(E + phi_base); delta_b = running * (exp(phi_b) - 1); running += delta_b
so the dollar lines add up exactly to the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from pipeline import config
from pipeline.train import make_pool

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "Washington, D.C.",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


@dataclass
class Breakdown:
    """Dollar lines; `lines[0]` is the base value. `buckets` maps line index -> bucket name."""
    lines: list[dict] = field(default_factory=list)
    buckets: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(line["amount"] for line in self.lines)

    @property
    def base(self) -> int:
        return self.lines[0]["amount"]

    def add(self, label: str, amount: int, bucket: str) -> None:
        self.lines.append({"label": label, "amount": int(amount)})
        self.buckets.append(bucket)

    def adjustments(self) -> dict[str, int]:
        return {b: line["amount"] for b, line in zip(self.buckets[1:], self.lines[1:])}


def round_to(value: float, step: int = config.ROUND_TO) -> int:
    return int(round(value / step) * step)


def shap_values(model: CatBoostRegressor, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-row SHAP contributions (columns = features) and the expected value."""
    raw = model.get_feature_importance(data=make_pool(df, with_label=False), type="ShapValues")
    phi = pd.DataFrame(raw[:, :-1], columns=config.FEATURES, index=df.index)
    return phi, raw[:, -1]


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def car_name(row: pd.Series) -> str:
    trim = f" {row['trim']}" if isinstance(row["trim"], str) else ""
    return f"{int(row['year'])} {row['make']} {row['model']}{trim}"


def mileage_label(mileage: float, age: float, amount: float) -> str:
    """'82k miles (above average for its age)'; the comparison is dropped if it contradicts the sign."""
    miles = f"{round(mileage / 1000):,}k miles"
    ratio = mileage / (config.AVG_MILES_PER_YEAR * max(age, 1))
    if ratio > 1 + config.MILEAGE_LABEL_TOLERANCE and amount < 0:
        return f"{miles} (above average for its age)"
    if ratio < 1 - config.MILEAGE_LABEL_TOLERANCE and amount > 0:
        return f"{miles} (below average for its age)"
    return miles


def bucket_label(bucket: str, row: pd.Series, amount: float = 0.0) -> str:
    if bucket == "mileage":
        return mileage_label(row["mileage"], row["age"], amount)
    if bucket == "accidents":
        n = int(row["accidentCount"])
        return "No accidents reported" if n == 0 else f"{plural(n, 'accident')} reported"
    if bucket == "owners":
        return plural(int(row["ownerCount"]), "previous owner")
    if bucket == "location":
        return f"Sold in {STATE_NAMES.get(row['sellerState'], row['sellerState'])}"
    raise ValueError(bucket)


def model_breakdown(row: pd.Series, phi: pd.Series, expected: float) -> Breakdown:
    """Breakdown of the model's prediction for one car (before scenarios)."""
    running = float(np.exp(expected + phi[config.BASE_FEATURES].sum()))
    deltas: dict[str, float] = {}
    for bucket in config.BUCKET_ORDER:
        b_phi = float(phi[config.BUCKET_FEATURES[bucket]].sum())
        deltas[bucket] = running * (np.exp(b_phi) - 1)
        running += deltas[bucket]

    fair = round_to(running)
    kept = {b: round_to(d) for b, d in deltas.items() if abs(d) >= config.MIN_LINE_DOLLARS}
    kept = {b: d for b, d in kept.items() if d != 0}
    bd = Breakdown()
    bd.add(f"Base value ({car_name(row)})", fair - sum(kept.values()), "base")
    for bucket, amount in kept.items():
        bd.add(bucket_label(bucket, row, amount), amount, bucket)
    return bd


def add_percent_line(bd: Breakdown, pct: float, label: str, bucket: str) -> int:
    """Append a scenario line worth pct of the current total. Returns the dollar amount."""
    amount = round_to(bd.total * pct)
    bd.add(label, amount, bucket)
    return amount

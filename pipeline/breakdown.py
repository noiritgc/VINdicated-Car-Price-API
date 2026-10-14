"""Turn the model's prediction into a player-facing dollar breakdown that sums to the fair price.

1. Mileage is judged against cars of the same age (a what-if):
       mileage line = price(actual miles) - price(typical miles for its age)
2. The typical-miles version of the car is split with SHAP (log-price space,
   log(pred) = E + sum(phi)) into base, accidents, owners and location, converted sequentially:
       running = exp(E + phi_base); delta_b = running * (exp(phi_b) - 1); running += delta_b
So the starting point means "this car with typical miles for its age", and the lines add up
exactly to the prediction.
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


# ---------------------------------------------------------------- typical miles by age
def typical_miles_table(df: pd.DataFrame) -> dict[int, int]:
    """Median mileage of real cars at each age (non-decreasing, rounded to 1,000)."""
    stats = df.groupby(df["age"].astype(int))["mileage"].agg(["median", "size"])
    table: dict[int, int] = {}
    running = 0.0
    for age in range(int(stats.index.max()) + 1):
        if age in stats.index and stats.loc[age, "size"] >= config.TYPICAL_MILES_MIN_ROWS:
            value = float(stats.loc[age, "median"])
        else:
            value = config.AVG_MILES_PER_YEAR * max(age, 1)
        running = max(running, value)
        table[age] = round_to(running, 1000)
    return table


def typical_miles(table: dict[int, int], age: float) -> int:
    age = int(age)
    return table.get(age, config.AVG_MILES_PER_YEAR * max(age, 1))


def with_typical_mileage(df: pd.DataFrame, table: dict[int, int]) -> pd.DataFrame:
    """Same cars, mileage replaced by the typical mileage for each car's age."""
    return df.assign(mileage=df["age"].map(lambda a: float(typical_miles(table, a))))


def mileage_label(mileage: float, typical: float) -> str:
    """'20k miles (typical for its age: ~6k)'."""
    return f"{round(mileage / 1000):,}k miles (typical for its age: ~{round(typical / 1000):,}k)"


def bucket_label(bucket: str, row: pd.Series, typical: float | None = None) -> str:
    if bucket == "mileage":
        return mileage_label(row["mileage"], typical if typical is not None else row["mileage"])
    if bucket == "accidents":
        n = int(row["accidentCount"])
        if n == 0:
            return "No accidents reported"
        detail = row.get("accidentDetail")
        return detail if isinstance(detail, str) else f"{plural(n, 'accident')} reported"
    if bucket == "owners":
        return plural(int(row["ownerCount"]), "previous owner")
    if bucket == "location":
        return f"Sold in {STATE_NAMES.get(row['sellerState'], row['sellerState'])}"
    raise ValueError(bucket)


def model_breakdown(row: pd.Series, phi_typical: pd.Series, expected: float,
                    price: float, typical: float) -> Breakdown:
    """Breakdown of the model's prediction for one car (before scenarios).

    phi_typical/expected: SHAP for this car at `typical` miles; price: prediction at actual miles.
    """
    running = float(np.exp(expected + phi_typical[config.BASE_FEATURES + config.MILEAGE_FEATURES].sum()))
    deltas: dict[str, float] = {}
    for bucket in config.BUCKET_ORDER:
        b_phi = float(phi_typical[config.BUCKET_FEATURES[bucket]].sum())
        deltas[bucket] = running * (np.exp(b_phi) - 1)
        running += deltas[bucket]
    deltas["mileage"] = price - running        # running is now the typical-miles price

    fair = round_to(price)
    kept = {b: round_to(deltas[b]) for b in config.LINE_ORDER if abs(deltas[b]) >= config.MIN_LINE_DOLLARS}
    kept = {b: d for b, d in kept.items() if d != 0}
    bd = Breakdown()
    bd.add(f"Base value ({car_name(row)})", fair - sum(kept.values()), "base")
    for bucket, amount in kept.items():
        bd.add(bucket_label(bucket, row, typical), amount, bucket)
    return bd


def add_percent_line(bd: Breakdown, pct: float, label: str, bucket: str) -> int:
    """Append a scenario line worth pct of the current total. Returns the dollar amount."""
    amount = round_to(bd.total * pct)
    bd.add(label, amount, bucket)
    return amount

"""Pipeline tests: breakdown math, synthetic prices, cleaning rules, rounds.json validity."""
from __future__ import annotations

import json
import math

import jsonschema
import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.breakdown import Breakdown, add_percent_line, model_breakdown
from pipeline.clean import clean
from pipeline.synthetic import expected_price
from pipeline.tips import format_tip


def car(**overrides) -> pd.Series:
    row = {"year": 2019, "make": "Honda", "model": "Civic", "trim": "EX", "bodyStyle": "sedan",
           "drivetrain": "Front-Wheel Drive", "fuelType": "Gasoline", "age": 7, "mileage": 82_000.0,
           "accidentCount": 1.0, "ownerCount": 3.0, "sellerState": "CA"}
    row.update(overrides)
    return pd.Series(row)


def fake_phi(**values: float) -> pd.Series:
    phi = pd.Series(0.0, index=config.FEATURES)
    for k, v in values.items():
        phi[k] = v
    return phi


# ---------------------------------------------------------------- breakdown
def test_breakdown_sums_exactly_to_rounded_prediction():
    phi = fake_phi(model=0.3, age=-0.2, mileage=-0.08, accidentCount=-0.07, ownerCount=-0.03, sellerState=0.02)
    expected = math.log(18_000)
    bd = model_breakdown(car(), phi, expected)
    prediction = math.exp(expected + phi.sum())
    assert bd.total == round(prediction / 100) * 100
    assert all(line["amount"] % 100 == 0 for line in bd.lines)
    assert bd.lines[0]["label"].startswith("Base value (2019 Honda Civic EX)")


def test_small_adjustments_fold_into_base():
    phi = fake_phi(mileage=0.001, accidentCount=-0.05)
    bd = model_breakdown(car(), phi, math.log(20_000))
    assert "mileage" not in bd.buckets
    assert "accidents" in bd.buckets
    assert bd.total == round(math.exp(math.log(20_000) + phi.sum()) / 100) * 100


def test_mileage_label_never_contradicts_sign():
    high_miles_but_positive = model_breakdown(car(mileage=150_000.0), fake_phi(mileage=0.05), math.log(20_000))
    assert high_miles_but_positive.lines[1]["label"] == "150k miles"
    high_miles_negative = model_breakdown(car(mileage=150_000.0), fake_phi(mileage=-0.05), math.log(20_000))
    assert "above average" in high_miles_negative.lines[1]["label"]


def test_percent_line_keeps_sum_exact():
    bd = Breakdown()
    bd.add("Base value", 15_000, "base")
    bd.add("1 accident reported", -1_200, "accidents")
    amount = add_percent_line(bd, -0.3, "Rebuilt title", "title")
    assert amount == -4_100  # 30% of 13,800, rounded to $100
    assert bd.total == 15_000 - 1_200 + amount


def test_tip_percent_uses_value_before_line():
    tip, source, url = format_tip("title", -4_500, 10_000)
    assert "-$4,500 (-45%)" in tip and source == "NMVTIS" and url.startswith("https://")


# ---------------------------------------------------------------- synthetic prices
def test_synthetic_price_moves_the_right_way():
    base = expected_price(car(accidentCount=0.0, ownerCount=1.0, mileage=84_000.0))
    assert expected_price(car(accidentCount=2.0, ownerCount=1.0, mileage=84_000.0)) < base
    assert expected_price(car(accidentCount=0.0, ownerCount=5.0, mileage=84_000.0)) < base
    assert expected_price(car(accidentCount=0.0, ownerCount=1.0, mileage=160_000.0)) < base
    assert expected_price(car(accidentCount=0.0, ownerCount=1.0, mileage=84_000.0, age=10)) < base


# ---------------------------------------------------------------- clean
def raw_frame() -> pd.DataFrame:
    base = {c: None for c in config.USED_COLUMNS}
    rows = []
    for i, (cond, rating, year, miles, price) in enumerate([
        ("USED", "FAIR_PRICE", 2018, 60_000, "15000"),
        ("CPO", "FAIR_PRICE", 2018, 60_000, "15000"),        # not USED
        ("USED", "OUTLIER", 2018, 60_000, "15000"),         # outlier
        ("USED", "FAIR_PRICE", 1998, 60_000, "15000"),      # too old
        ("USED", "FAIR_PRICE", 2018, 400_000, "15000"),     # too many miles
        ("USED", "FAIR_PRICE", 2018, 60_000, "900"),        # too cheap
        ("USED", "FAIR_PRICE", 2018, 60_000, "[PREMIUM]"),  # redacted
    ]):
        rows.append({**base, "listingId": str(i), "vehicleCondition": cond, "dealRatingKey": rating,
                     "year": year, "mileage": miles, "price": price, "expectedPrice": price,
                     "_firstSeenAt": "2026-09-01T00:00:00Z", "make": "Honda", "model": "Civic"})
    return pd.DataFrame(rows)


def test_clean_applies_every_filter():
    out = clean(raw_frame(), dev=False)
    assert out["listingId"].tolist() == ["0"]
    assert out.loc[0, "age"] == 8 and out.loc[0, "miles_per_year"] == 7_500


def test_clean_dedupes_on_listing_id_keeping_latest():
    df = pd.concat([raw_frame().iloc[[0]]] * 2, ignore_index=True)
    df.loc[1, "_firstSeenAt"] = "2026-09-05T00:00:00Z"
    df.loc[1, "price"] = "16000"
    out = clean(df, dev=False)
    assert len(out) == 1 and out.loc[0, "price"] == 16_000


def test_clean_refuses_fully_redacted_data_without_dev():
    df = raw_frame()
    df["price"] = config.REDACTED_TOKEN
    with pytest.raises(SystemExit):
        clean(df, dev=False)


# ---------------------------------------------------------------- rounds.json
@pytest.fixture(scope="module")
def rounds_payload() -> dict:
    if not config.ROUNDS_PATH.exists():
        pytest.skip("rounds.json not built yet")
    return json.loads(config.ROUNDS_PATH.read_text())


def test_rounds_match_schema_and_sum(rounds_payload):
    schema = json.loads(config.SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for r in rounds_payload["rounds"]:
        validator.validate(r)
        assert sum(line["amount"] for line in r["breakdown"]) == r["fair_price"]
        assert r["base_value"]["amount"] == r["breakdown"][0]["amount"]
        assert r["price_hint_range"]["low"] <= r["fair_price"] <= r["price_hint_range"]["high"]


def test_rounds_have_no_nan(rounds_payload):
    json.dumps(rounds_payload, allow_nan=False)


def test_pools_are_disjoint_and_complete(rounds_payload):
    quick, daily = set(rounds_payload["pools"]["quick"]), set(rounds_payload["pools"]["daily"])
    ids = {r["round_id"] for r in rounds_payload["rounds"]}
    assert not quick & daily
    assert quick | daily == ids
    assert len(quick) == config.QUICK_POOL_SIZE and len(daily) == config.DAILY_POOL_SIZE


def test_deal_or_scam_is_about_half_honest(rounds_payload):
    scams = [r["scam"] for r in rounds_payload["rounds"] if r["round_type"] == "deal_or_scam"]
    honest_share = np.mean([not s["is_scam"] for s in scams])
    assert abs(honest_share - config.HONEST_SHARE_OF_DEAL_OR_SCAM) <= 0.05
    assert all(s["true_value"] > 0 for s in scams)


# ---------------------------------------------------------------- carfax adapter
def carfax_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "_primaryKey": ["a", "b", "c"], "_firstSeenAt": ["2026-09-01T00:00:00Z"] * 3,
        "year": [2020, 2018, 2015], "make": ["Toyota", "Ford", "Honda"], "model": ["Camry", "F-150", "Civic"],
        "trim": ["LE", "Unspecified", "EX"], "subTrim": [None, None, "Sport"],
        "bodyStyle": ["Sedan", "Pickup", "Sedan"], "drivetrain": ["FWD", "4WD", "FWD"],
        "fuelType": ["Gasoline"] * 3, "engine": ["4 Cyl", "8 Cyl", "4 Cyl"], "mileage": [40_000.0, 90_000.0, 120_000.0],
        "onePrice": [21_000.0, None, 9_500.0], "vehicleCondition": ["Used"] * 3,
        "noAccidents": [True, None, None],
        "snapshotDamageSeverity": [None, None, "Accident Reported: Minor to Moderate Damage"],
        "ownerHistory": [['{"ownerNumber":1}'], [], ['{"ownerNumber":1}', '{"ownerNumber":2}', '{"ownerNumber":3}']],
        "sellerState": ["CA", "TX", "OH"],
    })


def test_carfax_adapter_maps_fields():
    from pipeline.sources import from_carfax
    out = from_carfax(carfax_raw())
    assert list(out.columns) == config.USED_COLUMNS
    assert out["listingId"].tolist() == ["a", "b", "c"]
    assert out["price"].tolist()[0] == 21_000 and pd.isna(out["price"].iloc[1])
    assert out["trim"].iloc[0] == "LE" and pd.isna(out["trim"].iloc[1]) and out["trim"].iloc[2] == "EX Sport"
    assert out["bodyStyle"].tolist() == ["sedan", "pickup_truck", "sedan"]
    # accidents: 0 = clean, NaN = unknown, 1 = reported (with a readable label)
    assert out["accidentCount"].iloc[0] == 0 and pd.isna(out["accidentCount"].iloc[1]) and out["accidentCount"].iloc[2] == 1
    assert out["accidentDetail"].iloc[2] == "Accident reported (minor to moderate damage)"
    assert out["ownerCount"].iloc[0] == 1 and pd.isna(out["ownerCount"].iloc[1]) and out["ownerCount"].iloc[2] == 3


def test_carfax_rows_clean_without_dev():
    from pipeline.sources import from_carfax
    out = clean(from_carfax(carfax_raw()), dev=False)
    assert out["listingId"].tolist() == ["a", "c"]   # 'b' has no One Price


def test_accident_label_uses_carfax_detail():
    row = car(accidentCount=1.0, accidentDetail="Accident reported (airbag deployed)")
    bd = model_breakdown(row, fake_phi(accidentCount=-0.1), math.log(20_000))
    assert "Accident reported (airbag deployed)" in [line["label"] for line in bd.lines]

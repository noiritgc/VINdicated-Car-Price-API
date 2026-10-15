import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from app import FRONTEND, app

c = TestClient(app)
CAMRY = {"make": "Toyota", "model": "Camry", "year": 2021, "mileage": 42000, "trim": "SE"}


def test_health():
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_metadata_has_catalog():
    j = c.get("/api/metadata").json()
    assert "Camry" in j["catalog"]["Toyota"]
    assert 0 < j["metrics"]["mape"] < 0.15


def test_predict_reasonable_price():
    j = c.post("/api/predict", json=CAMRY).json()
    assert 15_000 < j["predicted_price"] < 35_000
    assert j["price_range"]["low"] < j["predicted_price"] < j["price_range"]["high"]
    assert j["drivers"] and j["mileage_curve"] and j["warnings"] == []


def test_more_miles_never_costs_more():
    curve = c.post("/api/predict", json=CAMRY).json()["mileage_curve"]
    prices = [p["price"] for p in curve]
    assert prices == sorted(prices, reverse=True)


def test_accident_lowers_price():
    clean = c.post("/api/predict", json={**CAMRY, "accidentCount": 0}).json()["predicted_price"]
    hit = c.post("/api/predict", json={**CAMRY, "accidentCount": 1, "accidentSeverity": "severe"}).json()["predicted_price"]
    assert hit < clean


def test_newer_car_costs_more():
    old = c.post("/api/predict", json={**CAMRY, "year": 2015}).json()["predicted_price"]
    assert old < c.post("/api/predict", json=CAMRY).json()["predicted_price"]


def test_deal_verdict():
    j = c.post("/api/predict", json={**CAMRY, "askingPrice": 5000}).json()
    assert j["deal"]["verdict"] == "great deal"
    j = c.post("/api/predict", json={**CAMRY, "askingPrice": 90000}).json()
    assert j["deal"]["verdict"] == "overpriced"


def test_unknown_make_warns():
    j = c.post("/api/predict", json={**CAMRY, "make": "Zzz", "model": "Q"}).json()
    assert any("not in the training data" in w for w in j["warnings"])


def test_validation_errors():
    assert c.post("/api/predict", json={**CAMRY, "mileage": -1}).status_code == 422
    assert c.post("/api/predict", json={"make": "Toyota"}).status_code == 422


def test_batch():
    j = c.post("/api/predict/batch", json={"cars": [CAMRY, {**CAMRY, "year": 2018}]}).json()
    assert j["count"] == 2


@pytest.mark.skipif(not os.path.isdir(FRONTEND), reason="no frontend alongside the backend")
def test_frontend_served():
    r = c.get("/")
    assert r.status_code == 200 and "VINdicated" in r.text

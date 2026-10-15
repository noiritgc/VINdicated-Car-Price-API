"""
Car Price Predictor API (FastAPI)

Endpoints
  GET  /api/health              liveness + model info
  GET  /api/metadata            dropdown catalog (make -> model -> trims...), ranges, model metrics
  POST /api/predict             predict one car's price (+ range, drivers, mileage curve, deal verdict)
  POST /api/predict/batch       predict up to 500 cars at once
  GET  /                        test frontend
  GET  /docs                    interactive Swagger docs (auto-generated)

Run:  uvicorn app:app --reload --port 8000           (from inside backend/)
  or: uvicorn backend.app:app --reload --port 8000   (from the project root)

This folder is self-contained: copy backend/ into any project and run it.
Set CAR_API_FRONTEND to a folder with index.html to serve a frontend at "/".
"""
import json
import os
import sys
from typing import List, Literal, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import FEATURES, REFERENCE_YEAR, SEVERITY_NAMES, build_features  # noqa: E402

ART = os.path.join(HERE, "artifacts")
# Optional frontend: env var, else the sibling frontend/ folder in this repo (skipped if absent)
FRONTEND = os.environ.get("CAR_API_FRONTEND", os.path.join(os.path.dirname(HERE), "frontend"))

booster = lgb.Booster(model_file=os.path.join(ART, "model.txt"))
with open(os.path.join(ART, "meta.json")) as f:
    META = json.load(f)
VOCAB = META["vocab"]
Q_LO = META["interval_log_quantiles"]["p10"]
Q_HI = META["interval_log_quantiles"]["p90"]

# Human-readable names for the "what drove this price" breakdown
FEATURE_GROUPS = {
    "year": "Model year", "age": "Model year",
    "mileage": "Mileage", "log_mileage": "Mileage", "miles_per_year": "Mileage",
    "accidentCount": "Accident history", "accident_severity": "Accident history",
    "ownerCount": "Number of owners",
    "make": "Make & model", "model": "Make & model", "make_model": "Make & model",
    "trim": "Trim", "bodyStyle": "Body style", "drivetrain": "Drivetrain",
    "fuelType": "Fuel type", "engine": "Engine", "sellerState": "Seller location",
}

app = FastAPI(
    title="Car Price Predictor API",
    version="1.0.0",
    description="Predicts a used car's fair asking price from its specs and history. "
                f"Trained on {META['n_rows']:,} listings.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- schemas
class CarIn(BaseModel):
    make: str = Field(..., examples=["Toyota"])
    model: str = Field(..., examples=["Camry"])
    year: int = Field(..., ge=1980, le=REFERENCE_YEAR + 1, examples=[2021])
    mileage: float = Field(..., ge=0, le=500_000, examples=[42000])
    trim: Optional[str] = Field(None, examples=["SE"])
    bodyStyle: Optional[str] = Field(None, examples=["sedan"])
    drivetrain: Optional[str] = Field(None, examples=["Front-Wheel Drive"])
    fuelType: Optional[str] = Field(None, examples=["Gasoline"])
    engine: Optional[str] = Field(None, examples=["4 Cyl"])
    accidentCount: Optional[int] = Field(None, ge=0, le=10, examples=[0])
    accidentSeverity: Optional[Literal["none", "minor", "moderate", "severe", "airbag"]] = None
    ownerCount: Optional[int] = Field(None, ge=1, le=20, examples=[1])
    sellerState: Optional[str] = Field(None, min_length=2, max_length=2, examples=["CA"])
    askingPrice: Optional[float] = Field(None, gt=0, description="Optional: a listing's price to compare against")

    @field_validator("sellerState")
    @classmethod
    def upper_state(cls, v):
        return v.upper() if v else v


class Driver(BaseModel):
    factor: str
    effect_pct: float
    effect_usd: float


class Deal(BaseModel):
    askingPrice: float
    difference_usd: float
    difference_pct: float
    verdict: Literal["great deal", "good deal", "fair price", "above market", "overpriced"]


class PredictionOut(BaseModel):
    predicted_price: float
    price_range: dict
    currency: str = "USD"
    drivers: List[Driver]
    mileage_curve: List[dict]
    deal: Optional[Deal] = None
    warnings: List[str]
    input_used: dict


class BatchIn(BaseModel):
    cars: List[CarIn] = Field(..., min_length=1, max_length=500)


# --------------------------------------------------------------------------- helpers
def resolve(car: CarIn):
    """Fill blanks from the catalog's most common value for that make/model; collect warnings."""
    warnings = []
    d = car.model_dump()
    cat = META["catalog"].get(car.make, {}).get(car.model)
    if car.make not in VOCAB["make"]:
        warnings.append(f"Make '{car.make}' was not in the training data; estimate is less reliable.")
    elif cat is None:
        warnings.append(f"Model '{car.model}' was not seen for {car.make}; estimate leans on make-level patterns.")
    if cat:
        for field, key in [("bodyStyle", "bodyStyles"), ("drivetrain", "drivetrains"),
                           ("fuelType", "fuelTypes"), ("engine", "engines")]:
            if not d[field] and cat[key]:
                d[field] = cat[key][0]
        lo, hi = cat["years"]
        if not (lo - 1 <= car.year <= hi + 1):
            warnings.append(f"Training data has {car.make} {car.model} from {lo}–{hi}; {car.year} is outside that.")
    if d["trim"] and d["trim"] not in VOCAB["trim"]:
        warnings.append(f"Trim '{d['trim']}' not recognised; it was ignored.")
    if d["sellerState"] and d["sellerState"] not in VOCAB["sellerState"]:
        warnings.append(f"State '{d['sellerState']}' not in training data; it was ignored.")
    if car.mileage > META["ranges"]["mileage"][1]:
        warnings.append("Mileage is above anything in the training data.")

    sev = d.pop("accidentSeverity")
    if d["accidentCount"] == 0:
        sev_val = 0.0
    elif sev:
        sev_val = float(SEVERITY_NAMES[sev])
    elif d["accidentCount"]:
        sev_val = 1.5
    else:
        sev_val = np.nan
    d["accident_severity"] = sev_val
    d.pop("askingPrice")
    return d, warnings


def to_frame(rows):
    df = pd.DataFrame(rows)
    for c in ["trim", "bodyStyle", "drivetrain", "fuelType", "engine", "sellerState"]:
        df[c] = df[c].astype("object")
    return build_features(df, dict(VOCAB))


def verdict(diff_pct):
    if diff_pct <= -0.10: return "great deal"
    if diff_pct <= -0.04: return "good deal"
    if diff_pct < 0.05: return "fair price"
    if diff_pct < 0.12: return "above market"
    return "overpriced"


def predict_one(car: CarIn, detail=True) -> PredictionOut:
    row, warnings = resolve(car)
    X = to_frame([row])
    logp = float(booster.predict(X)[0])
    price = float(np.exp(logp))

    out = dict(
        predicted_price=round(price, -1),
        price_range={"low": round(price * np.exp(Q_LO), -1), "high": round(price * np.exp(Q_HI), -1),
                     "confidence": "80% of cars like this sell inside this range"},
        drivers=[], mileage_curve=[], warnings=warnings,
        input_used={k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()},
    )

    if detail:
        # SHAP-style contributions in log space -> % effect vs. the average car
        contrib = booster.predict(X, pred_contrib=True)[0][:-1]
        grouped = {}
        for f, c in zip(FEATURES, contrib):
            grouped[FEATURE_GROUPS[f]] = grouped.get(FEATURE_GROUPS[f], 0.0) + c
        drivers = []
        for name, c in sorted(grouped.items(), key=lambda kv: -abs(kv[1])):
            if abs(c) < 0.005:
                continue
            drivers.append(Driver(factor=name, effect_pct=round((np.exp(c) - 1) * 100, 1),
                                  effect_usd=round(price - price / np.exp(c), -1)))
        out["drivers"] = drivers[:7]

        grid = sorted(set([0, 10_000, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 200_000,
                           int(round(car.mileage, -3))]))
        curve_rows = [{**row, "mileage": m} for m in grid]
        curve = np.exp(booster.predict(to_frame(curve_rows)))
        out["mileage_curve"] = [{"mileage": m, "price": round(float(p), -1)} for m, p in zip(grid, curve)]

    if car.askingPrice:
        diff = car.askingPrice - price
        out["deal"] = Deal(askingPrice=car.askingPrice, difference_usd=round(diff, -1),
                           difference_pct=round(diff / price * 100, 1), verdict=verdict(diff / price))
    return PredictionOut(**out)


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"status": "ok", "model_trained_at": META["trained_at"], "training_rows": META["n_rows"],
            "trees": META["n_iterations"]}


@app.get("/api/metadata")
def metadata():
    return {
        "catalog": META["catalog"],
        "states": VOCAB["sellerState"],
        "bodyStyles": VOCAB["bodyStyle"], "drivetrains": VOCAB["drivetrain"],
        "fuelTypes": VOCAB["fuelType"], "engines": VOCAB["engine"],
        "accidentSeverities": META["accident_severities"],
        "ranges": META["ranges"],
        "metrics": META["cv_metrics"],
        "featureImportance": META["feature_importance"],
        "trainedAt": META["trained_at"], "trainingRows": META["n_rows"],
    }


@app.post("/api/predict", response_model=PredictionOut)
def predict(car: CarIn):
    try:
        return predict_one(car)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/predict/batch")
def predict_batch(body: BatchIn):
    resolved = [resolve(c) for c in body.cars]
    prices = np.exp(booster.predict(to_frame([r for r, _ in resolved])))
    return {"count": len(prices), "predictions": [
        {"index": i, "predicted_price": round(float(p), -1),
         "price_range": {"low": round(float(p * np.exp(Q_LO)), -1), "high": round(float(p * np.exp(Q_HI)), -1)},
         "warnings": w}
        for i, (p, (_, w)) in enumerate(zip(prices, resolved))]}


if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))

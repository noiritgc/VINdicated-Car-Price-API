# VINdicated Price Check: used-car price predictor

A machine-learning model that predicts a used car's fair asking price from its specs and history. It comes with a FastAPI backend and a browser frontend for trying it out.

- **Target:** `carfax_one_price` (the CARFAX "one price" dealer asking price, USD)
- **Data:** 21,532 listings in `data/training_data.csv`
- **Model:** LightGBM gradient-boosted trees trained on log(price). Price is constrained so that more miles or more accidents can only lower it.

## Accuracy (5-fold cross-validation, grouped by VIN)

| Metric | Value |
|---|---|
| Median error | ~3.8% |
| Mean absolute % error | ~6.1% |
| Within 10% of true price | ~84% |
| Within 20% of true price | ~95% |
| Mean absolute error | ~$1,840 |
| R² | 0.954 |

For comparison, a simple "median price for the same make/model/year" baseline gets about 10.4% MAPE, even when it is scored on the same data it was built from.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

**Backend only:** the `backend/` folder is self-contained (code, model files, requirements, Dockerfile, tests). Copy it into any project and see [backend/README.md](backend/README.md).

Open http://localhost:8000 for the frontend, or http://localhost:8000/docs for the interactive API docs.

> **macOS note:** LightGBM needs OpenMP. If you see a `libomp` error, run `brew install libomp`.

Docker (full app): `docker build -t car-price . && docker run -p 8000:8000 car-price`
Docker (API only): `docker build -t car-price-api backend && docker run -p 8000:8000 car-price-api`

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Status and model version |
| GET | `/api/metadata` | Valid makes → models → trims, states, accuracy metrics, feature importance |
| POST | `/api/predict` | One car: price, 80% range, price drivers, mileage curve, optional deal verdict |
| POST | `/api/predict/batch` | Up to 500 cars in one call |

Only `make`, `model`, `year` and `mileage` are required. If you leave out body style, drivetrain, fuel or engine, the API fills in the most common value for that make and model.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Camry","year":2021,"mileage":42000,"trim":"SE",
       "accidentCount":0,"ownerCount":1,"sellerState":"CA","askingPrice":22900}'
```

Example response (abridged):

```json
{
  "predicted_price": 23950,
  "price_range": {"low": 21880, "high": 26050},
  "drivers": [{"factor": "Trim", "effect_pct": -9.3, "effect_usd": -2450}, ...],
  "mileage_curve": [{"mileage": 0, "price": 28100}, ...],
  "deal": {"askingPrice": 22900, "difference_pct": -4.4, "verdict": "good deal"},
  "warnings": []
}
```

All fields: `make, model, year, mileage, trim, bodyStyle, drivetrain, fuelType, engine, accidentCount (0/1), accidentSeverity (none|minor|moderate|severe|airbag), ownerCount, sellerState (2-letter), askingPrice`.

## Retraining

```bash
python training/train.py --data data/training_data.csv
```

This writes `backend/artifacts/model.txt` and `meta.json`. The API loads both when it starts.

## Tests

```bash
pytest -q backend/tests
```

The tests cover the endpoints, input validation, and sanity checks: newer cars cost more, more miles never raises the price, and accidents lower the price.

## Project layout

```
backend/              Standalone API: copy this folder to reuse it anywhere
  app.py              FastAPI app (serves ../frontend too, if present)
  features.py         Feature engineering shared by training and the API
  artifacts/          Trained model + metadata
  tests/              API tests
  requirements.txt, Dockerfile, README.md
frontend/index.html   Test UI (single file, no build step)
training/train.py     Training + cross-validation (writes backend/artifacts/)
data/                 Training data
```

## Limits

- Prices are dealer asking prices from a snapshot taken in September 2026. Private-party prices usually run lower.
- Accuracy is weaker for rare makes and models, and for cars outside 2000–2026. The API returns `warnings` when that happens.
- The range is empirical: about 80% of held-out cars fell inside it.

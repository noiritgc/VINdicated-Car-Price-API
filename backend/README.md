# VINdicated Price Check API (backend)

Standalone FastAPI service that predicts a used car's fair asking price. Everything it needs is in this folder, so you can copy `backend/` into another project and run it on its own.

```
app.py            FastAPI app
features.py       Feature engineering (must match the one used in training)
artifacts/        model.txt (LightGBM) + meta.json (vocab, catalog, metrics)
tests/            pytest suite
requirements.txt  Dependencies
Dockerfile        Container for the API alone
```

## Run

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs. CORS is open, so any frontend can call it.

> **macOS:** LightGBM needs OpenMP. If you see a `libomp` error, run `brew install libomp`.

Docker: `docker build -t car-price-api . && docker run -p 8000:8000 car-price-api`

To also serve a web page at `/`, set `CAR_API_FRONTEND=/path/to/folder/with/index.html`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Status and model version |
| GET | `/api/metadata` | Valid makes → models → trims, states, accuracy metrics, feature importance |
| POST | `/api/predict` | One car: price, 80% range, price drivers, mileage curve, optional deal verdict |
| POST | `/api/predict/batch` | Up to 500 cars in one call |

Only `make`, `model`, `year` and `mileage` are required.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Camry","year":2021,"mileage":42000}'
```

## Tests

```bash
pytest -q tests
```

## Updating the model

Retraining lives in the main repo (`training/train.py`), which writes new files into `artifacts/`. Replace both `model.txt` and `meta.json` together.

# AutoLens — vehicle valuation API + test frontend

A working, locally runnable submission using the supplied vehicle datasets. The CatBoost model is **already trained and included**: no API key, cloud account, external inference service, or retraining is needed to run it. First-time dependency installation needs internet access.

## Run it

Use **Python 3.12** (the tested version).

### Windows

Extract this project, then double-click **start.bat**. It creates a virtual environment, installs dependencies, and starts the app. Open **http://127.0.0.1:8000**. Stop the server with Ctrl+C.

Or run these commands in PowerShell from the extracted `autolens` folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

Frontend: http://127.0.0.1:8000  
Interactive OpenAPI docs: http://127.0.0.1:8000/docs  
Alternative port: `python run.py --port 8001`

### Docker (configuration included; container build not tested here)

```bash
docker build -t autolens .
docker run --rm -p 127.0.0.1:8000:8000 autolens
```

## A two-minute demo

1. Open the frontend. It automatically selects a real vehicle from the held-out test set and calls the API.
2. Inspect the predicted price, calibrated range, reference-label error, strongest model influences, and comparable training vehicles.
3. Click **Try a held-out vehicle** to run inference on a fresh random test example.
4. Change mileage, trim, history, or asking price and click **Estimate vehicle value**. When you modify the sampled vehicle, its old reference label is no longer presented as ground truth for the modified input.
5. Try an unfamiliar make/model to see the limited-support warning and empty comparable list.
6. Open **Batch playground**, click **Load examples**, then **Run batch**.
7. Inspect **Live response**, download JSON, and open `/docs` for request validation and interactive calls.

The frontend and API are served together, so no frontend build step or CORS setup is necessary. The frontend has no external fonts, CDN libraries, or paid dependencies.

## API contract

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Model readiness and version |
| POST | `/api/v1/predict` | One valuation, range, SHAP influences, comparable vehicles, support warnings |
| POST | `/api/v1/predict/batch` | 1–100 valuations in a single request |
| GET | `/api/v1/cases?count=2&seed=42` | Reproducible random held-out cases with reference labels; omit seed for fresh examples |
| GET | `/api/v1/catalog?make=Toyota&model=Camry` | Training make/model/trim suggestions |
| GET | `/api/v1/metrics` | Test metrics, baseline comparisons, feature importance, segment errors, provenance |

`year`, `make`, `model`, and `mileage` are required. Other inputs are optional; missing history stays unknown, rather than being treated as zero accidents or one owner. Requests with unknown extra fields, invalid ranges, nonfinite numbers, or malformed JSON return HTTP 422. Requests receive an `X-Request-ID` and `X-Response-Time-Ms` header.

Optional fields: `trim`, `bodyStyle`, `drivetrain`, `fuelType`, `engine`, `sellerState`, `accidentCount`, `accidentDetail`, `ownerCount`, `askingPrice`. Field names match the provided training data. Make/model/trim and drivetrain aliases are normalized consistently during training and inference. `askingPrice` is only used for comparison and is never a model feature.

Example with curl, from this project directory:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  --data-binary @example-request.json
```

On Windows use `curl.exe` and put the command on one line. Alternatively, use `/docs`.

Minimal Python request:

```python
import json
from urllib.request import Request, urlopen

vehicle = {"year": 2022, "make": "Toyota", "model": "Camry", "mileage": 42000}
request = Request("http://127.0.0.1:8000/api/v1/predict",
                  data=json.dumps(vehicle).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request) as response:
    print(json.load(response))
```

See `example-request.json` and `example-response.json` for a complete real request/response. Batch body: `{"vehicles": [<vehicle>, <vehicle>]}`. The response preserves input order. `batch_processing_ms` is total processing time for the entire batch, not per-vehicle time.

## Model and honest evaluation

**Target:** `carfax_one_price`, the supplied CARFAX reference-price label. It is not verified transaction price and is not independently established fair market value. Asking-price comparisons are comparisons to this label prediction.

| Item | Result |
| --- | ---: |
| Input training rows | 21,532 |
| Unique eligible vehicles | 21,519 |
| Repeated vehicle rows removed | 13 |
| Training / calibration / test | 15,063 / 3,228 / 3,228 |
| Model test MAE | $2,421.37 |
| Model test RMSE | $5,304.99 |
| Model test R² | 0.9209 |
| Median absolute percentage error | 5.21% |
| Make/model median baseline MAE | $8,801.71 |
| Nominal range coverage | 90% |
| Observed test range coverage | 90.8% |

The model reduces MAE by approximately 72.5% versus the training-only make/model median baseline. Median percentage error is the median absolute percentage error across test vehicles; it does not mean every estimate is within 5.21%.

### Training decisions

- CatBoost regression on **log price**, 1,000 trees, depth 7, fixed hyperparameters.
- Numeric inputs: year, mileage, accidentCount, ownerCount.
- Categorical inputs: make, model, trim, bodyStyle, drivetrain, fuelType, engine, sellerState, accidentDetail.
- Keys, prices, seller badge, listing description, timestamps, and raw price-derived fields are excluded from model features. `age` is excluded because it duplicates year for this snapshot.
- First 17 characters of the supplied key identify vehicle groups. Repeated groups are deduplicated before splitting. Cross-split group overlap is asserted during training.
- Fixed random 70/15/15 split, seeds 42/43. Calibration and test rows are not used to fit or select the model.
- The range uses a finite-sample corrected 90th percentile of absolute log residuals on calibration rows. Test rows independently assess coverage.
- Unknown categorical values use an explicit unknown token; missing numeric history uses -1.
- Comparable retrieval uses only training vehicles, ranked by year difference, mileage difference, and trim match. It does not use the predicted price or test reference label to rank neighbors.
- The 26,497-row raw source provides listing context through a compact key/city/color/condition/service-record extract. It supplies no additional training labels. All 21,532 training keys matched that source. The full raw source is not duplicated in this package.

### Limits

This is a snapshot of the supplied 2026 data, with a random vehicle holdout rather than a future-time or geographic holdout. It does not establish accuracy on later markets, rare makes, or unfamiliar trims. The conformal range has marginal coverage under exchangeability, not a per-vehicle or per-segment guarantee. In particular, sparse and out-of-distribution inputs can have worse coverage. The app flags limited support and out-of-range year/mileage.

The training `accidentCount` contains only 0/1 among known values; it may represent a binary indicator rather than a full accident count. Values above 1 are accepted with an explicit extrapolation warning. Missing accident history does not prove no accidents. SHAP explanations describe model associations in log-price space and are not causal dollar impacts.

The demo defaults to localhost. For public deployment, put it behind your organization's authenticated gateway and configure rate limits, TLS, monitoring, and resource limits. No public deployment has been created.

## Reproduce / verify

From this project directory, inside your installed Python environment:

```bash
python train.py
python -m pytest -q
```

Training regenerates the model, metrics, comparable dataset and held-out demo cases. Original cleaned training data and the compact raw context extract are included. Training data SHA-256, library versions, segment errors, feature importance, baseline scores and interval details are retained in `artifacts/metrics.json`. Small floating-point differences across hardware or library versions are possible.

**Verification performed:** six API integration tests covering real inference, response ranges, deterministic samples, batch/single consistency, unknown categories, validation, aliases, price comparison, and validation of all held-out cases. The test suite passed. A real HTTP server smoke check also passed. Frontend JavaScript syntax and DOM ID references were checked; browser rendering and click-through were not verified because the browser download was unavailable in this environment. A 20-request in-process warm-model benchmark is in `artifacts/benchmark.json`; it is not a concurrency or network latency benchmark.

## Project layout

```text
autolens/
  app/main.py                 FastAPI endpoints and model service
  app/features.py             Shared training/serving transformations
  app/static/index.html       Responsive mock frontend and API playground
  artifacts/valuation.cbm     Already-trained CatBoost model
  artifacts/metrics.json      Independent held-out evaluation
  artifacts/comparables.json  Training-only reference examples
  artifacts/cases.json        Held-out demo cases
  data/                      Reproducible training inputs
  train.py                    Training, calibration, evaluation
  run.py                      Single-command local server
  start.bat                   Windows setup and launch
  tests/test_api.py            API integration tests
  example-request.json        Real request example
  example-response.json       Real response example
  requirements.txt            Pinned tested dependencies
  Dockerfile                  Optional container configuration
```

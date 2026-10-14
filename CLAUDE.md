# VINdicated Car Price API — Project Spec

**Scope: this repo is the API only.** Another team builds the frontend app and only calls our API.
We ship: the offline pipeline (data → model → rounds.json), the backend API, and a small mock
frontend (`mock_frontend/`, served at `/demo`) for testing and demos. No React Native app here.

## 1. Goal
Teach people what used cars are really worth so they don't overpay or get scammed.
Players see a real car with its situation (mileage, accidents, owners, title, state, seller type), guess the price, then see the fair price and exactly what pushed it up or down.

## 2. Architecture (one-way data flow)

```
Data files → pipeline/ (clean → train → price → build rounds) → rounds.json
          → backend/ (DB + API) → frontend (other team; mock_frontend/ here for testing)
```

- The pipeline runs offline, once (rerun when data changes). No ML runs live.
- Frontends only talk to our API. Never to outside data sources. No API keys on phones.
- The API contract (README "For the frontend team" + `/docs`) is the product. Don't break it.

## 3. Repo layout

```
/pipeline      Python: data cleaning, model training, round generation
  /data/raw    input files (gitignored)
  /data/out    rounds.json, price_model.cbm, report.md
  clean.py  train.py  evaluate.py  build_rounds.py  photos.py  run_all.py
  synthetic.py (--dev prices)  breakdown.py (SHAP → dollars)  scenarios.py  tips.py
  scam_templates.yaml  config.py  common.py
/backend       FastAPI + SQLAlchemy (SQLite locally, Postgres/Supabase in prod)
  main.py  db.py  models.py  scoring.py  settings.py  load_rounds.py
/mock_frontend index.html (demo client served at /demo)
/shared        round.schema.json (single source of truth)
/scripts       check_sample.py (validate a purchased data file)
/tests         pytest
```

## 4. Data sources

| Source | Use | Access |
|---|---|---|
| CarGurus listings (Rebrowser) | Main data: cars, prices, fair price, accidents, owners, state | Paid CSV from rebrowser.net. Free sample (prices redacted) on GitHub: `rebrowser/carguruscom-dataset`, `car-listings/data/*.csv` |
| ForCar open data | Title effects (salvage / rebuilt %) | Free CSV, forcar.org/open-data (CC BY 4.0) |
| Craigslist Used Cars (Kaggle `austinreese/craigslist-carstrucks-data`) | Private vs dealer discount only (URL contains `/cto/` = owner, `/ctd/` = dealer) | Free |
| Wikimedia Commons API | One photo per year/make/model | Free, store author + license |
| NHTSA / NICB / FTC guidance | Text for scam tips | Free, cite on reveal |

CarGurus columns we use:
`listingId, price, expectedPrice, dealRatingKey, mileage, vehicleCondition, year, make, model, trim, bodyStyle, drivetrain, fuelType, engine, accidentCount, ownerCount, sellerState, _firstSeenAt`

Free sample note: `price`, `expectedPrice`, `vin`, `images`, `listingUrl`, `dealScore` contain the string `[PREMIUM]` (100% of rows). Treat as missing. `--dev` mode fills synthetic prices so everything runs end-to-end before paid data arrives.

**Current data source (decided 2026-10-14): Rebrowser's free Carfax sample** (`github.com/rebrowser/carfax-dataset`,
26,497 used cars, Sept 2026) in `pipeline/data/raw/carfax/`. Its `onePrice` (CARFAX One Price value estimate) is
NOT redacted and is the training target. Why: buying CarGurus prices failed: a purchased 50-row export still had
`price`/`expectedPrice` empty (premium fields need a plan the user won't pay for), and rebrowser.net timed out (524) for
logged-in pages. Carfax facts: `listingId`/`vin` are premium, so `_primaryKey` is the ID; accidents = `noAccidents`
True (0) / `snapshotDamageSeverity` set (1, label from severity) / else unknown; owners = len(`ownerHistory`).
AutoTrader free sample (`raw/autotrader/`) has real KBB fair price ranges → `scripts/compare_kbb.py`.
CarGurus code path is kept (`--source cargurus`) in case paid data appears later.

CarGurus data facts (2026-10-13, free sample = 25,103 rows, Sept 2026, Hugging Face `rebrowser/carguruscom-dataset`):
- `priceDifferential` = expectedPrice − price (positive = good deal), despite the dataset's schema.json saying the opposite. Verified against `dealRatingKey`.
- All sample rows are `sellerType == DEALER`; private-seller rounds come from the scenario discount.

Licensing: CarGurus data is research / non-commercial license. Credit "Rebrowser" and "ForCar" on an About screen. No monetization without a commercial license.

## 5. Shared round schema (`/shared/round.schema.json`)

Every team builds against this. Agree on it day 1.

```json
{
  "round_id": "r_000123",
  "round_type": "price_guess",
  "difficulty": "medium",
  "car": { "year": 2019, "make": "Honda", "model": "Civic", "trim": "EX", "body_style": "sedan" },
  "photo": { "url": "https://upload.wikimedia.org/...", "credit": "Author, CC BY-SA 4.0" },
  "mileage": 82000,
  "state": "CA",
  "seller_type": "dealer",
  "tags": ["1 accident reported", "3 previous owners", "Clean title"],
  "asking_price": null,
  "fair_price": 16400,
  "breakdown": [
    { "label": "Base value (2019 Civic EX)", "amount": 19800 },
    { "label": "82k miles (above average)", "amount": -1900 },
    { "label": "1 accident reported", "amount": -1400 },
    { "label": "3 previous owners", "amount": -600 },
    { "label": "Sold in California", "amount": 500 }
  ],
  "price_hint_range": { "low": 7000, "high": 32000 },
  "tip": "Always pull a history report. One accident can cut 7–10% off the value.",
  "tip_source": "NHTSA",
  "tip_url": "https://...",
  "scam": null
}
```
(Abbreviated; `shared/round.schema.json` is authoritative. Added vs. original spec: `price_hint_range`,
`tip_url`, `photo.fallback` (body style when no photo), `base_value` {amount, label}, and scam fields below.)
`base_value` is PUBLIC (shown before guessing as the starting point) and must equal `breakdown[0].amount`.
Chosen over MSRP (2026-10-13): MSRP isn't in the data, and the model's base value is what the reveal starts from.

Rules:
- `breakdown` amounts must sum exactly to `fair_price` (base + adjustments). Round to nearest $100.
- `round_type`: `price_guess` | `deal_or_scam`.
- `difficulty`: `easy` | `medium` | `hard`.
- `seller_type`: `dealer` | `private`.
- For `deal_or_scam` rounds, `asking_price` is set and `scam` is:
  ```json
  { "template": "too_good_to_be_true", "is_scam": true,
    "seller_notes": ["Seller is moving abroad", "Wants a $500 deposit via their escrow link"],
    "red_flags": ["Price far below market", "..."],
    "correct_action": "walk_away", "true_value": 13100 }
  ```
  `correct_action`: `buy` | `negotiate` | `walk_away`.
  `seller_notes` are shown before guessing (for honest rounds too, so they don't give it away);
  `red_flags` (empty for honest rounds) are revealed after. For deal_or_scam, `fair_price` == `true_value`.

## 6. Pipeline (Python 3.11, pandas, catboost; CatBoost has SHAP built in, no extra library)

### 6.1 clean.py
- Load all CSVs, replace `[PREMIUM]` with NaN.
- Keep `vehicleCondition == USED`. Drop `dealRatingKey == OUTLIER`.
- Dedupe on `listingId` (keep latest `_firstSeenAt`).
- Drop: price missing, price < $1,500 or > $150,000, mileage missing or > 300k, year < 2000.
- Derive: `age = listing_year - year`, `miles_per_year = mileage / max(age, 1)`.
- Output `data/out/clean.parquet` + row counts at each step.

### 6.2 train.py
- Target: `log(price)`.
- Features: `make, model, trim, bodyStyle, drivetrain, fuelType, age, mileage, accidentCount, ownerCount, sellerState`. Categorical: make, model, trim, bodyStyle, drivetrain, fuelType, sellerState.
- CatBoostRegressor, RMSE loss, early stopping. 80/20 split by `listingId`, fixed seed.
- Save `price_model.cbm`.

### 6.3 evaluate.py → `report.md`
- On test set: MAE ($), MAPE (%), % within 10%, % within 20%.
- Same metrics for CarGurus `expectedPrice` vs actual price (benchmark).
- Error by make, by age bucket, by state. List the 20 worst predictions.
- Flag make/model combos with < 30 training rows → exclude from rounds.
- Target: median error ≤ 10%. If not met, tune and/or restrict rounds to well-covered models.

### 6.4 Breakdown from SHAP
`model.get_feature_importance(data=Pool(X), type='ShapValues')` gives log-space contributions φᵢ plus expected value E.
- Group features into player-facing buckets:
  - Base value = E + φ(make, model, trim, bodyStyle, drivetrain, fuelType, age)
  - Mileage = φ(mileage)
  - Accidents = φ(accidentCount)
  - Owners = φ(ownerCount)
  - Location = φ(sellerState)
- Convert to dollars sequentially (sums exactly):
  `running = exp(base)`; for each bucket b: `delta_b = running * (exp(φ_b) - 1)`; `running += delta_b`.
- Skip buckets with |delta| < $100 (fold into base). Round to $100 and fix rounding drift on the base line.
- Fair price = CatBoost prediction, cross-checked against `expectedPrice` when available (log if > 15% apart; prefer excluding that car).

### 6.5 scenarios.py (applied after the model, added as extra breakdown lines)
- Title: for a chosen subset, set tag "Salvage title" or "Rebuilt title" and apply ForCar's title % for that model (fallback: overall average). All others get "Clean title".
- Private seller: discount `d` = median of (private / dealer) price ratio from Craigslist, matched on make, model, year bucket, mileage bucket. Add line "Private-party sale".

### 6.6 Scam rounds (`deal_or_scam`)
- Hand-written template library in `scam_templates.yaml`, about 10 templates. Each: name, red_flags text, asking-price rule, true-value rule, correct_action, tip, source.
- Templates:
  - too_good_to_be_true: asking 35–50% below fair, deposit or escrow request → walk_away
  - odometer_rollback: shown miles vs. a clue with higher miles; true_value = model re-priced at real miles → walk_away
  - flood_title_wash: clean title claimed, moved from flood state; true_value uses salvage % → walk_away
  - curbstoning: "private" seller with many cars, title not in their name → walk_away
  - hidden_lien: no title in hand → walk_away
  - overpriced_dealer: real `OVERPRICED` listing, asking 15%+ over fair → negotiate
  - honest_private: fair asking price, no red flags → buy
  - fair_dealer: asking within 5% of fair → buy
- About 50% of `deal_or_scam` rounds must be honest (buy / negotiate) so players don't learn "private = scam".

### 6.7 build_rounds.py
- Pool: 800 rounds total.
  - At least 10 per state (take all if a state has fewer good rows).
  - Mix: 60% plain dealer, 15% title scenarios, 10% private, 15% `deal_or_scam`.
  - Oversample teaching cases: accidents > 0, owners ≥ 3, high miles, OVERPRICED.
- Split: 650 Quick Play, 150 Daily Challenge (no overlap).
- Difficulty: easy = common model, ≤ 1 notable tag, small adjustments; hard = rare trim, ≥ 3 notable tags, or large total |adjustments|; else medium. Thresholds in config.py were recalibrated on the data (spec values made ~75% hard).
- Tip: chosen by the largest |adjustment| bucket from `tips.py` (one tip text per bucket + per scam template).
- Validate every round against `round.schema.json`. Output `rounds.json`.

### 6.8 photos.py
- Query Wikimedia Commons per unique (year range, make, model):
  `https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch=<make model>&prop=imageinfo&iiprop=url|extmetadata&format=json`
- Store url + author + license. Cache results to `photos_cache.json`. Fallback: generic body-style illustration bundled in the app.

## 7. Backend (`/backend`, FastAPI + Postgres; host on Cloud Run, DB on Supabase free tier)

Tables:
- `rounds(round_id PK, pool ['quick'|'daily'], data JSONB)` — the answer fields are never returned before a guess.
- `players(player_id PK, created_at, current_streak, best_streak, last_daily_date)`
- `guesses(id PK, player_id, round_id, mode, guess, score, action, created_at)`
- `daily_schedule(date PK, round_id)` — pre-filled from the daily pool.

Endpoints:
- `POST /players` → new anonymous `player_id` (app stores it on device).
- `GET /rounds/quick?n=5` → 5 random quick-pool rounds, answer fields stripped (`fair_price`, `breakdown`, `tip`, `scam.*` except `red_flags`).
- `GET /rounds/daily?date=YYYY-MM-DD` → today's round, stripped.
- `POST /guesses {player_id, round_id, mode, guess, action?}` → full reveal: fair price, breakdown, tip, score.
- `GET /players/{id}` → streaks and stats.

Scoring (server side):
- `err = |guess - fair| / fair`
- `price_score = round(100 * max(0, 1 - err / 0.5))` → 100 if exact, 0 if 50%+ off.
- `deal_or_scam`: +50 for the correct action.
- Streak: +1 if daily played on consecutive days, else reset to 1.

Load script: `backend/load_rounds.py rounds.json` upserts rounds and fills `daily_schedule`.

## 8. Frontend (not in this repo)

The real app is built by another team against the API. This repo only has `mock_frontend/index.html`
(served at `/demo`): Home, Round, Reveal, Summary, using only our API. Keep it working as the
reference client whenever the API changes. Extra API endpoints added for the frontend team:
`GET /about` (credits, dev_mode), `GET /health`, `/docs` (OpenAPI). `GET /rounds/quick` also accepts
`player_id` (skip seen rounds), `difficulty`, `round_type`.

## 9. Six-week plan

| Week | Pipeline / ML | Backend | App (other team; we keep mock_frontend in sync) |
|---|---|---|---|
| 1 | Schema agreed. clean.py on free sample. Buy 50-row test | DB tables, `/players`, mock rounds loaded | Expo setup, navigation, Home with mocks |
| 2 | train.py + evaluate.py in `--dev` mode. Buy full data | `/rounds/quick`, `/rounds/daily`, `/guesses` | Round screen |
| 3 | Train on real data, report.md, SHAP breakdown | Scoring + streaks | Reveal screen |
| 4 | Scenarios, scam templates, tips, photos | Deploy to Cloud Run + Supabase | Summary, deal_or_scam UI |
| 5 | build_rounds.py → real rounds.json, spot-check 50 rounds by hand | Load real rounds, daily schedule | Connect to real API, polish |
| 6 | Fix bad rounds | Bug fixes | Testing, demo build |

## 10. Working rules for Claude Code
- Read this file before starting any task. If a change affects `round.schema.json`, update it and all three parts together.
- Python: type hints, small functions, `config.py` for all thresholds (no magic numbers in code).
- Never commit raw data or secrets. `.env` for DB URL and API base URL.
- Every pipeline script prints row counts and writes a log line so results are traceable.
- Tests: `pytest` for pipeline (breakdown sums to fair price, schema validation, no NaN) and backend (scoring, streaks, answers stripped from GET).

## 11. Decisions made / still open
- Decided: Quick Play avoids rounds a player already guessed (when `player_id` is passed), then allows repeats once exhausted.
- Decided: deal_or_scam = price guess against `true_value` (0–100) + 50 for the correct action (max 150).
- Decided: one daily guess per player per day (409 on replay); dailies more than 1 day ahead are hidden.
- Open: final scoring curve and round count per session (default n=5).
- Open: ForCar / Craigslist files. Loaders expect `pipeline/data/raw/forcar_title_effects.csv` (make, model, title_type, pct_change) and `craigslist_vehicles.csv` (Kaggle columns); until then config fallbacks are used.

# VINdicated Car Price API

**This repository is the backend API for a used-car pricing game.**
The game itself (the mobile/web app) is built by **another team** that only calls this API.
Everything here exists so that one HTTP call returns a ready-to-play round:

```
GET /rounds/quick?n=5   →  5 real used cars, with everything needed to show them (no answers)
POST /guesses           →  the fair price, a dollar-by-dollar breakdown, a score and a safety tip
```

Players see a real car (year, make, model, mileage, accidents, owners, title, state, seller type),
guess its price, then learn what pushed the price up or down. "Deal or Scam" rounds also teach
people to spot used-car scams.

> **Status:** fully working end to end on **real data**. Fair prices come from a CatBoost model
> trained on ~21,500 real **CARFAX One Price** value estimates (Rebrowser's free Carfax sample):
> 4.5% median error, 80% of cars within 10%. See `pipeline/data/out/report.md`.

---

## For the frontend team

- **Interactive docs (try every endpoint):** `/docs` (Swagger) or `/redoc`
- **Machine-readable spec:** `/openapi.json`
- **Working reference client:** `/demo` (a mock frontend in `mock_frontend/index.html`)
- **No API keys, no login.** Players are anonymous IDs you store on the device.
- **CORS** is open by default (`CORS_ORIGINS` env var).

### Typical flow

```
1. First launch:   POST /players                       → {"player_id": "..."}   (store it on device)
2. Quick Play:     GET  /rounds/quick?n=5&player_id=…  → {"rounds": [PublicRound × 5]}
3. Each guess:     POST /guesses                       → Reveal
4. Daily:          GET  /rounds/daily?player_id=…      → {"date", "round", "already_played"}
                   POST /guesses (mode "daily")        → Reveal (+ streak)
5. Profile:        GET  /players/{player_id}           → streaks and stats
```

### Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/players` | Create an anonymous player. Returns `player_id`. |
| `GET` | `/players/{player_id}` | Current/best streak, rounds played, total & average score, average % off. |
| `GET` | `/rounds/quick` | Random Quick Play rounds. Query: `n` (1–20, default 5), `player_id` (skip rounds they've played), `difficulty` (`easy`/`medium`/`hard`), `round_type` (`price_guess`/`deal_or_scam`). |
| `GET` | `/rounds/daily` | Today's daily challenge (UTC). Query: `date=YYYY-MM-DD` (up to 1 day ahead for time zones), `player_id` (adds `already_played`). |
| `POST` | `/guesses` | Submit a guess, get the full reveal. |
| `GET` | `/about` | App description, data credits (show on an About screen), whether data is dev. |
| `GET` | `/health` | Health check + number of rounds loaded. |

### PublicRound (what you get before guessing)

```json
{
  "round_id": "r_000713",
  "round_type": "deal_or_scam",
  "difficulty": "easy",
  "car": {"year": 2019, "make": "Subaru", "model": "Crosstrek", "trim": "2.0i Premium AWD", "body_style": "crossover"},
  "photo": {"url": null, "credit": null, "fallback": "crossover"},
  "mileage": 71462,
  "state": "NC",
  "seller_type": "private",
  "tags": ["No accidents reported", "1 previous owner", "Clean title"],
  "asking_price": 13400,
  "base_value": {"amount": 15600, "label": "Typical value of a 2019 Subaru Crosstrek 2.0i Premium AWD with average miles, history and location"},
  "price_hint_range": {"low": 7000, "high": 33000},
  "scam": {"seller_notes": ["Title is 'at the bank'; seller will mail it after you pay", "Wants cash today", "Car runs and drives great"]},
  "max_score": 150
}
```

- `photo.url` can be `null` → show your own generic image for `photo.fallback` (a body style like `sedan`, `suv`, `pickup_truck`). Always show `photo.credit` when present (license requirement).
- `base_value` is the **starting point** to show before the guess: what this year/make/model/trim is
  typically worth with **typical miles for its age**, average history and location. The player's job is to adjust it for this
  car's details. It always equals the first line of the revealed `breakdown`, so the reveal reads as
  "starting point → adjustments → fair price". A good default for the price slider.
- `price_hint_range` is the slider range. It's based on the model's typical price, not this car's answer.
- `asking_price` and `scam` are `null` for `price_guess` rounds.
- For `deal_or_scam` rounds, show `asking_price` and `scam.seller_notes`, and ask the player for **both** a price guess (what the car is really worth) **and** an action: Buy / Negotiate / Walk away. About half of these rounds are honest, so the notes are not always red flags.

### POST /guesses

```json
{"player_id": "…", "round_id": "r_000713", "mode": "quick", "guess": 12000, "action": "walk_away"}
```

- `mode`: `"quick"` or `"daily"` (must match where the round came from).
- `action`: required for `deal_or_scam`, must be omitted for `price_guess`. One of `buy`, `negotiate`, `walk_away`.

**Reveal response:**

```json
{
  "round_id": "r_000713", "round_type": "deal_or_scam", "mode": "quick",
  "guess": 12000, "fair_price": 15200, "difference": -3200, "pct_off": 21.1,
  "price_score": 58, "action": "walk_away", "action_correct": true, "action_score": 50,
  "score": 108, "max_score": 150, "asking_price": 13400,
  "breakdown": [
    {"label": "Base value (2019 Subaru Crosstrek 2.0i Premium AWD)", "amount": 15600},
    {"label": "71k miles", "amount": 200},
    {"label": "No accidents reported", "amount": 600},
    {"label": "1 previous owner", "amount": 400},
    {"label": "Sold in North Carolina", "amount": -300},
    {"label": "Private-party sale", "amount": -1300}
  ],
  "tip": "Never pay without the title in hand. Check the title and any brands in the national NMVTIS database first.",
  "tip_source": "NMVTIS", "tip_url": "https://vehiclehistory.bja.ojp.gov/",
  "scam": {"is_scam": true, "seller_notes": ["…"], "red_flags": ["No title in hand usually means a loan (lien) is still owed; you could lose the car", "Cash-only, same-day pressure"], "correct_action": "walk_away", "true_value": 15200},
  "streak": null
}
```

- `breakdown` always starts with the base value, and the amounts **sum exactly** to `fair_price`.
- `streak` (`{"current", "best"}`) is set for daily guesses.

### Scoring

- `price_score = round(100 × max(0, 1 − |guess − fair| / fair / 0.5))`: 100 if exact, 80 at 10% off, 0 at 50%+ off.
- `deal_or_scam`: **+50** for the correct action (max 150).
- Daily streak: +1 for consecutive days, a gap resets it to 1. `current_streak` shows 0 once a day is missed.

### Errors

Standard HTTP codes with `{"detail": "..."}`: `404` unknown player/round/no daily scheduled,
`409` daily already played, `422` invalid body (e.g. missing `action` on a deal_or_scam round),
`400` wrong mode for the round or a future daily.

---

## How it works

```
CarGurus listings → pipeline/ (clean → train → evaluate → build rounds) → rounds.json
                  → backend/ (load into DB, serve API) → frontend (other team)
```

No machine learning runs live. The pipeline runs offline and produces a fixed pool of
**800 rounds** (650 Quick Play, 150 Daily) that the API serves.

| Folder | What's in it |
|---|---|
| `pipeline/` | Data cleaning, CatBoost price model, SHAP price breakdowns, scenarios, scam templates, round builder, photo lookup |
| `backend/` | FastAPI app, database tables, scoring, `load_rounds.py` |
| `shared/round.schema.json` | The round format (single source of truth) |
| `mock_frontend/` | Demo page served at `/demo` (testing only) |
| `scripts/check_sample.py` | Checks a purchased data file has real prices and all needed columns |
| `tests/` | `pytest` for pipeline and backend |

### The price model

- **CatBoost** regression on `log(price)` with make, model, trim, body style, drivetrain, fuel type,
  age, mileage, accident count, owner count, state.
- **Mileage is judged against cars of the same age.** The mileage line is a what-if:
  `price(actual miles) − price(typical miles for its age)`, where typical miles come from the real
  data (new ≈ 6k, 1 yr ≈ 18k, 4 yrs ≈ 57k, 7 yrs ≈ 92k; see `pipeline/data/out/typical_miles.json`).
  So 20k miles on a 2026 car is a *minus*, and 20k on a 2022 car is a *plus*. The model also gets
  miles-per-year as an input (A/B tested: median error 4.47% → 4.37%).
- **Accident severity is measured, not guessed.** `pipeline/severity.py` cross-fits the model and
  measures how much each Carfax severity level lowers value: minor −2.0%, unspecified −2.3%,
  moderate −2.7%, severe/airbag −6.4% (90% confidence ranges in `report.md`). Fair price for a car
  with an accident = no-accident price × that effect. These are Carfax's valuations of mostly
  repaired, dealer-listed cars, which is why they're smaller than the 10–20% often quoted online.
- **Breakdown:** the typical-miles, no-accident version of the car is split with CatBoost's built-in
  SHAP values into *base value, owners, location*; accidents and mileage are what-if lines. All
  lines sum exactly to the fair price.
- **Scenarios** add lines on top: salvage/rebuilt title, private-party sale.
- **Deal or Scam:** 12 templates in `pipeline/scam_templates.yaml` (7 scams: too good to be true,
  gift card payment, odometer rollback, flood title wash, curbstoning, hidden lien, VIN mismatch;
  5 honest: overpriced dealer → negotiate, fair dealer, great deal, honest private seller,
  pricey private seller → negotiate). Half of these rounds are honest.
- Model accuracy is in `pipeline/data/out/report.md`: **4.5% median error** vs Carfax values on held-out
  cars, plus an independent check against **Kelley Blue Book** fair price ranges (median 4% above KBB's midpoint).

---

## Run it locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 1. Get the data (free, non-commercial): put the daily .parquet files from
#    github.com/rebrowser/carfax-dataset (car-listings/data/) in pipeline/data/raw/carfax/
#    Optional KBB check: github.com/rebrowser/autotrader-dataset -> pipeline/data/raw/autotrader/

# 2. Build rounds
python -m pipeline.run_all             # clean -> train -> severity -> evaluate -> build_rounds
python scripts/compare_kbb.py        # optional: compare with Kelley Blue Book ranges
python -m pipeline.photos            # optional, slow first time (cached afterwards)

# 3. Load rounds into the DB and start the API
python -m backend.load_rounds pipeline/data/out/rounds.json --reset
uvicorn backend.main:app --reload    # http://localhost:8000/docs and /demo

# Tests
pytest
```

### Data sources

| Source | Folder | Price used | Notes |
|---|---|---|---|
| **Carfax** (default) | `raw/carfax/` | `onePrice`: CARFAX One Price value estimate | Free sample has it unredacted. Accidents = reported / not (with damage severity); owners counted from owner history. |
| CarGurus | `raw/cargurus/` | `price` (asking price) | Price is a *premium* field (paid). `--source cargurus --dev` runs on the free sample with synthetic prices. |
| AutoTrader | `raw/autotrader/` | KBB fair price range | Only used by `scripts/compare_kbb.py` as an outside check. |

To use purchased CarGurus data instead: `python scripts/check_sample.py file.csv` (must print READY),
put it in `raw/cargurus/`, then `python -m pipeline.run_all --source cargurus`.

After any rebuild: `python -m backend.load_rounds pipeline/data/out/rounds.json --reset`.

Optional extra data, used automatically when present in `pipeline/data/raw/`:
- `forcar_title_effects.csv` (columns `make, model, title_type, pct_change`) → real salvage/rebuilt discounts
- `craigslist_vehicles.csv` (Kaggle `austinreese/craigslist-carstrucks-data`) → real private-party discount

Until then, the fallback values in `pipeline/config.py` are used (salvage −45%, rebuilt −30%, private −8%).

### Deploy (Cloud Run + Supabase)

The `Dockerfile` builds the API only (no ML libraries). Set `DATABASE_URL` to the Supabase Postgres
URL (`postgresql+psycopg://…`), run `load_rounds.py` once against it, then deploy the container.

---

## Data, credits and license

- Car values: **CARFAX One Price** estimates via **Rebrowser**'s Carfax dataset; KBB ranges via Rebrowser's
  AutoTrader dataset. Both **CC BY-NC 4.0** (non-commercial). No monetization without a commercial license.
- Title effects: **ForCar** open data (CC BY 4.0), when added.
- Photos: Wikimedia Commons contributors, credited per photo.
- Safety tips: FTC, NHTSA, NICB, NMVTIS (linked on every tip).

Raw data, trained models and `.env` are never committed (see `.gitignore`).

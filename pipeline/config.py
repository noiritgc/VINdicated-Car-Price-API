"""All pipeline paths, thresholds and constants. No magic numbers elsewhere."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_DIR = PIPELINE_DIR.parent
RAW_DIR = PIPELINE_DIR / "data" / "raw"
OUT_DIR = PIPELINE_DIR / "data" / "out"
SHARED_DIR = REPO_DIR / "shared"

CLEAN_PATH = OUT_DIR / "clean.parquet"
MODEL_PATH = OUT_DIR / "price_model.cbm"
SPLIT_PATH = OUT_DIR / "split.parquet"
COVERAGE_PATH = OUT_DIR / "coverage.json"
REPORT_PATH = OUT_DIR / "report.md"
ROUNDS_PATH = OUT_DIR / "rounds.json"
PHOTOS_CACHE_PATH = OUT_DIR / "photos_cache.json"
LOG_PATH = OUT_DIR / "pipeline.log"
SCHEMA_PATH = SHARED_DIR / "round.schema.json"
SCAM_TEMPLATES_PATH = PIPELINE_DIR / "scam_templates.yaml"

# Optional extra sources (used only if present in RAW_DIR)
CRAIGSLIST_PATH = RAW_DIR / "craigslist_vehicles.csv"
FORCAR_TITLE_PATH = RAW_DIR / "forcar_title_effects.csv"

SEED = 42

# ---------------------------------------------------------------- clean.py
REDACTED_TOKEN = "[PREMIUM]"
KEEP_CONDITIONS = ("USED",)
DROP_DEAL_RATINGS = ("OUTLIER",)
MIN_PRICE = 1_500
MAX_PRICE = 150_000
MAX_MILEAGE = 300_000
MIN_YEAR = 2000

USED_COLUMNS = [
    "listingId", "price", "expectedPrice", "priceDifferential", "dealRatingKey",
    "mileage", "vehicleCondition", "year", "make", "model", "trim", "bodyStyle",
    "drivetrain", "fuelType", "engine", "accidentCount", "ownerCount",
    "sellerState", "_firstSeenAt",
]
# Columns the paid file must have real values in (checked by scripts/check_sample.py)
REQUIRED_REAL_COLUMNS = ["price", "expectedPrice"]

# ---------------------------------------------------------------- synthetic prices (--dev only)
# Rough new-car price by body style, then multiplied by make tier etc.
DEV_BODY_BASE = {
    "sedan": 29_000, "hatchback": 25_000, "coupe": 34_000, "convertible": 42_000,
    "wagon": 33_000, "suv": 40_000, "crossover": 34_000, "minivan": 39_000,
    "pickup_truck": 47_000, "van": 42_000,
}
DEV_BODY_BASE_DEFAULT = 32_000
DEV_MAKE_TIER = {
    "Porsche": 2.4, "Maserati": 2.2, "Land Rover": 1.9, "Mercedes-Benz": 1.65, "BMW": 1.6,
    "Audi": 1.5, "Tesla": 1.5, "Lexus": 1.5, "Genesis": 1.4, "Cadillac": 1.45,
    "Lincoln": 1.4, "Volvo": 1.35, "Acura": 1.3, "INFINITI": 1.3, "Jaguar": 1.6,
    "Alfa Romeo": 1.4, "GMC": 1.15, "RAM": 1.1, "Toyota": 1.05, "Honda": 1.02,
    "Subaru": 1.0, "Mazda": 1.0, "Ford": 1.0, "Chevrolet": 0.98, "Jeep": 1.0,
    "Dodge": 0.97, "Chrysler": 0.95, "Buick": 1.0, "Volkswagen": 0.98,
    "Hyundai": 0.92, "Kia": 0.92, "Nissan": 0.92, "Mitsubishi": 0.82, "FIAT": 0.8,
    "MINI": 0.95,
}
DEV_MAKE_TIER_DEFAULT = 1.0
DEV_MODEL_SPREAD = 0.15          # +/- deterministic per-model variation
DEV_TRIM_PREMIUM_WORDS = {
    "Limited": 0.12, "Platinum": 0.2, "Denali": 0.22, "Premium": 0.08, "Touring": 0.08,
    "Sport": 0.04, "Lariat": 0.12, "King Ranch": 0.2, "High Country": 0.2, "Ultimate": 0.15,
    "Reserve": 0.15, "AMG": 0.35, "Turbo": 0.1, "SRT": 0.3, "TRD": 0.1, "Raptor": 0.35,
}
DEV_TRIM_BASE_WORDS = {"Base": -0.05, "S": -0.03, "LX": -0.03, "SE": -0.02, "L": -0.04}
DEV_DRIVETRAIN_ADJ = {"All-Wheel Drive": 0.06, "Four-Wheel Drive": 0.08}
DEV_FUEL_ADJ = {"Diesel": 0.1, "Hybrid": 0.05, "Electric": 0.0}
DEV_FIRST_YEAR_KEEP = 0.82        # value kept after year 1
DEV_YEARLY_KEEP = 0.88            # value kept each further year
DEV_MIN_VALUE_SHARE = 0.07
DEV_MILES_PER_YEAR = 12_000
DEV_MILEAGE_COEF = 3.2e-6        # log-value change per mile above expected
DEV_MILEAGE_CLIP = (0.6, 1.25)
DEV_ACCIDENT_KEEP = 0.91          # per reported accident
DEV_OWNER_PENALTY = 0.03          # per owner beyond the first
DEV_OWNER_PENALTY_CAP = 0.12
DEV_STATE_ADJ = {
    "CA": 0.04, "WA": 0.03, "OR": 0.02, "TX": 0.02, "CO": 0.02, "AZ": 0.02, "FL": 0.01,
    "NV": 0.02, "HI": 0.05, "AK": 0.05, "MI": -0.03, "OH": -0.02, "IN": -0.02,
    "IL": -0.01, "NY": -0.01, "PA": -0.02, "WI": -0.02, "MN": -0.01,
}
DEV_NOISE_SIGMA = 0.06
DEV_MIN_PRICE_SHARE_OF_EXPECTED = 0.5

# ---------------------------------------------------------------- train.py
CATEGORICAL_FEATURES = [
    "make", "model", "trim", "bodyStyle", "drivetrain", "fuelType", "sellerState",
]
NUMERIC_FEATURES = ["age", "mileage", "accidentCount", "ownerCount"]
FEATURES = [
    "make", "model", "trim", "bodyStyle", "drivetrain", "fuelType",
    "age", "mileage", "accidentCount", "ownerCount", "sellerState",
]
MISSING_CATEGORY = "Unknown"
TEST_SHARE = 0.2
VALID_SHARE_OF_TRAIN = 0.1
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "early_stopping_rounds": 150,
    "random_seed": SEED,
    "verbose": 500,
}

# ---------------------------------------------------------------- evaluate.py
MIN_TRAIN_ROWS_PER_MODEL = 30
TARGET_MEDIAN_ERROR = 0.10
WORST_N = 20
AGE_BUCKETS = [0, 3, 6, 10, 15, 30]

# ---------------------------------------------------------------- breakdown (SHAP)
BASE_FEATURES = ["make", "model", "trim", "bodyStyle", "drivetrain", "fuelType", "age"]
BUCKET_FEATURES = {
    "mileage": ["mileage"],
    "accidents": ["accidentCount"],
    "owners": ["ownerCount"],
    "location": ["sellerState"],
}
BUCKET_ORDER = ["mileage", "accidents", "owners", "location"]
MIN_LINE_DOLLARS = 100
ROUND_TO = 100
AVG_MILES_PER_YEAR = 12_000      # used for "(above/below average)" label
MILEAGE_LABEL_TOLERANCE = 0.15
MAX_EXPECTED_PRICE_GAP = 0.15    # model vs CarGurus expectedPrice cross-check

# ---------------------------------------------------------------- scenarios.py
# Fallbacks until ForCar / Craigslist files are dropped in RAW_DIR.
TITLE_DISCOUNTS = {"Salvage title": -0.45, "Rebuilt title": -0.30}
PRIVATE_PARTY_DISCOUNT = -0.08
CRAIGSLIST_MIN_PAIRS = 20
YEAR_BUCKET_SIZE = 3
MILEAGE_BUCKET_SIZE = 30_000

# ---------------------------------------------------------------- build_rounds.py
POOL_SIZE = 800
QUICK_POOL_SIZE = 650
DAILY_POOL_SIZE = 150
MIN_ROUNDS_PER_STATE = 10
ROUND_MIX = {"plain": 0.60, "title": 0.15, "private": 0.10, "deal_or_scam": 0.15}
HONEST_SHARE_OF_DEAL_OR_SCAM = 0.5
TEACHING_WEIGHTS = {"accident": 1.5, "many_owners": 1.0, "high_miles": 1.0, "overpriced": 1.0}
MANY_OWNERS = 3
HIGH_MILES = 120_000
# Difficulty thresholds calibrated on the data (spec values made ~75% of rounds "hard").
COMMON_MODEL_MIN_ROWS = 100
RARE_TRIM_MAX_ROWS = 3
EASY_MAX_TAGS = 1
EASY_MAX_ADJ_SHARE = 0.25
HARD_MIN_TAGS = 3
HARD_MIN_ADJ_SHARE = 0.45
PRICE_HINT_LOW = 0.4
PRICE_HINT_HIGH = 1.8
PRICE_HINT_ROUND = 1_000
PRICE_HINT_DEFAULT = (1_000, 80_000)

# ---------------------------------------------------------------- photos.py
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_USER_AGENT = "VINdicatedCarPriceAPI/0.1 (educational project; github.com/noiritgc/VINdicated-Car-Price-API)"
PHOTO_YEAR_BUCKET = 5
PHOTO_SEARCH_LIMIT = 10
PHOTO_ALLOWED_LICENSE_PREFIXES = ("CC BY", "CC0", "Public domain", "PD")
PHOTO_REQUEST_PAUSE_S = 1.0
PHOTO_MAX_RETRIES = 4
PHOTO_REJECT_WORDS = ("engine", "interior", "dashboard", "badge", "logo", "emblem", "wheel", "seat",
                      "trunk", "cockpit", "steering", "headlight", "taillight", "detail", "crash", "wreck")
PHOTO_BACKOFF_S = 10.0

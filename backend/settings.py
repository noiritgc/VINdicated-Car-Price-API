"""Backend settings (env) and game constants."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vindicated.db")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# Scoring
PRICE_SCORE_MAX = 100
PRICE_ZERO_AT_ERROR = 0.5      # 50%+ off scores 0
ACTION_BONUS = 50              # deal_or_scam: correct buy / negotiate / walk_away
MAX_GUESS = 1_000_000

# Rounds
QUICK_DEFAULT_N = 5
QUICK_MAX_N = 20
DAILY_MAX_DAYS_AHEAD = 1       # lets clients in time zones ahead of UTC fetch "today"

# Fields hidden until a guess is submitted
ANSWER_FIELDS = ("fair_price", "breakdown", "tip", "tip_source", "tip_url")
PUBLIC_SCAM_FIELDS = ("seller_notes",)

CREDITS = [
    "Car listing data: Rebrowser (CarGurus dataset), CC BY-NC 4.0",
    "Title effects: ForCar open data, CC BY 4.0",
    "Photos: Wikimedia Commons contributors (credited per photo)",
    "Safety tips: FTC, NHTSA, NICB, NMVTIS",
]

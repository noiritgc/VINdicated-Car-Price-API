"""Request/response shapes (pydantic). These drive the OpenAPI docs at /docs."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend import settings

Mode = Literal["quick", "daily"]
Action = Literal["buy", "negotiate", "walk_away"]


class Car(BaseModel):
    year: int
    make: str
    model: str
    trim: str | None
    body_style: str | None


class Photo(BaseModel):
    url: str | None
    credit: str | None
    fallback: str = Field(description="Body style; show a generic image for it when url is null")


class BaseValue(BaseModel):
    amount: int = Field(description="Starting point: typical value of this year/make/model/trim with average "
                                    "miles, history and location. Equals the first breakdown line.")
    label: str


class PriceRange(BaseModel):
    low: int
    high: int


class BreakdownLine(BaseModel):
    label: str
    amount: int


class PublicScam(BaseModel):
    seller_notes: list[str] = Field(description="Details about the seller/listing; some may be red flags")


class PublicRound(BaseModel):
    """A round as shown BEFORE guessing: no answers."""
    round_id: str
    round_type: Literal["price_guess", "deal_or_scam"]
    difficulty: Literal["easy", "medium", "hard"]
    car: Car
    photo: Photo
    mileage: int
    state: str
    seller_type: Literal["dealer", "private"]
    tags: list[str]
    asking_price: int | None
    base_value: BaseValue
    price_hint_range: PriceRange
    scam: PublicScam | None
    max_score: int


class QuickRoundsResponse(BaseModel):
    rounds: list[PublicRound]


class DailyRoundResponse(BaseModel):
    date: date
    round: PublicRound
    already_played: bool = False


class PlayerCreated(BaseModel):
    player_id: str


class PlayerStats(BaseModel):
    player_id: str
    created_at: datetime
    current_streak: int
    best_streak: int
    last_daily_date: date | None
    rounds_played: int
    total_score: int
    average_score: float | None
    average_pct_off: float | None


class GuessIn(BaseModel):
    player_id: str
    round_id: str
    mode: Mode
    guess: int = Field(gt=0, le=settings.MAX_GUESS, description="Guessed price in dollars")
    action: Action | None = Field(None, description="Required for deal_or_scam rounds")


class ScamReveal(BaseModel):
    is_scam: bool
    seller_notes: list[str]
    red_flags: list[str]
    correct_action: Action
    true_value: int


class StreakOut(BaseModel):
    current: int
    best: int


class Reveal(BaseModel):
    """Everything shown AFTER guessing."""
    round_id: str
    round_type: str
    mode: Mode
    guess: int
    fair_price: int
    difference: int = Field(description="guess - fair_price")
    pct_off: float = Field(description="|guess - fair| / fair, as a percentage")
    price_score: int
    action: Action | None
    action_correct: bool | None
    action_score: int
    score: int
    max_score: int
    asking_price: int | None
    breakdown: list[BreakdownLine]
    tip: str
    tip_source: str
    tip_url: str | None
    scam: ScamReveal | None
    streak: StreakOut | None = Field(None, description="Set for daily mode")


class About(BaseModel):
    name: str
    description: str
    dev_mode: bool | None
    round_counts: dict
    credits: list[str]

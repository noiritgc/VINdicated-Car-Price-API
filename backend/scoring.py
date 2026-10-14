"""Pure scoring and streak rules (no DB), so they're easy to test."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from backend import settings


def price_error(guess: int, fair: int) -> float:
    return abs(guess - fair) / fair


def price_score(guess: int, fair: int) -> int:
    """100 for an exact guess, falling linearly to 0 at 50% off."""
    err = price_error(guess, fair)
    return round(settings.PRICE_SCORE_MAX * max(0.0, 1 - err / settings.PRICE_ZERO_AT_ERROR))


def action_score(action: str | None, correct: str | None) -> int:
    return settings.ACTION_BONUS if action is not None and action == correct else 0


def max_score(round_type: str) -> int:
    bonus = settings.ACTION_BONUS if round_type == "deal_or_scam" else 0
    return settings.PRICE_SCORE_MAX + bonus


@dataclass
class Streak:
    current: int
    best: int
    last_daily_date: date | None


def update_streak(streak: Streak, played: date) -> Streak:
    """Apply a daily play on date `played`. Consecutive days +1, a gap resets to 1.
    Playing an older daily (catch-up) doesn't change the streak."""
    last = streak.last_daily_date
    if last is not None and played <= last:
        return streak
    current = streak.current + 1 if last is not None and played - last == timedelta(days=1) else 1
    return Streak(current=current, best=max(streak.best, current), last_daily_date=played)


def effective_streak(streak: Streak, today: date) -> int:
    """Streak to display: 0 if the player missed yesterday's daily."""
    last = streak.last_daily_date
    if last is None or today - last > timedelta(days=1):
        return 0
    return streak.current

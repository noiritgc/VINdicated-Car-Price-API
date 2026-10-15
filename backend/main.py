"""VINdicated Car Price API.

Run locally:  uvicorn backend.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import random
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend import scoring, settings
from backend.db import DailySchedule, Guess, Meta, Player, Round, get_session, init_db
from backend.models import (About, DailyRoundResponse, GuessIn, PlayerCreated, PlayerStats,
                            PublicRound, QuickRoundsResponse, Reveal, StreakOut)

DEMO_DIR = Path(__file__).resolve().parent.parent / "mock_frontend"
DB = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="VINdicated Car Price API",
    version="1.0.0",
    description="Serves used-car pricing game rounds and scores guesses. "
                "GET endpoints never include answers; POST /guesses returns the full reveal.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])
if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=DEMO_DIR, html=True), name="demo")


# ---------------------------------------------------------------- helpers
def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def public_round(data: dict) -> dict:
    """Strip every answer field from a full round."""
    out = {k: v for k, v in data.items() if k not in settings.ANSWER_FIELDS}
    if data.get("scam"):
        out["scam"] = {k: data["scam"][k] for k in settings.PUBLIC_SCAM_FIELDS}
    out["max_score"] = scoring.max_score(data["round_type"])
    return out


def get_player(db: Session, player_id: str) -> Player:
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(404, "Unknown player_id. Create one with POST /players.")
    return player


def get_round(db: Session, round_id: str) -> Round:
    rnd = db.get(Round, round_id)
    if rnd is None:
        raise HTTPException(404, f"Unknown round_id {round_id}")
    return rnd


# ---------------------------------------------------------------- meta
@app.get("/health", tags=["meta"])
def health(db: DB) -> dict:
    return {"status": "ok", "rounds_loaded": db.scalar(select(func.count()).select_from(Round))}


@app.get("/about", response_model=About, tags=["meta"])
def about(db: DB) -> About:
    meta = db.get(Meta, "pipeline")
    value = meta.value if meta else {}
    return About(
        name="VINdicated",
        description="Learn what used cars are really worth: guess the price, then see what moved it.",
        dev_mode=value.get("dev_mode"),
        data_source=value.get("source"),
        round_counts=value.get("counts", {}),
        credits=settings.CREDITS,
    )


# ---------------------------------------------------------------- players
@app.post("/players", response_model=PlayerCreated, status_code=201, tags=["players"])
def create_player(db: DB) -> PlayerCreated:
    player = Player(player_id=str(uuid.uuid4()))
    db.add(player)
    db.commit()
    return PlayerCreated(player_id=player.player_id)


@app.get("/players/{player_id}", response_model=PlayerStats, tags=["players"])
def player_stats(player_id: str, db: DB) -> PlayerStats:
    player = get_player(db, player_id)
    played, total = db.execute(
        select(func.count(Guess.id), func.coalesce(func.sum(Guess.score), 0))
        .where(Guess.player_id == player_id)).one()
    rows = db.execute(select(Guess.guess, Round.data).join(Round, Guess.round_id == Round.round_id)
                      .where(Guess.player_id == player_id)).all()
    pct = [scoring.price_error(g, d["fair_price"]) * 100 for g, d in rows]
    streak = scoring.Streak(player.current_streak, player.best_streak, player.last_daily_date)
    return PlayerStats(
        player_id=player.player_id,
        created_at=player.created_at,
        current_streak=scoring.effective_streak(streak, utc_today()),
        best_streak=player.best_streak,
        last_daily_date=player.last_daily_date,
        rounds_played=played,
        total_score=total,
        average_score=round(total / played, 1) if played else None,
        average_pct_off=round(sum(pct) / len(pct), 1) if pct else None,
    )


# ---------------------------------------------------------------- rounds
@app.get("/rounds/quick", response_model=QuickRoundsResponse, tags=["rounds"])
def quick_rounds(
    db: DB,
    n: Annotated[int, Query(ge=1, le=settings.QUICK_MAX_N)] = settings.QUICK_DEFAULT_N,
    player_id: Annotated[str | None, Query(description="Avoid rounds this player already played")] = None,
    difficulty: Annotated[str | None, Query(pattern="^(easy|medium|hard)$")] = None,
    round_type: Annotated[str | None, Query(pattern="^(price_guess|deal_or_scam)$")] = None,
) -> QuickRoundsResponse:
    query = select(Round.round_id).where(Round.pool == "quick")
    if difficulty:
        query = query.where(Round.difficulty == difficulty)
    if round_type:
        query = query.where(Round.round_type == round_type)
    ids = list(db.scalars(query))
    if not ids:
        raise HTTPException(404, "No rounds match. Have rounds been loaded (backend/load_rounds.py)?")

    seen: set[str] = set()
    if player_id:
        seen = set(db.scalars(select(Guess.round_id).where(Guess.player_id == player_id)))
    fresh = [i for i in ids if i not in seen]
    random.shuffle(fresh)
    picked = fresh[:n]
    if len(picked) < n:  # player has seen everything: allow repeats
        repeats = [i for i in ids if i not in picked]
        picked += random.sample(repeats, min(n - len(picked), len(repeats)))
    rounds = [public_round(db.get(Round, i).data) for i in picked]
    return QuickRoundsResponse(rounds=[PublicRound(**r) for r in rounds])


@app.get("/rounds/daily", response_model=DailyRoundResponse, tags=["rounds"])
def daily_round(
    db: DB,
    date_: Annotated[date | None, Query(alias="date", description="YYYY-MM-DD, defaults to today (UTC)")] = None,
    player_id: str | None = None,
) -> DailyRoundResponse:
    day = date_ or utc_today()
    if day > utc_today() + timedelta(days=settings.DAILY_MAX_DAYS_AHEAD):
        raise HTTPException(400, "That daily challenge isn't available yet.")
    entry = db.get(DailySchedule, day)
    if entry is None:
        raise HTTPException(404, f"No daily challenge scheduled for {day}")
    played = False
    if player_id:
        played = db.scalar(select(func.count(Guess.id)).where(
            Guess.player_id == player_id, Guess.round_id == entry.round_id, Guess.mode == "daily")) > 0
    rnd = get_round(db, entry.round_id)
    return DailyRoundResponse(date=day, round=PublicRound(**public_round(rnd.data)), already_played=played)


# ---------------------------------------------------------------- guesses
@app.post("/guesses", response_model=Reveal, tags=["guesses"])
def submit_guess(body: GuessIn, db: DB) -> Reveal:
    player = get_player(db, body.player_id)
    rnd = get_round(db, body.round_id)
    data = rnd.data
    is_dos = data["round_type"] == "deal_or_scam"
    if is_dos and body.action is None:
        raise HTTPException(422, "deal_or_scam rounds need an action: buy, negotiate or walk_away")
    if not is_dos and body.action is not None:
        raise HTTPException(422, "action is only for deal_or_scam rounds")

    daily_date = None
    if body.mode == "daily":
        daily_date = db.scalar(select(DailySchedule.date).where(DailySchedule.round_id == rnd.round_id))
        if daily_date is None:
            raise HTTPException(400, "This round is not a daily challenge")
        if daily_date > utc_today() + timedelta(days=settings.DAILY_MAX_DAYS_AHEAD):
            raise HTTPException(400, "That daily challenge isn't available yet.")
        dup = db.scalar(select(func.count(Guess.id)).where(
            Guess.player_id == player.player_id, Guess.round_id == rnd.round_id, Guess.mode == "daily"))
        if dup:
            raise HTTPException(409, "You already played this daily challenge")
    elif rnd.pool != "quick":
        raise HTTPException(400, "Daily rounds can only be played in daily mode")

    fair = data["fair_price"]
    p_score = scoring.price_score(body.guess, fair)
    correct = data["scam"]["correct_action"] if is_dos else None
    a_score = scoring.action_score(body.action, correct)
    score = p_score + a_score
    db.add(Guess(player_id=player.player_id, round_id=rnd.round_id, mode=body.mode,
                 guess=body.guess, score=score, action=body.action))

    streak_out = None
    if daily_date is not None:
        s = scoring.update_streak(
            scoring.Streak(player.current_streak, player.best_streak, player.last_daily_date), daily_date)
        player.current_streak, player.best_streak, player.last_daily_date = s.current, s.best, s.last_daily_date
        streak_out = StreakOut(current=scoring.effective_streak(s, utc_today()), best=s.best)
    db.commit()

    return Reveal(
        round_id=rnd.round_id,
        round_type=data["round_type"],
        mode=body.mode,
        guess=body.guess,
        fair_price=fair,
        difference=body.guess - fair,
        pct_off=round(scoring.price_error(body.guess, fair) * 100, 1),
        price_score=p_score,
        action=body.action,
        action_correct=(body.action == correct) if is_dos else None,
        action_score=a_score,
        score=score,
        max_score=scoring.max_score(data["round_type"]),
        asking_price=data["asking_price"],
        breakdown=data["breakdown"],
        tip=data["tip"],
        tip_source=data["tip_source"],
        tip_url=data.get("tip_url"),
        scam={k: v for k, v in data["scam"].items() if k != "template"} if is_dos else None,
        streak=streak_out,
    )

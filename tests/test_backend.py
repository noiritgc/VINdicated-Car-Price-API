"""Backend tests: scoring, streaks, answers stripped from GETs, guess flows."""
from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend import main, scoring, settings
from backend.db import DailySchedule, Round, get_session, init_db, make_engine

PRICE_ROUND = {
    "round_id": "r_000001", "round_type": "price_guess", "difficulty": "easy",
    "car": {"year": 2019, "make": "Honda", "model": "Civic", "trim": "EX", "body_style": "sedan"},
    "photo": {"url": None, "credit": None, "fallback": "sedan"},
    "mileage": 82000, "state": "CA", "seller_type": "dealer",
    "tags": ["1 accident reported", "3 previous owners", "Clean title"],
    "asking_price": None, "price_hint_range": {"low": 7000, "high": 32000},
    "base_value": {"amount": 19800, "label": "Typical value of a 2019 Honda Civic EX with average miles, history and location"},
    "fair_price": 16400,
    "breakdown": [{"label": "Base value", "amount": 19800}, {"label": "82k miles", "amount": -3400}],
    "tip": "Always pull a history report.", "tip_source": "FTC", "tip_url": "https://example.org",
    "scam": None,
}
SCAM_ROUND = {
    **copy.deepcopy(PRICE_ROUND), "round_id": "r_000002", "round_type": "deal_or_scam",
    "asking_price": 8000, "seller_type": "private",
    "scam": {"template": "too_good_to_be_true", "is_scam": True, "seller_notes": ["Moving abroad"],
             "red_flags": ["Price far below market"], "correct_action": "walk_away", "true_value": 16400},
}
DAILY_ROUND = {**copy.deepcopy(PRICE_ROUND), "round_id": "r_000003"}
FUTURE_ROUND = {**copy.deepcopy(PRICE_ROUND), "round_id": "r_000004"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    today = main.utc_today()
    with Session() as db:
        for r, pool in [(PRICE_ROUND, "quick"), (SCAM_ROUND, "quick"), (DAILY_ROUND, "daily"), (FUTURE_ROUND, "daily")]:
            db.add(Round(round_id=r["round_id"], pool=pool, round_type=r["round_type"],
                         difficulty=r["difficulty"], data=r))
        db.add(DailySchedule(date=today, round_id="r_000003"))
        db.add(DailySchedule(date=today + timedelta(days=5), round_id="r_000004"))
        db.commit()

    def session_override():
        with Session() as s:
            yield s

    main.app.dependency_overrides[get_session] = session_override
    monkeypatch.setattr(main, "init_db", lambda: None)
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


def new_player(client) -> str:
    return client.post("/players").json()["player_id"]


def guess(client, player, round_id, mode="quick", value=16400, action=None):
    body = {"player_id": player, "round_id": round_id, "mode": mode, "guess": value}
    if action:
        body["action"] = action
    return client.post("/guesses", json=body)


# ---------------------------------------------------------------- scoring (pure)
@pytest.mark.parametrize("g, expected", [(16400, 100), (16400 * 1.25, 50), (16400 * 0.5, 0), (16400 * 3, 0), (16400 * 0.9, 80)])
def test_price_score_curve(g, expected):
    assert scoring.price_score(int(g), 16400) == expected


def test_action_score():
    assert scoring.action_score("walk_away", "walk_away") == settings.ACTION_BONUS
    assert scoring.action_score("buy", "walk_away") == 0
    assert scoring.action_score(None, None) == 0


def test_streak_rules():
    d = date(2026, 10, 1)
    s = scoring.Streak(0, 0, None)
    s = scoring.update_streak(s, d)
    assert (s.current, s.best) == (1, 1)
    s = scoring.update_streak(s, d + timedelta(days=1))
    assert (s.current, s.best) == (2, 2)
    s = scoring.update_streak(s, d + timedelta(days=4))          # missed days -> reset
    assert (s.current, s.best) == (1, 2)
    assert scoring.update_streak(s, d) == s                      # catch-up play doesn't change it
    assert scoring.effective_streak(s, d + timedelta(days=5)) == 1
    assert scoring.effective_streak(s, d + timedelta(days=6)) == 0


# ---------------------------------------------------------------- answers never leak
ANSWER_KEYS = {"fair_price", "breakdown", "tip", "tip_source", "tip_url"}


def test_quick_rounds_strip_answers(client):
    rounds = client.get("/rounds/quick?n=5").json()["rounds"]
    assert {r["round_id"] for r in rounds} == {"r_000001", "r_000002"}   # quick pool only
    for r in rounds:
        assert not ANSWER_KEYS & r.keys()
        assert r["base_value"]["amount"] == 19800       # starting point is shown before guessing
        if r["scam"]:
            assert r["scam"] == {"seller_notes": ["Moving abroad"]}
            assert r["max_score"] == 150


def test_daily_round_strips_answers(client):
    body = client.get("/rounds/daily").json()
    assert body["round"]["round_id"] == "r_000003"
    assert not ANSWER_KEYS & body["round"].keys()


def test_future_daily_is_hidden(client):
    future = (main.utc_today() + timedelta(days=5)).isoformat()
    assert client.get(f"/rounds/daily?date={future}").status_code == 400


# ---------------------------------------------------------------- guesses
def test_quick_guess_reveals_answer(client):
    p = new_player(client)
    r = guess(client, p, "r_000001", value=18040).json()
    assert r["fair_price"] == 16400 and r["price_score"] == 80 and r["score"] == 80
    assert r["breakdown"][0]["amount"] == 19800 and r["streak"] is None


def test_deal_or_scam_needs_action_and_scores_bonus(client):
    p = new_player(client)
    assert guess(client, p, "r_000002").status_code == 422
    r = guess(client, p, "r_000002", action="walk_away").json()
    assert r["score"] == 150 and r["action_correct"] is True
    assert r["scam"]["red_flags"] == ["Price far below market"]
    assert "template" not in r["scam"]


def test_action_rejected_on_price_round(client):
    assert guess(client, new_player(client), "r_000001", action="buy").status_code == 422


def test_daily_flow_streak_and_single_play(client):
    p = new_player(client)
    r = guess(client, p, "r_000003", mode="daily").json()
    assert r["streak"] == {"current": 1, "best": 1}
    assert guess(client, p, "r_000003", mode="daily").status_code == 409
    assert client.get(f"/rounds/daily?player_id={p}").json()["already_played"] is True
    stats = client.get(f"/players/{p}").json()
    assert stats["current_streak"] == 1 and stats["rounds_played"] == 1 and stats["total_score"] == 100


def test_daily_round_not_playable_in_quick_mode(client):
    assert guess(client, new_player(client), "r_000003", mode="quick").status_code == 400


def test_future_daily_not_playable(client):
    assert guess(client, new_player(client), "r_000004", mode="daily").status_code == 400


def test_unknown_player_and_round(client):
    assert guess(client, "nope", "r_000001").status_code == 404
    assert guess(client, new_player(client), "r_999999").status_code == 404


def test_quick_avoids_seen_rounds(client):
    p = new_player(client)
    guess(client, p, "r_000001")
    rounds = client.get(f"/rounds/quick?n=1&player_id={p}").json()["rounds"]
    assert rounds[0]["round_id"] == "r_000002"

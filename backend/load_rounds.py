"""Load rounds.json into the database and fill the daily schedule.

Usage:
    python -m backend.load_rounds pipeline/data/out/rounds.json
    python -m backend.load_rounds pipeline/data/out/rounds.json --reset   # wipe all tables first
                                                                         # (do this when switching dev -> real data)
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.db import Base, DailySchedule, Meta, Round, SessionLocal, engine, init_db


def upsert_rounds(db: Session, payload: dict) -> int:
    pool_of = {rid: pool for pool, ids in payload["pools"].items() for rid in ids}
    for r in payload["rounds"]:
        db.merge(Round(round_id=r["round_id"], pool=pool_of[r["round_id"]],
                       round_type=r["round_type"], difficulty=r["difficulty"], data=r))
    db.merge(Meta(key="pipeline", value=payload["meta"]))
    return len(payload["rounds"])


def fill_schedule(db: Session, daily_ids: list[str], start: date, seed: int = 42) -> int:
    """Keep past dates; (re)assign today onward using daily rounds not used in the past."""
    db.execute(delete(DailySchedule).where(DailySchedule.date >= start))
    used = set(db.scalars(select(DailySchedule.round_id)))
    remaining = [rid for rid in daily_ids if rid not in used]
    random.Random(seed).shuffle(remaining)
    for offset, rid in enumerate(remaining):
        db.add(DailySchedule(date=start + timedelta(days=offset), round_id=rid))
    return len(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounds_json", type=Path)
    parser.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="first daily date (default: today UTC)")
    args = parser.parse_args()

    payload = json.loads(args.rounds_json.read_text())
    if args.reset:
        Base.metadata.drop_all(engine)
    init_db()
    start = args.start or datetime.now(timezone.utc).date()
    with SessionLocal() as db:
        n = upsert_rounds(db, payload)
        db.flush()
        days = fill_schedule(db, payload["pools"]["daily"], start)
        db.commit()
    print(f"loaded {n} rounds; scheduled {days} daily challenges from {start} "
          f"to {start + timedelta(days=days - 1)} (dev_mode={payload['meta']['dev_mode']})")


if __name__ == "__main__":
    main()

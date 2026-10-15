"""Database tables (SQLAlchemy). SQLite locally, Postgres in production."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend import settings

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Round(Base):
    __tablename__ = "rounds"
    round_id: Mapped[str] = mapped_column(String, primary_key=True)
    pool: Mapped[str] = mapped_column(String, index=True)          # quick | daily
    round_type: Mapped[str] = mapped_column(String, index=True)
    difficulty: Mapped[str] = mapped_column(String, index=True)
    data: Mapped[dict] = mapped_column(JSONType)


class Player(Base):
    __tablename__ = "players"
    player_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    best_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_daily_date: Mapped[date | None] = mapped_column(Date, nullable=True)


class Guess(Base):
    __tablename__ = "guesses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"), index=True)
    round_id: Mapped[str] = mapped_column(ForeignKey("rounds.round_id"), index=True)
    mode: Mapped[str] = mapped_column(String)                       # quick | daily
    guess: Mapped[int] = mapped_column(Integer)
    score: Mapped[int] = mapped_column(Integer)
    action: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DailySchedule(Base):
    __tablename__ = "daily_schedule"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    round_id: Mapped[str] = mapped_column(ForeignKey("rounds.round_id"))


class Meta(Base):
    """Key/value info about the loaded round set (e.g. dev_mode, counts)."""
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[dict] = mapped_column(JSONType)


def make_engine(url: str = settings.DATABASE_URL):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db(bind=None) -> None:
    Base.metadata.create_all(bind or engine)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session

"""Small shared helpers: logging and loading raw files."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import config


def log(step: str, message: str) -> None:
    """Print a message and append it to the pipeline log so runs are traceable."""
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} [{step}] {message}"
    print(line)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with config.LOG_PATH.open("a") as f:
        f.write(line + "\n")


def raw_listing_files(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """CarGurus listing files in raw_dir. Parquet wins over a CSV with the same name."""
    parquet = {p.stem: p for p in raw_dir.glob("*.parquet")}
    csv = {p.stem: p for p in raw_dir.glob("*.csv") if p.stem not in parquet}
    files = {**csv, **parquet}
    extra = {config.CRAIGSLIST_PATH.stem, config.FORCAR_TITLE_PATH.stem}
    return sorted(p for stem, p in files.items() if stem not in extra)


def load_raw_listings(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Concatenate every raw listing file, keeping only the columns we use."""
    files = raw_listing_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No .parquet or .csv listing files in {raw_dir}")
    frames = []
    for path in files:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
        missing = [c for c in config.USED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        frames.append(df[config.USED_COLUMNS])
    return pd.concat(frames, ignore_index=True)

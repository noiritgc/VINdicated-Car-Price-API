"""Small shared helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from pipeline import config


def log(step: str, message: str) -> None:
    """Print a message and append it to the pipeline log so runs are traceable."""
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} [{step}] {message}"
    print(line)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with config.LOG_PATH.open("a") as f:
        f.write(line + "\n")



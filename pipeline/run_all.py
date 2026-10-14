"""Run the whole pipeline: clean -> train -> evaluate -> build_rounds (-> photos).

Usage:
    python -m pipeline.run_all --dev            # free sample, synthetic prices
    python -m pipeline.run_all                  # paid data with real prices
    python -m pipeline.run_all --photos         # also fetch Wikimedia photos (slow, cached)
"""
from __future__ import annotations

import argparse
import sys

from pipeline import build_rounds, clean, evaluate, photos, train
from pipeline.common import log


def run(step_main, argv: list[str]) -> None:
    saved = sys.argv
    sys.argv = [saved[0], *argv]
    try:
        step_main()
    finally:
        sys.argv = saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="fill synthetic prices")
    parser.add_argument("--photos", action="store_true", help="fetch Wikimedia photos after building rounds")
    args = parser.parse_args()

    log("run_all", f"start (dev={args.dev}, photos={args.photos})")
    run(clean.main, ["--dev"] if args.dev else [])
    run(train.main, [])
    run(evaluate.main, [])
    run(build_rounds.main, [])
    if args.photos:
        run(photos.main, [])
    log("run_all", "done")


if __name__ == "__main__":
    main()

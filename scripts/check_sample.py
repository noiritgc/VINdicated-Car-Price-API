"""Check a purchased CarGurus file (e.g. the 50-row test) has everything the pipeline needs.

Usage: python scripts/check_sample.py path/to/file.csv [more files...]
Exit code 0 = ready to use, 1 = something is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.sources import from_cargurus  # noqa: E402

# Columns we need populated in most rows (share of rows that must be real values)
MIN_FILL = {"price": 0.95, "expectedPrice": 0.80, "mileage": 0.9, "year": 0.99, "make": 0.99,
            "model": 0.99, "accidentCount": 0.7, "ownerCount": 0.7, "sellerState": 0.95}


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def check(df: pd.DataFrame) -> list[str]:
    problems = []
    needed = [c for c in config.USED_COLUMNS if c not in ("accidentDetail", "accidentSeverity")]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    print(f"\n{len(df)} rows. Fill rate (real, non-redacted values):")
    for col in needed:
        if col not in df.columns:
            continue
        s = df[col]
        real = s.notna() & (s.astype(str) != config.REDACTED_TOKEN)
        need = MIN_FILL.get(col)
        flag = ""
        if need is not None and real.mean() < need:
            flag = f"  <-- need >= {need:.0%}"
            problems.append(f"{col} only {real.mean():.0%} filled")
        print(f"  {col:18} {real.mean():6.0%}{flag}")

    for col in config.REQUIRED_REAL_COLUMNS:
        if col in df.columns:
            nums = pd.to_numeric(df[col].replace(config.REDACTED_TOKEN, None), errors="coerce").dropna()
            if len(nums):
                print(f"  {col} range: ${nums.min():,.0f} - ${nums.max():,.0f}, median ${nums.median():,.0f}")
    if "sellerType" in df.columns:
        print(f"  sellerType values: {df['sellerType'].value_counts().to_dict()}")
    if {"price", "expectedPrice"} <= set(df.columns):
        p = pd.to_numeric(df["price"], errors="coerce")
        e = pd.to_numeric(df["expectedPrice"], errors="coerce")
        ratio = (p / e).dropna()
        if len(ratio):
            print(f"  price / expectedPrice median: {ratio.median():.3f}")
    return problems


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    df = pd.concat([load(p) for p in paths], ignore_index=True)
    problems = check(df)
    if not problems:
        print("\nRunning the real (non-dev) cleaning rules on this file:")
        out = clean(from_cargurus(df), dev=False)
        print(f"  {len(out)} of {len(df)} rows survive cleaning")
    print("\nRESULT:", "READY" if not problems else "NOT READY")
    for p in problems:
        print("  -", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

"""Step 3: evaluate the model on the test split, write report.md and coverage.json.

Usage: python -m pipeline.evaluate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from pipeline import config
from pipeline.common import log
from pipeline.severity import final_prices, load_effects

STEP = "evaluate"


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    err = (predicted - actual).abs()
    pct = err / actual
    return {
        "rows": int(len(actual)),
        "MAE ($)": round(float(err.mean()), 0),
        "MAPE (%)": round(float(pct.mean() * 100), 1),
        "median error (%)": round(float(pct.median() * 100), 1),
        "within 10% (%)": round(float((pct <= 0.10).mean() * 100), 1),
        "within 20% (%)": round(float((pct <= 0.20).mean() * 100), 1),
    }


def grouped_errors(test: pd.DataFrame, by: str, min_rows: int = 20) -> pd.DataFrame:
    g = test.groupby(by, observed=True)["pct_err"]
    out = pd.DataFrame({"rows": g.size(), "median error (%)": (g.median() * 100).round(1)})
    return out[out["rows"] >= min_rows].sort_values("median error (%)", ascending=False)


def coverage(train_rows: pd.DataFrame) -> dict[str, list[str]]:
    counts = train_rows.groupby(["make", "model"]).size()
    ok = counts[counts >= config.MIN_TRAIN_ROWS_PER_MODEL]
    bad = counts[counts < config.MIN_TRAIN_ROWS_PER_MODEL]
    key = lambda idx: [f"{m}|{n}" for m, n in idx]  # noqa: E731
    return {"covered": key(ok.index), "excluded": key(bad.index)}


def table(df: pd.DataFrame) -> str:
    return df.to_markdown() if hasattr(df, "to_markdown") and _has_tabulate() else f"```\n{df.to_string()}\n```"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


SOURCE_NOTES = {
    "carfax": "> Trained on real **CARFAX One Price** value estimates (Rebrowser Carfax sample).",
    "cargurus": "> Trained on real CarGurus asking prices.",
}


def accuracy_table(test: pd.DataFrame, model_m: dict) -> list[str]:
    bench = test[test["expectedPrice"].notna()]
    if bench.empty:
        return ["| metric | our model |", "|---|---|", *[f"| {k} | {v} |" for k, v in model_m.items()]]
    bench_m = metrics(bench["price"], bench["expectedPrice"])
    return ["| metric | our model | CarGurus expectedPrice |", "|---|---|---|",
            *[f"| {k} | {model_m[k]} | {bench_m[k]} |" for k in model_m]]


def severity_section(test: pd.DataFrame) -> list[str]:
    if not config.SEVERITY_PATH.exists():
        return []
    effects = json.loads(config.SEVERITY_PATH.read_text())
    rows = [f"| {name} | {e['n']:,} | {e['pct']:+.1f}% | {e['ci90'][0]:+.1f}% to {e['ci90'][1]:+.1f}% |"
            for name, e in sorted(effects.items(), key=lambda kv: -kv[1]["pct"])]
    acc = test[test["accidentCount"] == 1]
    err = (acc["pred"] - acc["price"]).abs() / acc["price"]
    return ["## Accident severity effects", "",
            "Measured by cross-fitting (pipeline/severity.py) on non-test cars. Fair price for a car with an",
            "accident = no-accident model price × this effect.", "",
            "| severity | cars | effect on value | 90% CI |", "|---|---|---|---|", *rows, "",
            f"Held-out accident cars: {len(acc)}, median error {err.median():.1%}.", ""]


def build_report(test: pd.DataFrame, cov: dict[str, list[str]], dev: bool, source: str) -> str:
    model_m = metrics(test["price"], test["pred"])
    met = model_m["median error (%)"] <= config.TARGET_MEDIAN_ERROR * 100
    worst = test.assign(err=(test["pred"] - test["price"]).abs()).nlargest(config.WORST_N, "pct_err")
    worst = worst[["year", "make", "model", "trim", "mileage", "sellerState", "price", "pred", "pct_err"]]
    worst = worst.assign(pred=worst["pred"].round(0), pct_err=(worst["pct_err"] * 100).round(1))

    lines = [
        "# Price model report",
        "",
        "> **DEV MODE:** prices are synthetic (free sample has prices redacted). "
        "These numbers only prove the pipeline works; rerun on paid data for real accuracy."
        if dev else SOURCE_NOTES.get(source, ""),
        "",
        "## Test-set accuracy",
        "",
        *accuracy_table(test, model_m),
        "",
        f"Target median error <= {config.TARGET_MEDIAN_ERROR:.0%}: **{'MET' if met else 'NOT MET'}**",
        "",
        "Note: some error is unavoidable. Options, condition and local demand aren't in the features, "
        "so two cars that look identical here can be worth different amounts.",
        "",
        *severity_section(test),
        "## Error by make", "", table(grouped_errors(test, "make")), "",
        "## Error by age (years)", "", table(grouped_errors(test, "age_bucket")), "",
        "## Error by state", "", table(grouped_errors(test, "sellerState")), "",
        f"## {config.WORST_N} worst predictions", "", table(worst.reset_index(drop=True)), "",
        "## Coverage",
        "",
        f"{len(cov['covered'])} make/model combos have >= {config.MIN_TRAIN_ROWS_PER_MODEL} training rows "
        f"(used for rounds). {len(cov['excluded'])} are excluded.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = pd.read_parquet(config.CLEAN_PATH).merge(pd.read_parquet(config.SPLIT_PATH), on="listingId")
    dev = bool(df["is_dev"].iloc[0])
    source = str(df["source"].iloc[0]) if "source" in df.columns else "cargurus"
    model = CatBoostRegressor()
    model.load_model(str(config.MODEL_PATH))

    test = df[df["split"] == "test"].copy()
    effects = load_effects()
    test["pred"] = final_prices(model, test, effects)
    test["pct_err"] = (test["pred"] - test["price"]).abs() / test["price"]
    test["age_bucket"] = pd.cut(test["age"], config.AGE_BUCKETS, right=False)

    cov = coverage(df[df["split"] != "test"])
    config.COVERAGE_PATH.write_text(json.dumps(cov, indent=1))
    config.REPORT_PATH.write_text(build_report(test, cov, dev, source))
    m = metrics(test["price"], test["pred"])
    log(STEP, f"test rows {m['rows']}, median error {m['median error (%)']}%, "
              f"within 10% {m['within 10% (%)']}%, covered models {len(cov['covered'])}")
    log(STEP, f"wrote {config.REPORT_PATH.name} and {config.COVERAGE_PATH.name}")


if __name__ == "__main__":
    main()

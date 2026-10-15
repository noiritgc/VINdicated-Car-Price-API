"""Step 4: build the round pool (rounds.json) from the model, SHAP breakdowns and scenarios.

Usage: python -m pipeline.build_rounds
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import jsonschema
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

from pipeline import config
from pipeline.breakdown import (Breakdown, add_percent_line, bucket_label, car_name, model_breakdown,
                                round_to, shap_values, typical_miles, typical_miles_table,
                                with_typical_mileage)
from pipeline.common import log
from pipeline.photos import load_cache, photo_key
from pipeline.scenarios import private_discount, title_discount
from pipeline.severity import accident_effect, final_prices, load_effects, no_accident
from pipeline.tips import SOURCES, format_tip

STEP = "build_rounds"
TITLES = list(config.TITLE_DISCOUNTS)
REQUIREMENTS = {
    "high_miles": lambda r: r["mileage"] >= config.HIGH_MILES,
    "overpriced": lambda r: r["dealRatingKey"] == "OVERPRICED",
    "good_deal": lambda r: r["dealRatingKey"] in ("GREAT_PRICE", "GOOD_PRICE"),
}


# ---------------------------------------------------------------- selection
def eligible_rows(df: pd.DataFrame, model: CatBoostRegressor, effects: dict[str, float]) -> pd.DataFrame:
    covered = set(json.loads(config.COVERAGE_PATH.read_text())["covered"])
    out = df.dropna(subset=["accidentCount", "ownerCount", "bodyStyle"]).copy()
    log(STEP, f"{len(out)} rows with accident/owner/body data")
    out = out[(out["make"] + "|" + out["model"]).isin(covered)]
    log(STEP, f"{len(out)} rows in well-covered make/models")
    out["pred"] = final_prices(model, out, effects)
    # Cross-check against an independent valuation: CarGurus expectedPrice, or for Carfax the
    # car's own One Price value. Keeps every revealed answer close to a real valuation.
    reference = out["expectedPrice"].fillna(out["price"])
    gap = (out["pred"] - reference).abs() / reference
    bad = gap > config.MAX_EXPECTED_PRICE_GAP
    log(STEP, f"dropping {int(bad.sum())} rows where the model and the reference valuation differ by > "
              f"{config.MAX_EXPECTED_PRICE_GAP:.0%}")
    return out[~bad.fillna(False)]


def teaching_weight(df: pd.DataFrame) -> pd.Series:
    w = config.TEACHING_WEIGHTS
    return (1.0 + w["accident"] * (df["accidentCount"] > 0)
            + w["many_owners"] * (df["ownerCount"] >= config.MANY_OWNERS)
            + w["high_miles"] * (df["mileage"] >= config.HIGH_MILES)
            + w["overpriced"] * (df["dealRatingKey"] == "OVERPRICED"))


def select_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    weights = teaching_weight(df)
    chosen: list = []
    for _, group in df.groupby("sellerState"):
        n = min(config.MIN_ROUNDS_PER_STATE, len(group))
        p = weights[group.index] / weights[group.index].sum()
        chosen += list(rng.choice(group.index, size=n, replace=False, p=p))
    rest = df.index.difference(chosen)
    n_rest = config.POOL_SIZE - len(chosen)
    p = weights[rest] / weights[rest].sum()
    chosen += list(rng.choice(rest, size=n_rest, replace=False, p=p))
    log(STEP, f"selected {len(chosen)} rows ({df.loc[chosen, 'sellerState'].nunique()} states)")
    return df.loc[chosen].sample(frac=1.0, random_state=config.SEED)


def assign_kinds(rows: pd.DataFrame, templates: dict, rng: np.random.Generator) -> list[tuple]:
    """Return (index, kind, template_name) for every selected row."""
    n = len(rows)
    n_dos = round(n * config.ROUND_MIX["deal_or_scam"])
    n_honest = round(n_dos * config.HONEST_SHARE_OF_DEAL_OR_SCAM)
    scams = [k for k, t in templates.items() if t["is_scam"]]
    honest = [k for k, t in templates.items() if not t["is_scam"]]
    wanted = [scams[i % len(scams)] for i in range(n_dos - n_honest)]
    wanted += [honest[i % len(honest)] for i in range(n_honest)]

    free = list(rows.index)
    plan: list[tuple] = []
    for name in sorted(wanted, key=lambda t: "requires" not in templates[t]):
        req = REQUIREMENTS.get(templates[name].get("requires", ""), lambda r: True)
        match = next((i for i in free if req(rows.loc[i])), free[0])
        free.remove(match)
        plan.append((match, "deal_or_scam", name))

    n_title = round(n * config.ROUND_MIX["title"])
    n_private = round(n * config.ROUND_MIX["private"])
    plan += [(i, "title", None) for i in free[:n_title]]
    plan += [(i, "private", None) for i in free[n_title:n_title + n_private]]
    plan += [(i, "plain", None) for i in free[n_title + n_private:]]
    rng.shuffle(plan)
    return plan


# ---------------------------------------------------------------- one round
def base_tags(row: pd.Series, title: str = "Clean title") -> list[str]:
    return [bucket_label("accidents", row), bucket_label("owners", row), title]


def price_hint(hint: float, fair: int, asking: int | None) -> dict[str, int]:
    step = config.PRICE_HINT_ROUND
    low = math.floor(hint * config.PRICE_HINT_LOW / step) * step
    high = math.ceil(hint * config.PRICE_HINT_HIGH / step) * step
    for value in (fair, asking or fair):
        low = min(low, math.floor(value * 0.8 / step) * step)
        high = max(high, math.ceil(value * 1.2 / step) * step)
    return {"low": max(int(low), 0), "high": int(high)}


def difficulty(row: pd.Series, bd: Breakdown, tags: list[str], seller: str, ctx: dict) -> str:
    notable = sum([
        row["accidentCount"] > 0,
        row["ownerCount"] >= config.MANY_OWNERS,
        row["mileage"] >= config.HIGH_MILES,
        any(t in TITLES for t in tags),
        seller == "private",
    ])
    adj_share = sum(abs(a) for a in bd.adjustments().values()) / max(bd.base, 1)
    common = ctx["model_counts"].get((row["make"], row["model"]), 0) >= config.COMMON_MODEL_MIN_ROWS
    rare_trim = ctx["trim_counts"].get((row["make"], row["model"], row["trim"]), 0) < config.RARE_TRIM_MAX_ROWS
    if rare_trim or notable >= config.HARD_MIN_TAGS or adj_share > config.HARD_MIN_ADJ_SHARE:
        return "hard"
    if common and notable <= config.EASY_MAX_TAGS and adj_share <= config.EASY_MAX_ADJ_SHARE:
        return "easy"
    return "medium"


def bucket_tip(bd: Breakdown) -> tuple[str, str, str]:
    """Tip for the largest adjustment; its % is relative to the value just before that line."""
    amounts = [line["amount"] for line in bd.lines]
    if len(amounts) == 1:
        return format_tip("base", 0, bd.base)
    i = max(range(1, len(amounts)), key=lambda k: abs(amounts[k]))
    return format_tip(bd.buckets[i], amounts[i], sum(amounts[:i]))


def apply_template(row: pd.Series, bd: Breakdown, tpl: dict, ctx: dict,
                   rng: np.random.Generator) -> tuple[int, int, list[str]]:
    """Apply a deal_or_scam template. Returns (shown_mileage, asking_price, seller_notes)."""
    claimed = bd.total
    shown_miles = int(row["mileage"])
    real_miles = shown_miles
    if tpl["true_value"] == "remile":
        shown_miles = round_to(real_miles * rng.uniform(0.35, 0.5), 1000)
        both = pd.DataFrame([row, row]).assign(mileage=[float(real_miles), float(shown_miles)])
        real_price, shown_price = final_prices(ctx["model"], both, ctx["effects"])
        claimed = round_to(bd.total * shown_price / real_price)   # what it'd be worth at the shown miles
        for line, bucket in zip(bd.lines, bd.buckets):
            if bucket == "mileage":
                line["label"] = f"Real mileage {round(real_miles / 1000)}k (odometer shows {round(shown_miles / 1000)}k)"
    elif tpl["true_value"] == "title_pct":
        pct = title_discount(row["make"], row["model"], "Salvage title")
        add_percent_line(bd, pct, "Hidden flood damage (value of a salvage-title car)", "flood")

    asking = round_to(claimed * rng.uniform(tpl["asking"]["min"], tpl["asking"]["max"]))
    real_k = f"{round(real_miles, -3):,}"
    notes = [n.replace("{real_miles}", real_k) for n in tpl["seller_notes"]]
    return shown_miles, asking, notes


def build_round(row: pd.Series, phi: pd.Series, expected: float, kind: str, tpl_name: str | None,
                ctx: dict, rng: np.random.Generator) -> dict:
    """phi/expected: SHAP of this car at typical miles for its age (see breakdown.py)."""
    typical = typical_miles(ctx["typical_miles"], row["age"])
    effect = accident_effect(row, ctx["effects"]) or 0.0
    bd = model_breakdown(row, phi, expected, ctx["prices"][row.name], typical, effect)
    seller, title, asking, scam = "dealer", "Clean title", None, None
    mileage = int(row["mileage"])

    if kind == "title":
        title = TITLES[int(rng.integers(len(TITLES)))]
        add_percent_line(bd, title_discount(row["make"], row["model"], title), title, "title")
    tpl = ctx["templates"][tpl_name] if tpl_name else None
    if kind == "private" or (tpl and tpl["seller_type"] == "private"):
        seller = "private"
        pct = private_discount(row["make"], row["model"], int(row["year"]), row["mileage"])
        add_percent_line(bd, pct, "Private-party sale", "private")

    if tpl:
        mileage, asking, notes = apply_template(row, bd, tpl, ctx, rng)
        tip, tip_source, tip_url = tpl["tip"], tpl["tip_source"], SOURCES[tpl["tip_source"]]
        scam = {"template": tpl_name, "is_scam": tpl["is_scam"], "seller_notes": notes,
                "red_flags": tpl["red_flags"], "correct_action": tpl["correct_action"],
                "true_value": bd.total}
    else:
        tip, tip_source, tip_url = bucket_tip(bd)

    tags = base_tags(row, title)
    photo = ctx["photos"].get(photo_key(row["make"], row["model"], int(row["year"]))) or {}
    return {
        "round_type": "deal_or_scam" if tpl else "price_guess",
        "difficulty": difficulty(row, bd, tags, seller, ctx),
        "car": {"year": int(row["year"]), "make": row["make"], "model": row["model"],
                "trim": row["trim"] if isinstance(row["trim"], str) else None,
                "body_style": row["bodyStyle"]},
        "photo": {"url": photo.get("url"), "credit": photo.get("credit"), "fallback": row["bodyStyle"]},
        "mileage": mileage,
        "state": row["sellerState"],
        "seller_type": seller,
        "tags": tags,
        "asking_price": asking,
        "base_value": {"amount": bd.base,
                       "label": f"Typical value of a {car_name(row)} with typical miles for its age "
                                f"(~{round(typical / 1000):,}k), no accidents, average owners and location"},
        "price_hint_range": price_hint(
            ctx["hints"][(row["make"], row["model"], row["year"] // config.PRICE_HINT_YEAR_BUCKET)], bd.total, asking),
        "fair_price": bd.total,
        "breakdown": bd.lines,
        "tip": tip,
        "tip_source": tip_source,
        "tip_url": tip_url,
        "scam": scam,
    }


# ---------------------------------------------------------------- pools + validation
def split_pools(rounds: list[dict], rng: np.random.Generator) -> dict[str, list[str]]:
    """Stratified by round_type so the daily pool gets the same mix."""
    daily: list[str] = []
    by_type: dict[str, list[str]] = {}
    for r in rounds:
        by_type.setdefault(r["round_type"], []).append(r["round_id"])
    share = config.DAILY_POOL_SIZE / len(rounds)
    for ids in by_type.values():
        k = round(len(ids) * share)
        daily += list(rng.choice(ids, size=k, replace=False))
    daily = daily[:config.DAILY_POOL_SIZE]
    daily_set = set(daily)
    quick = [r["round_id"] for r in rounds if r["round_id"] not in daily_set]
    return {"quick": quick, "daily": sorted(daily)}


def validate(rounds: list[dict]) -> None:
    schema = json.loads(config.SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for r in rounds:
        validator.validate(r)
        total = sum(line["amount"] for line in r["breakdown"])
        assert total == r["fair_price"], f"{r['round_id']}: breakdown {total} != fair {r['fair_price']}"
        assert r["base_value"]["amount"] == r["breakdown"][0]["amount"], f"{r['round_id']}: base_value mismatch"
    json.dumps(rounds, allow_nan=False)  # raises on NaN
    log(STEP, f"validated {len(rounds)} rounds against {config.SCHEMA_PATH.name}")


def main() -> None:
    rng = np.random.default_rng(config.SEED)
    df = pd.read_parquet(config.CLEAN_PATH)
    dev = bool(df["is_dev"].iloc[0])
    source = str(df["source"].iloc[0]) if "source" in df.columns else "cargurus"
    model = CatBoostRegressor()
    model.load_model(str(config.MODEL_PATH))

    effects = load_effects()
    log(STEP, f"accident severity effects: { {k: f'{np.expm1(v):+.1%}' for k, v in effects.items()} }")
    pool = eligible_rows(df, model, effects)
    rows = select_rows(pool, rng)
    miles_table = typical_miles_table(df)
    config.TYPICAL_MILES_PATH.write_text(json.dumps(miles_table, indent=1))
    log(STEP, f"typical miles by age: {dict(list(miles_table.items())[:8])} ...")
    phi, expected = shap_values(model, with_typical_mileage(no_accident(rows), miles_table))
    exp_by_idx = dict(zip(rows.index, expected))
    ctx = {
        "model": model,
        "templates": yaml.safe_load(config.SCAM_TEMPLATES_PATH.read_text()),
        "photos": load_cache(),
        "hints": pool.assign(yb=pool["year"] // config.PRICE_HINT_YEAR_BUCKET)
                     .groupby(["make", "model", "yb"])["pred"].median().to_dict(),
        "model_counts": df.groupby(["make", "model"]).size().to_dict(),
        "trim_counts": df.groupby(["make", "model", "trim"]).size().to_dict(),
        "typical_miles": miles_table,
        "prices": dict(zip(rows.index, final_prices(model, rows, effects))),
        "effects": effects,
    }

    rounds = []
    for n, (idx, kind, tpl) in enumerate(assign_kinds(rows, ctx["templates"], rng), 1):
        r = build_round(rows.loc[idx], phi.loc[idx], exp_by_idx[idx], kind, tpl, ctx, rng)
        rounds.append({"round_id": f"r_{n:06d}", **r})
    validate(rounds)
    pools = split_pools(rounds, rng)

    counts = {
        "round_type": pd.Series([r["round_type"] for r in rounds]).value_counts().to_dict(),
        "difficulty": pd.Series([r["difficulty"] for r in rounds]).value_counts().to_dict(),
        "seller_type": pd.Series([r["seller_type"] for r in rounds]).value_counts().to_dict(),
        "states": len({r["state"] for r in rounds}),
        "pools": {k: len(v) for k, v in pools.items()},
    }
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dev_mode": dev, "source": source, "counts": counts}
    config.ROUNDS_PATH.write_text(json.dumps({"meta": meta, "pools": pools, "rounds": rounds}, indent=1))
    log(STEP, f"wrote {len(rounds)} rounds to {config.ROUNDS_PATH.name}: {counts}")


if __name__ == "__main__":
    main()

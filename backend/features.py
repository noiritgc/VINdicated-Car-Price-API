"""Feature engineering shared by training and the API, so both see identical inputs."""
import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026  # dataset snapshot year; age = REFERENCE_YEAR - year

NUMERIC = ["year", "age", "mileage", "miles_per_year", "log_mileage",
           "accidentCount", "accident_severity", "ownerCount"]
CATEGORICAL = ["make", "model", "trim", "bodyStyle", "drivetrain", "fuelType",
               "engine", "sellerState", "make_model"]
FEATURES = NUMERIC + CATEGORICAL

# accidentDetail text -> ordinal severity
SEVERITY_TEXT = [
    ("airbag", 4), ("moderate to severe", 3), ("severe", 3),
    ("moderate", 2), ("minor", 1),
]
SEVERITY_NAMES = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "airbag": 4}


def severity_from_detail(text):
    if not isinstance(text, str) or not text.strip():
        return np.nan
    t = text.lower()
    for key, val in SEVERITY_TEXT:
        if key in t:
            return val
    return 1.5  # "Accident reported" with no detail


def _cat(series, categories):
    """Categorical with a fixed vocabulary; unseen values become missing (NaN)."""
    s = series.astype("object")
    s = s.where(s.isin(categories))
    return pd.Categorical(s, categories=categories)


def build_features(df: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["year"] = df["year"].astype(float)
    out["age"] = (REFERENCE_YEAR - df["year"]).clip(lower=0).astype(float)
    out["mileage"] = df["mileage"].astype(float)
    out["miles_per_year"] = out["mileage"] / out["age"].clip(lower=0.5)
    out["log_mileage"] = np.log1p(out["mileage"])
    out["accidentCount"] = df.get("accidentCount", pd.Series(np.nan, index=df.index)).astype(float)

    if "accident_severity" in df:
        sev = df["accident_severity"].astype(float)
    else:
        sev = df.get("accidentDetail", pd.Series(None, index=df.index)).map(severity_from_detail)
        sev = sev.where(~(out["accidentCount"] == 0), 0.0)
    out["accident_severity"] = sev
    out["ownerCount"] = df.get("ownerCount", pd.Series(np.nan, index=df.index)).astype(float)

    for c in ["make", "model", "trim", "bodyStyle", "drivetrain", "fuelType", "engine", "sellerState"]:
        out[c] = _cat(df[c], vocab[c])
    mm = (df["make"].astype(str) + "|" + df["model"].astype(str))
    mm_vocab = vocab.get("make_model") or sorted(mm.unique().tolist())
    vocab["make_model"] = mm_vocab
    out["make_model"] = _cat(mm, mm_vocab)
    return out[FEATURES]

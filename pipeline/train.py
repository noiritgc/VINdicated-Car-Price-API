"""Step 2: train the CatBoost price model on log(price).

Usage: python -m pipeline.train
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from pipeline import config
from pipeline.common import log

STEP = "train"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model input frame: categoricals as strings (missing -> 'Unknown'), numerics as float."""
    X = df.reindex(columns=config.FEATURES).copy()
    age = pd.to_numeric(df["age"], errors="coerce")
    X["miles_per_year"] = pd.to_numeric(df["mileage"], errors="coerce") / age.clip(lower=1)
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].fillna(config.MISSING_CATEGORY).astype(str)
    for col in config.NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    return X


def make_pool(df: pd.DataFrame, with_label: bool = True) -> Pool:
    label = np.log(df["price"]) if with_label else None
    return Pool(prepare_features(df), label=label, cat_features=config.CATEGORICAL_FEATURES)


def split_by_listing(df: pd.DataFrame, seed: int = config.SEED) -> pd.Series:
    """Assign 'train' / 'valid' / 'test' per listingId with a fixed seed."""
    ids = df["listingId"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n_test = int(len(ids) * config.TEST_SHARE)
    n_valid = int((len(ids) - n_test) * config.VALID_SHARE_OF_TRAIN)
    labels = {i: "test" for i in ids[:n_test]}
    labels.update({i: "valid" for i in ids[n_test:n_test + n_valid]})
    labels.update({i: "train" for i in ids[n_test + n_valid:]})
    return df["listingId"].map(labels)


def train(df: pd.DataFrame) -> tuple[CatBoostRegressor, pd.Series]:
    split = split_by_listing(df)
    for name in ("train", "valid", "test"):
        log(STEP, f"{name}: {(split == name).sum()} rows")
    model = CatBoostRegressor(**config.CATBOOST_PARAMS)
    model.fit(make_pool(df[split == "train"]), eval_set=make_pool(df[split == "valid"]),
              use_best_model=True)
    log(STEP, f"best iteration {model.get_best_iteration()}, "
              f"valid RMSE(log) {model.get_best_score()['validation']['RMSE']:.4f}")
    return model, split


def predict_price(model: CatBoostRegressor, df: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(make_pool(df, with_label=False)))


def main() -> None:
    df = pd.read_parquet(config.CLEAN_PATH)
    log(STEP, f"loaded {len(df)} clean rows (dev={bool(df['is_dev'].iloc[0])})")
    model, split = train(df)
    model.save_model(str(config.MODEL_PATH))
    pd.DataFrame({"listingId": df["listingId"], "split": split}).to_parquet(config.SPLIT_PATH, index=False)
    log(STEP, f"saved {config.MODEL_PATH.name} and {config.SPLIT_PATH.name}")


if __name__ == "__main__":
    main()

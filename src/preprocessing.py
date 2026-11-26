"""Preprocessing and reproducible candidate estimator construction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def build_candidate(name: str, *, seed: int, iterations: int = 200) -> Any:
    """Build one fixed-seed candidate using the shared feature schema."""
    if name == "catboost":
        return CatBoostRegressor(
            loss_function="RMSE", iterations=iterations, learning_rate=0.05,
            depth=7, l2_leaf_reg=3, random_seed=seed, verbose=False,
            allow_writing_files=False, thread_count=1,
        )
    categorical = ColumnTransformer(
        [(
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(CATEGORICAL_FEATURES),
        )],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    if name == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=iterations, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10, loss="huber", random_state=seed,
        )
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=iterations, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model family {name!r}")
    return Pipeline([("encode", categorical), ("regressor", estimator)])


def fit_candidate(model: Any, name: str, features: pd.DataFrame, target: pd.Series) -> Any:
    """Fit a candidate, explicitly passing CatBoost's categorical columns."""
    matrix = features.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].astype(str)
    if name == "catboost":
        model.fit(matrix, target.astype(float), cat_features=list(CATEGORICAL_FEATURES))
    else:
        model.fit(matrix, target.astype(float))
    return model


def predict_candidate(model: Any, name: str, features: pd.DataFrame) -> pd.Series:
    """Predict from the recorded ordered schema and enforce non-negative output."""
    import numpy as np

    matrix = features.loc[:, FEATURE_COLUMNS].copy()
    if name == "persistence":
        prediction = pd.to_numeric(matrix["dengue_cases"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("persistence received missing or non-finite current case values")
        return pd.Series(np.maximum(prediction, 0.0), index=features.index, name="prediction")
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].astype(str)
    prediction = np.asarray(model.predict(matrix), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{name} produced non-finite predictions")
    return pd.Series(np.maximum(prediction, 0.0), index=features.index, name="prediction")

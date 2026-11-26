"""Canonical feature and forecast-target schema for the dengue pipeline.

The cleaner creates all lagged and rolling columns. This module declares and
checks that contract; it deliberately does not recalculate cleaner features.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "dengue_cases"
CATEGORICAL_FEATURES = ("city", "ward_id")
NUMERIC_FEATURES = (
    "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
    "avg_temp_C", "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases",
    *(f"rain_lag_{week}w" for week in range(1, 5)),
    "rain_sum_2w", "rain_sum_4w",
    *(f"dengue_cases_lag_{week}w" for week in range(1, 5)),
    *(f"avg_temp_C_lag_{week}w" for week in range(1, 5)),
    *(f"relative_humidity_pct_lag_{week}w" for week in range(1, 5)),
    "avg_temp_C_mean_prev_4w", "relative_humidity_pct_mean_prev_4w",
    "calendar_week", "calendar_month",
)
FEATURE_COLUMNS = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
SUPPORTED_HORIZONS = (2, 3, 4)
TARGET_CONSTRUCTION_MAX_WEEKS = 52


def validate_feature_schema(frame: pd.DataFrame) -> None:
    """Raise a useful error when cleaner output lacks or corrupts model fields."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed data is missing required feature columns: {missing}")
    for column in CATEGORICAL_FEATURES:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Categorical feature {column!r} contains missing/blank values")
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"Numeric feature {column!r} contains missing or non-finite values")


def construct_targets(frame: pd.DataFrame, horizons: Iterable[int] = SUPPORTED_HORIZONS) -> pd.DataFrame:
    """Add same-ward future dengue values and dates after chronological sorting.

    Input is expected to have passed weekly-continuity validation. That makes a
    grouped shift of *h* observations exactly an h-week forecast.
    """
    result = frame.copy()
    if "date" not in result or "ward_id" not in result or TARGET_COLUMN not in result:
        raise ValueError("Target construction requires date, ward_id, and dengue_cases")
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)
    grouped = result.groupby("ward_id", sort=False, observed=True)
    for horizon in horizons:
        if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= TARGET_CONSTRUCTION_MAX_WEEKS:
            raise ValueError(
                f"Target-construction horizon must be 1–{TARGET_CONSTRUCTION_MAX_WEEKS} weeks"
            )
        result[f"target_{horizon}w"] = grouped[TARGET_COLUMN].shift(-horizon)
        result[f"target_date_{horizon}w"] = grouped["date"].shift(-horizon)
    return result

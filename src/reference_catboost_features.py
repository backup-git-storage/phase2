"""Reference-repository CatBoost feature contract for Phase2 data.

The reference implementation consumes the raw columns present in the Phase2
processed CSV and engineers features per ward. All lagged/rolling values use
observations strictly before the origin; current-week values are only included
where the reference uses contemporaneous weather or dengue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "date", "city", "ward_id", "population_density_sqkm",
    "water_logging_risk_index", "sanitation_score", "avg_temp_C",
    "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases",
)
CATEGORICAL_FEATURES = ("city", "ward_id")


def feature_names() -> list[str]:
    names = [c for c in RAW_COLUMNS if c not in {"date", "dengue_cases"}]
    for lag in range(1, 5):
        names.extend((f"rainfall_lag_{lag}w", f"avg_temp_lag_{lag}w",
                      f"humidity_lag_{lag}w", f"dengue_cases_lag_{lag}w"))
    for window in (2, 4, 8):
        names.extend((f"rainfall_mean_prev_{window}w", f"rainfall_sum_prev_{window}w"))
    for window in (2, 4, 8):
        names.append(f"avg_temp_mean_prev_{window}w")
    for window in (2, 4, 8):
        names.append(f"humidity_mean_prev_{window}w")
    for window in (2, 4, 8):
        names.append(f"dengue_mean_prev_{window}w")
    names.extend(("dengue_change_1w", "dengue_change_2w", "dengue_change_4w",
                  "dengue_growth_ratio_1w", "dengue_growth_ratio_2w",
                  "calendar_week", "calendar_month", "week_sin", "week_cos",
                  "month_sin", "month_cos"))
    return names

FEATURE_COLUMNS = feature_names()


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate reference features in deterministic order without crossing wards."""
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing reference source columns: {missing}")
    data = frame.loc[:, RAW_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)
    if data.duplicated(["ward_id", "date"]).any():
        raise ValueError("Dataset contains duplicate ward/date observations")
    for ward, group in data.groupby("ward_id", sort=False, observed=True):
        if len(group) > 1 and not group.date.diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError(f"Ward {ward!r} does not have continuous weekly observations")
    grouped = data.groupby("ward_id", sort=False, observed=True)
    for col in RAW_COLUMNS:
        if col not in {"date", "city", "ward_id"}:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for lag in range(1, 5):
        for source, name in (("rainfall_mm", "rainfall"), ("avg_temp_C", "avg_temp"),
                             ("relative_humidity_pct", "humidity"), ("dengue_cases", "dengue_cases")):
            data[f"{name}_lag_{lag}w"] = grouped[source].shift(lag)
    grouped = data.groupby("ward_id", sort=False, observed=True)
    for window in (2, 4, 8):
        previous = grouped["rainfall_mm"].transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
        data[f"rainfall_mean_prev_{window}w"] = previous
        data[f"rainfall_sum_prev_{window}w"] = grouped["rainfall_mm"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).sum())
    for source, prefix in (("avg_temp_C", "avg_temp"), ("relative_humidity_pct", "humidity"),
                           ("dengue_cases", "dengue")):
        grouped = data.groupby("ward_id", sort=False, observed=True)
        for window in (2, 4, 8):
            data[f"{prefix}_mean_prev_{window}w"] = grouped[source].transform(
                lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    data["dengue_change_1w"] = data.dengue_cases - data.dengue_cases_lag_1w
    data["dengue_change_2w"] = data.dengue_cases_lag_1w - data.dengue_cases_lag_2w
    data["dengue_change_4w"] = data.dengue_cases_lag_1w - data.dengue_cases_lag_4w
    data["dengue_growth_ratio_1w"] = data.dengue_cases / (data.dengue_cases_lag_1w + 1)
    data["dengue_growth_ratio_2w"] = data.dengue_cases / (data.dengue_cases_lag_2w + 1)
    iso = data.date.dt.isocalendar()
    data["calendar_week"] = iso.week.astype(int)
    data["calendar_month"] = data.date.dt.month
    data["week_sin"] = np.sin(2 * np.pi * data.calendar_week / 52)
    data["week_cos"] = np.cos(2 * np.pi * data.calendar_week / 52)
    data["month_sin"] = np.sin(2 * np.pi * data.calendar_month / 12)
    data["month_cos"] = np.cos(2 * np.pi * data.calendar_month / 12)
    data["city"] = data.city.astype(str)
    data["ward_id"] = data.ward_id.astype(str)
    return data


def usable_feature_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows with complete features and finite numeric inputs."""
    missing = [c for c in FEATURE_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"Feature frame is missing columns: {missing}")
    matrix = frame.loc[:, FEATURE_COLUMNS]
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
    finite = np.isfinite(matrix[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)
    cats = matrix[list(CATEGORICAL_FEATURES)].notna().all(axis=1).to_numpy()
    return pd.Series(finite & cats, index=frame.index)

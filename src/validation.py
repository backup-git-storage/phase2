"""Validation for the tabular source and cleaner-produced training dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_SOURCE_COLUMNS = (
    "date", "city", "ward_id", "population_density_sqkm",
    "water_logging_risk_index", "sanitation_score", "max_temp_c",
    "min_temp_c", "rainfall_mm", "relative_humidity_pct", "dengue_cases",
)
TEMP_ALIASES = ("avg_temp_c", "avg_temp_C")
NUMERIC_SOURCE_COLUMNS = (
    "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
    "avg_temp", "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases",
)


def validate_dataset(frame: pd.DataFrame, *, require_weekly: bool = True) -> pd.DataFrame:
    """Validate source values, identifiers, duplicate keys and weekly spacing.

    Returns a copy with parsed dates and numeric source values. Missing values
    are rejected here; the CSV cleaner reports and removes missing source rows
    before reaching this validator.
    """
    if frame.empty:
        raise ValueError("Dataset is empty")
    missing = [name for name in REQUIRED_SOURCE_COLUMNS if name not in frame]
    if not any(name in frame for name in TEMP_ALIASES):
        missing.append("avg_temp_c or avg_temp_C")
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    data = frame.copy()
    average_column = next(name for name in TEMP_ALIASES if name in data)
    data["_average_temp"] = pd.to_numeric(data[average_column], errors="coerce")
    if data[list(REQUIRED_SOURCE_COLUMNS)].isna().any().any() or data[["city", "ward_id"]].isna().any().any():
        raise ValueError("Source dataset contains missing required values")
    parsed = pd.to_datetime(data["date"], format="ISO8601", errors="coerce", utc=True)
    if parsed.isna().any():
        bad_rows = data.index[parsed.isna()].tolist()[:10]
        raise ValueError(f"Invalid date values at row(s): {bad_rows}")
    data["date"] = parsed.dt.tz_convert(None).dt.normalize()
    numeric_map = {"avg_temp": "_average_temp"}
    for name in NUMERIC_SOURCE_COLUMNS:
        column = numeric_map.get(name, name)
        data[column] = pd.to_numeric(data[column], errors="coerce")
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Numeric source column {column!r} contains non-finite or invalid values")
    nonnegative = ("rainfall_mm", "dengue_cases")
    for name in nonnegative:
        if (data[name] < 0).any():
            raise ValueError(f"{name} must be non-negative")
    if (data["population_density_sqkm"] <= 0).any():
        raise ValueError("population_density_sqkm must be positive")
    for name in ("water_logging_risk_index", "sanitation_score"):
        if not data[name].between(0, 1).all():
            raise ValueError(f"{name} must be within [0, 1]")
    if not data["relative_humidity_pct"].between(0, 100).all():
        raise ValueError("relative_humidity_pct must be within [0, 100]")
    for name in ("_average_temp", "max_temp_c", "min_temp_c"):
        if not data[name].between(-90, 60).all():
            raise ValueError(f"{name} must be within plausible bounds [-90, 60] °C")
    if not ((data["min_temp_c"] <= data["_average_temp"]) & (data["_average_temp"] <= data["max_temp_c"])).all():
        raise ValueError("Temperature columns must satisfy min_temp_c <= average <= max_temp_c")
    if not np.equal(data["dengue_cases"], np.floor(data["dengue_cases"])).all():
        raise ValueError("dengue_cases contains invalid non-integer target values")
    for column in ("city", "ward_id"):
        labels = data[column].astype(str)
        if labels.str.strip().ne(labels).any() or labels.eq("").any():
            raise ValueError(f"{column} contains blank labels or surrounding whitespace")
        if labels.str.casefold().duplicated(keep=False).any():
            duplicates = sorted(labels[labels.str.casefold().duplicated(keep=False)].unique())
            if len({value.casefold() for value in duplicates}) < len(duplicates):
                raise ValueError(f"{column} has case-inconsistent labels: {duplicates[:10]}")
    city_counts = data.groupby("ward_id", observed=True)["city"].nunique()
    if (city_counts > 1).any():
        raise ValueError(f"Ward identifiers map to multiple cities: {city_counts[city_counts > 1].index.tolist()[:10]}")
    if data.duplicated(["date", "ward_id"]).any():
        raise ValueError("Duplicate (date, ward_id) records found")
    if require_weekly:
        ordered = data.sort_values(["ward_id", "date"], kind="stable")
        intervals = ordered.groupby("ward_id", observed=True)["date"].diff().dropna().dt.days
        if not intervals.eq(7).all():
            raise ValueError("Weekly continuity failed: every within-ward observation must be exactly 7 days apart")
    data = data.drop(columns=["_average_temp"])
    return data

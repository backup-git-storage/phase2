"""Transparent prototype risk bands and observed-input context."""

from __future__ import annotations

import math

import pandas as pd

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
MIN_ACTIVE_WEEKS = 8


def risk_level_index(level: str) -> int:
    """Return the ordinal position used only for tier-distance metrics."""
    try:
        return RISK_LEVELS.index(str(level))
    except ValueError as exc:
        raise ValueError(f"Unknown risk level {level!r}") from exc


def risk_thresholds(historical_cases: pd.Series) -> dict[str, float]:
    """Return ward-relative case-count cutoffs; these are not clinical thresholds."""
    values = pd.to_numeric(historical_cases, errors="coerce").dropna()
    values = values[values >= 0]
    if values.empty:
        raise ValueError("Cannot classify risk without historical case observations")
    active = values[values > 0]
    if len(active) >= MIN_ACTIVE_WEEKS:
        low, medium, high = (float(active.quantile(q)) for q in (0.50, 0.75, 0.90))
    else:
        # Sparse-history fallback is a visible prototype heuristic, not a warning threshold.
        low, medium, high = 3.0, 6.0, 10.0
    low = max(1.0, low)
    medium = max(low + 1.0, medium)
    high = max(medium + 1.0, high)
    return {"low_max": low, "medium_max": medium, "high_max": high}


def classify_risk(predicted_cases: float, historical_cases: pd.Series) -> dict:
    """Classify a point forecast using only the ward history available at its origin."""
    value = float(predicted_cases)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Predicted cases must be a finite, non-negative number")
    thresholds = risk_thresholds(historical_cases)
    if value < 1:
        level = "LOW"
    elif value <= thresholds["low_max"]:
        level = "LOW"
    elif value <= thresholds["medium_max"]:
        level = "MEDIUM"
    elif value <= thresholds["high_max"]:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return {
        "level": level,
        "method": "ward_active_week_percentiles",
        "thresholds": thresholds,
        "warning": (
            "Experimental ward-relative weekly new-case bands based on historical active weeks; "
            "not clinically validated public-health alerts and not a probability."
        ),
    }


def _relative_label(value: float, historical: pd.Series, unit: str) -> str:
    values = pd.to_numeric(historical, errors="coerce").dropna()
    if values.empty:
        return f"{value:.1f}{unit}; historical comparison unavailable"
    relation = "above" if value > float(values.quantile(0.75)) else "within or below"
    return f"{value:.1f}{unit}; {relation} the ward's historical 75th percentile"


def describe_drivers(ward_history: pd.DataFrame, latest: pd.Series) -> dict[str, str]:
    """Summarize observed context without claiming feature attribution or causality."""
    return {
        "rainfall": _relative_label(float(latest["rainfall_mm"]), ward_history["rainfall_mm"], " mm"),
        "temperature": _relative_label(float(latest["avg_temp_C"]), ward_history["avg_temp_C"], " °C"),
        "humidity": _relative_label(float(latest["relative_humidity_pct"]), ward_history["relative_humidity_pct"], "%"),
        "recentCases": (
            f"{float(latest['dengue_cases']):g} new cases in the observed reporting week; "
            "context only, not a causal explanation"
        ),
    }

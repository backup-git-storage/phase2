"""Cached inference for portable, leakage-aware weekly model runs."""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, SUPPORTED_HORIZONS
from src.risk_engine import classify_risk, describe_drivers
from src.train import ARTIFACT_SCHEMA_VERSION, EVALUATION_METHOD, TARGET_DEFINITION, display_case_count
from src.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dengue_dataset.csv"
RUNS_DIR = ROOT / "runs"
MIN_HISTORY_WEEKS = 8
MAX_CURRENT_DATA_AGE_DAYS = 14


class ForecastUnavailableError(RuntimeError):
    """The application lacks sufficiently fresh, valid forecast data."""


class ModelUnavailableError(RuntimeError):
    """No compatible trained model artifact is available."""


@lru_cache(maxsize=1)
def load_raw_data() -> pd.DataFrame:
    """Read and validate the processed weekly feature table once per process."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Processed weekly ward data is unavailable")
    try:
        frame = pd.read_csv(DATA_PATH)
        data = validate_dataset(frame)
    except Exception as exc:
        raise ForecastUnavailableError("Processed weekly ward data is invalid") from exc
    missing = [name for name in FEATURE_COLUMNS if name not in data]
    if missing:
        raise ForecastUnavailableError("Processed data does not match the model feature contract")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    for name in FEATURE_COLUMNS:
        if name not in {"city", "ward_id"}:
            data[name] = pd.to_numeric(data[name], errors="coerce")
    numeric = data.loc[:, [c for c in FEATURE_COLUMNS if c not in {"city", "ward_id"}]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ForecastUnavailableError("Processed model features contain missing or non-finite values")
    return data.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)


def get_supported_areas() -> dict[str, list[str]]:
    data = load_raw_data()
    return {str(city): sorted(group.ward_id.astype(str).unique().tolist())
            for city, group in data.groupby("city", sort=True, observed=True)}


def data_freshness(today: date | None = None) -> dict:
    data_through = load_raw_data()["date"].max().date()
    today = today or date.today()
    age_days = max(0, (today - data_through).days)
    stale = age_days > MAX_CURRENT_DATA_AGE_DAYS
    return {
        "dataThrough": data_through.isoformat(), "ageDays": age_days,
        "status": "stale" if stale else "current",
        "currentForecastAvailable": not stale,
        "maxAgeDays": MAX_CURRENT_DATA_AGE_DAYS,
        "message": (
            f"Data ends {data_through.isoformat()} ({age_days} days old). Current forecasting is disabled; "
            "choose a historical origin whose target week is present for a historical replay."
            if stale else "The latest weekly data is within the configured 14-day freshness window."
        ),
    }


def get_data_through() -> str:
    return load_raw_data()["date"].max().strftime("%Y-%m-%d")


def get_available_horizons() -> list[int]:
    available = []
    for horizon in SUPPORTED_HORIZONS:
        try:
            load_model_bundle(horizon)
            if not replay_origins(horizon):
                continue
        except (FileNotFoundError, ModelUnavailableError, ValueError, OSError, json.JSONDecodeError):
            continue
        available.append(horizon)
    return available


def latest_replay_origin(horizon_weeks: int) -> str:
    origins = replay_origins(horizon_weeks)
    if not origins:
        raise ForecastUnavailableError("No leakage-safe rolling-origin test replays are available for this horizon")
    return origins[-1]


def replay_origin_bounds(horizon_weeks: int) -> dict[str, str | list[str]]:
    origins = replay_origins(horizon_weeks)
    if not origins:
        raise ForecastUnavailableError("No leakage-safe rolling-origin test replays are available for this horizon")
    return {"min": origins[0], "max": origins[-1], "default": origins[-1], "available": origins}


def _run_directory(horizon_weeks: int) -> Path:
    override = os.getenv("AEDISIGHT_MODEL_RUN", "").strip()
    if override:
        path = Path(override).expanduser()
        run_dir = path if path.is_absolute() else (ROOT / path).resolve()
        return run_dir
    candidates: list[tuple[str, Path]] = []
    if RUNS_DIR.is_dir():
        for metadata_path in RUNS_DIR.glob("*/metadata.json"):
            run_dir = metadata_path.parent
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            horizon_meta = metadata.get("horizons", {}).get(str(horizon_weeks), {})
            artifact = horizon_meta.get("model_artifact")
            compatible = (
                metadata.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
                and metadata.get("evaluation_method") == EVALUATION_METHOD
                and metadata.get("target_definition") == TARGET_DEFINITION
                and metadata.get("feature_names") == list(FEATURE_COLUMNS)
                and horizon_meta.get("target_definition") == TARGET_DEFINITION
                and horizon_meta.get("horizon_weeks") == horizon_weeks
            )
            if compatible and artifact and (run_dir / artifact).is_file():
                candidates.append((str(metadata.get("run_created_utc", "")), run_dir))
    if not candidates:
        raise FileNotFoundError("No compatible trained model run is available for this horizon")
    return max(candidates, key=lambda item: item[0])[1]


@lru_cache(maxsize=3)
def load_model_bundle(horizon_weeks: int) -> dict:
    """Load one compatible artifact per horizon and validate its full contract."""
    if horizon_weeks not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon; choose one of {SUPPORTED_HORIZONS} weeks")
    try:
        run_dir = _run_directory(horizon_weeks)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        horizon_meta = metadata["horizons"][str(horizon_weeks)]
        artifact_path = (run_dir / horizon_meta["model_artifact"]).resolve()
        if run_dir.resolve() not in artifact_path.parents or not artifact_path.is_file():
            raise ModelUnavailableError("The configured model artifact is missing or outside its run directory")
        bundle = joblib.load(artifact_path)
    except FileNotFoundError as exc:
        raise
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError("A trained model artifact could not be loaded") from exc
    required = {"model", "model_family", "horizon_weeks", "feature_names", "categorical_features",
                "target_definition", "training_data_cutoff", "artifact_schema_version"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ModelUnavailableError("The model artifact uses an incompatible format")
    if (bundle["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION
            or int(bundle["horizon_weeks"]) != horizon_weeks
            or list(bundle["feature_names"]) != list(FEATURE_COLUMNS)
            or list(bundle["categorical_features"]) != ["city", "ward_id"]
            or bundle["target_definition"] != TARGET_DEFINITION
            or bundle["model_family"] != horizon_meta.get("model_family")
            or bundle["training_data_cutoff"] != horizon_meta.get("training_data_cutoff")
            or metadata.get("evaluation_method") != EVALUATION_METHOD):
        raise ModelUnavailableError("Model horizon, target definition, or ordered features do not match inference")
    return bundle


@lru_cache(maxsize=3)
def _load_replay_predictions(horizon_weeks: int) -> pd.DataFrame:
    """Load selected-model predictions saved from held-out rolling test origins."""
    bundle = load_model_bundle(horizon_weeks)
    run_dir = _run_directory(horizon_weeks)
    path = run_dir / "results" / "test_predictions.csv"
    if not path.is_file():
        raise ModelUnavailableError("Saved leakage-safe historical replay predictions are unavailable")
    try:
        predictions = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ModelUnavailableError("Saved historical replay predictions could not be read") from exc
    required = {
        "horizon_weeks", "model", "partition", "ward_id", "forecast_origin_date",
        "feature_information_date", "training_label_cutoff", "target_date",
        "actual_target", "prediction",
    }
    if not required.issubset(predictions.columns):
        raise ModelUnavailableError("Saved historical replay predictions use an incompatible schema")
    selected = predictions.loc[
        predictions.horizon_weeks.eq(horizon_weeks)
        & predictions.model.eq(bundle["model_family"])
        & predictions.partition.eq("test")
    ].copy()
    if selected.empty:
        raise ModelUnavailableError("No held-out replay predictions match the selected model")
    return selected


def replay_origins(horizon_weeks: int) -> list[str]:
    """Return only held-out rolling-origin dates with leakage-safe saved forecasts."""
    predictions = _load_replay_predictions(horizon_weeks)
    return sorted(predictions.forecast_origin_date.astype(str).unique().tolist())


@lru_cache(maxsize=3)
def get_model_performance(horizon_weeks: int) -> dict | None:
    """Read the genuine saved rolling-origin validation/test metrics, if present."""
    try:
        bundle = load_model_bundle(horizon_weeks)
        run_dir = _run_directory(horizon_weeks)
        validation = pd.read_csv(run_dir / "results" / "validation_model_comparison.csv")
        test = pd.read_csv(run_dir / "results" / "final_test_metrics.csv")
        family = bundle["model_family"]
        val = validation.loc[
            validation.horizon_weeks.eq(horizon_weeks)
            & validation.model.eq(family)
            & validation.partition.eq("validation")
        ]
        held_out = test.loc[
            test.horizon_weeks.eq(horizon_weeks)
            & test.model.eq(family)
            & test.partition.eq("test")
        ]
        if val.empty or held_out.empty:
            return None
        validation_row, test_row = val.iloc[0], held_out.iloc[0]
        baseline_rows = test.loc[
            test.horizon_weeks.eq(horizon_weeks)
            & test.model.isin(["persistence", "seasonal_naive"])
            & test.partition.eq("test")
        ].set_index("model")
        risk_path = run_dir / "results" / "risk_metrics.csv"
        risk_value = None
        if risk_path.is_file():
            risk_table = pd.read_csv(risk_path)
            risk_match = risk_table.loc[
                risk_table.horizon_weeks.eq(horizon_weeks)
                & risk_table.model.eq(family)
                & risk_table.partition.eq("test")
            ]
            if not risk_match.empty:
                rr = risk_match.iloc[0]
                risk_value = {
                    key: float(rr[key]) if pd.notna(rr[key]) else None
                    for key in ("exact_tier_accuracy", "within_one_tier_accuracy", "high_critical_precision", "high_critical_recall", "critical_recall")
                }
                risk_value["missed_critical_count"] = int(rr.missed_critical_count)
        return {
            "model": family,
            "validationMae": float(validation_row.mae),
            "validationRmse": float(validation_row.rmse),
            "validationRows": int(validation_row.rows),
            "testMae": float(test_row.mae),
            "testRmse": float(test_row.rmse),
            "testR2": float(test_row.r2) if pd.notna(test_row.r2) else None,
            "testRows": int(test_row.rows),
            "originStart": str(test_row.origin_start),
            "originEnd": str(test_row.origin_end),
            "targetStart": str(test_row.target_start),
            "targetEnd": str(test_row.target_end),
            "persistenceMae": float(baseline_rows.loc["persistence", "mae"]) if "persistence" in baseline_rows.index else None,
            "seasonalNaiveMae": float(baseline_rows.loc["seasonal_naive", "mae"]) if "seasonal_naive" in baseline_rows.index else None,
            "risk": risk_value,
        }
    except (FileNotFoundError, ModelUnavailableError, ValueError, KeyError, OSError, pd.errors.ParserError):
        return None


def _format_observation(row: pd.Series) -> dict:
    return {"date": pd.Timestamp(row.date).strftime("%Y-%m-%d"), "cases": float(row.dengue_cases)}


def generate_forecast(
    city: str, ward_id: str, horizon_weeks: int,
    as_of: str | None = None, today: date | None = None,
) -> dict:
    """Generate a current forecast or an explicitly historical, scored replay.

    Historical `as_of` dates must exactly match a weekly record and the target
    observation must exist. A stale dataset never generates a current forecast.
    """
    if horizon_weeks not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon; choose one of {SUPPORTED_HORIZONS} weeks")
    data = load_raw_data()
    area_map = get_supported_areas()
    if city not in area_map:
        raise LookupError("Unknown city")
    if ward_id not in {ward for city_wards in area_map.values() for ward in city_wards}:
        raise LookupError("Unknown ward")
    if ward_id not in area_map[city]:
        raise LookupError("Ward does not belong to the selected city")
    ward_data = data.loc[data.ward_id.astype(str).eq(str(ward_id))].sort_values("date")
    cutoff = data["date"].max()
    freshness = data_freshness(today)
    historical = as_of is not None
    if historical:
        try:
            origin = pd.Timestamp(date.fromisoformat(as_of))
        except (TypeError, ValueError) as exc:
            raise ValueError("Forecast origin must be a valid YYYY-MM-DD weekly observation date") from exc
        origin_rows = ward_data.loc[ward_data.date.eq(origin)]
        if origin_rows.empty:
            raise ValueError("Forecast origin is unsupported; choose an exact weekly observation date")
    else:
        if freshness["status"] == "stale":
            raise ForecastUnavailableError(freshness["message"])
        origin = ward_data.date.max()
        origin_rows = ward_data.loc[ward_data.date.eq(origin)]
    if len(ward_data.loc[ward_data.date.le(origin)]) < MIN_HISTORY_WEEKS:
        raise ValueError("Insufficient weekly history before the selected forecast origin")

    latest = origin_rows.iloc[-1]
    target_date = origin + pd.Timedelta(weeks=horizon_weeks)
    target_rows = ward_data.loc[ward_data.date.eq(target_date)]
    if historical and target_rows.empty:
        raise ValueError("Historical replay target week is not in the dataset; choose an earlier origin")
    if not historical and target_date <= origin:
        raise ValueError("Target observation date must be after the forecast origin")

    try:
        bundle = load_model_bundle(horizon_weeks)
    except FileNotFoundError as exc:
        raise ModelUnavailableError("No compatible model is available for this forecast horizon") from exc
    except ModelUnavailableError:
        raise
    family = bundle["model_family"]
    model_training_cutoff = str(bundle["training_data_cutoff"])
    replay_partition = None
    if historical:
        saved_predictions = _load_replay_predictions(horizon_weeks)
        replay_rows = saved_predictions.loc[
            saved_predictions.ward_id.astype(str).eq(str(ward_id))
            & saved_predictions.forecast_origin_date.astype(str).eq(origin.date().isoformat())
        ]
        if len(replay_rows) != 1:
            raise ValueError(
                "Historical replay is available only for saved held-out rolling-origin test weeks; "
                "choose one of the supported replay origins."
            )
        replay = replay_rows.iloc[0]
        expected_target = origin + pd.Timedelta(weeks=horizon_weeks)
        feature_date = pd.Timestamp(replay.feature_information_date)
        label_cutoff = pd.Timestamp(replay.training_label_cutoff)
        if (
            feature_date != origin
            or label_cutoff > feature_date
            or pd.Timestamp(replay.target_date) != expected_target
            or not np.isclose(float(replay.actual_target), float(target_rows.iloc[0].dengue_cases))
        ):
            raise ModelUnavailableError("Saved replay prediction does not match the origin, target, or available-label contract")
        raw_prediction = float(replay.prediction)
        model_training_cutoff = label_cutoff.date().isoformat()
        replay_partition = str(replay.partition)
    else:
        matrix = pd.DataFrame([latest[list(bundle["feature_names"])]], columns=bundle["feature_names"])
        for column in bundle["categorical_features"]:
            matrix[column] = matrix[column].astype(str)
        for column in bundle["feature_names"]:
            if column not in bundle["categorical_features"]:
                matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        numeric = matrix.loc[:, [c for c in bundle["feature_names"] if c not in bundle["categorical_features"]]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ForecastUnavailableError("Forecast-origin features contain missing or non-finite values")
        if family == "persistence":
            raw_prediction = float(latest["dengue_cases"])
        elif family == "seasonal_naive":
            seasonal_date = target_date - pd.Timedelta(weeks=52)
            seasonal_row = ward_data.loc[ward_data.date.eq(seasonal_date)]
            if seasonal_row.empty or seasonal_date > origin:
                raise ModelUnavailableError("Seasonal-naive origin value is unavailable")
            raw_prediction = float(seasonal_row.iloc[0]["dengue_cases"])
        else:
            try:
                raw_prediction = float(np.asarray(bundle["model"].predict(matrix), dtype=float).reshape(-1)[0])
            except Exception as exc:
                raise ModelUnavailableError("The compatible model failed to produce a forecast") from exc
    if not np.isfinite(raw_prediction):
        raise ModelUnavailableError("Model produced a non-finite forecast")
    nonnegative_raw = max(0.0, raw_prediction)
    predicted_count = display_case_count(nonnegative_raw)
    history_available = ward_data.loc[ward_data.date.le(origin)]
    risk = classify_risk(predicted_count, history_available.dengue_cases)

    target_rain_date = target_date - pd.Timedelta(weeks=3)
    rainfall_context = None
    if target_rain_date <= origin:
        rain_row = ward_data.loc[ward_data.date.eq(target_rain_date)]
        if not rain_row.empty:
            rainfall_context = {"date": target_rain_date.date().isoformat(), "rainfall_mm": float(rain_row.iloc[0].rainfall_mm)}
    available_historical_target = None if target_rows.empty else float(target_rows.iloc[0].dengue_cases)
    warnings = [risk["warning"]]
    if freshness["status"] == "stale":
        warnings.insert(0, freshness["message"])
    if target_rain_date > origin:
        warnings.append("Rainfall approximately three weeks before the target would occur after this origin and is unavailable; no future rainfall was used.")
    return {
        "city": str(city), "wardId": str(ward_id),
        "horizonWeeks": horizon_weeks, "targetDefinition": TARGET_DEFINITION,
        "predictedCases": predicted_count, "rawPredictedCases": nonnegative_raw,
        "riskCategory": risk["level"], "riskMethod": risk["method"],
        "riskThresholds": risk["thresholds"], "riskWarning": risk["warning"],
        "observationDate": origin.date().isoformat(), "forecastOrigin": origin.date().isoformat(),
        "targetDate": target_date.date().isoformat(), "targetReportingWeek": target_date.date().isoformat(),
        "actualCases": available_historical_target,
        "dataThrough": cutoff.date().isoformat(), "dataFreshness": freshness["status"],
        "dataAgeDays": freshness["ageDays"], "mode": "historical_replay" if historical else "current_forecast",
        "dataIsHistorical": historical, "warnings": warnings,
        "drivers": describe_drivers(history_available, latest),
        "rainfallTiming": {
            "target_minus_three_weeks": rainfall_context,
            "interpretation": "Observed timing context only; not a causal or feature-attribution claim.",
        },
        "observedFeatures": {
            name: float(latest[name]) for name in (
                "avg_temp_C", "relative_humidity_pct", "rainfall_mm",
                *(f"rain_lag_{week}w" for week in range(1, 5)),
                "rain_sum_2w", "rain_sum_4w", "population_density_sqkm",
                "water_logging_risk_index", "sanitation_score",
            ) if name in latest.index and pd.notna(latest[name])
        },
        "history": [_format_observation(row) for row in history_available.tail(16).itertuples(index=False)],
        "forecast": [{"date": target_date.date().isoformat(), "cases": predicted_count}],
        "model": family, "modelTrainingDataCutoff": model_training_cutoff,
        "historicalReplayPartition": replay_partition,
        "modelPerformance": get_model_performance(horizon_weeks),
        "modelHorizonWeeks": int(bundle["horizon_weeks"]),
    }

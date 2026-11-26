"""Leakage-aware weekly dengue model selection, backtesting and artifact creation.

Each backtest origin is simulated independently. Training labels are admitted
only when their target reporting week was available by the simulated origin.
The reporting-delay sensitivity shifts both the feature cutoff and target lead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.feature_engineering import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, SUPPORTED_HORIZONS,
    construct_targets, validate_feature_schema,
)
from src.preprocessing import build_candidate, fit_candidate, predict_candidate
from src.risk_engine import classify_risk, risk_level_index
from src.validation import validate_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "processed" / "dengue_dataset.csv"
DEFAULT_OUTPUT_PARENT = ROOT / "runs"
DEFAULT_CANDIDATES = ("catboost", "gradient_boosting", "hist_gradient_boosting")
BASELINES = ("persistence", "seasonal_naive")
TARGET_DEFINITION = "Weekly new dengue cases reported for the ward and target observation week; not cumulative."
EVALUATION_METHOD = "expanding_window_rolling_origin_v1"
ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SplitConfig:
    """Calendar ranges for rolling validation/test forecast issuance dates."""

    train_through: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    origin_step_weeks: int = 4

    def validate(self) -> None:
        if self.train_through >= self.validation_start:
            raise ValueError("train_through must precede validation_start")
        if self.validation_start > self.validation_end or self.validation_end >= self.test_start:
            raise ValueError("Validation and test origin periods must be ordered and non-overlapping")
        if self.test_start > self.test_end:
            raise ValueError("test_start must not exceed test_end")
        if self.origin_step_weeks < 1:
            raise ValueError("origin_step_weeks must be positive")


def _portable_input_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def calculate_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    """Calculate regression metrics on a fixed, paired sample."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Metrics require non-empty, equally sized one-dimensional values")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics cannot be calculated from non-finite values")
    error = p - y
    denominator = float(np.square(y - y.mean()).sum())
    r2 = float("nan") if denominator == 0 else float(1.0 - np.square(error).sum() / denominator)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())), "r2": r2}


def display_case_count(prediction: float) -> int:
    """Clip below zero and round half-up for a whole weekly reported count.

    Continuous non-negative model outputs are retained in evaluation CSVs.
    This conversion is for the user-facing count and risk band only.
    """
    value = float(prediction)
    if not math.isfinite(value):
        raise ValueError("Prediction must be finite")
    return int(math.floor(max(0.0, value) + 0.5))


def select_candidate(candidates: list[tuple[float, str, Any, dict[str, float]]]) -> tuple[float, str, Any, dict[str, float]]:
    """Select by validation MAE, preferring persistence then seasonal naive on ties."""
    if not candidates:
        raise ValueError("At least one validation candidate is required")
    tie_order = {"persistence": 0, "seasonal_naive": 1}
    return min(candidates, key=lambda row: (row[0], tie_order.get(row[1], 2), row[1]))


def _feature_importances(model: Any, model_name: str) -> pd.DataFrame:
    if model_name in BASELINES or model is None:
        return pd.DataFrame(columns=["feature", "importance"])
    if model_name == "catboost":
        values = model.get_feature_importance()
        names = list(FEATURE_COLUMNS)
    else:
        estimator = model.named_steps["regressor"]
        values = getattr(estimator, "feature_importances_", None)
        names = list(model.named_steps["encode"].get_feature_names_out())
    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": names, "importance": np.asarray(values, dtype=float)}).sort_values("importance", ascending=False)


def _origin_dates(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, max_lead: int, step_weeks: int) -> list[pd.Timestamp]:
    last_targetable_origin = data["date"].max() - pd.Timedelta(weeks=max_lead)
    dates = sorted(data.loc[data["date"].between(start, end) & (data["date"] <= last_targetable_origin), "date"].drop_duplicates())
    if not dates:
        return []
    selected = dates[::step_weeks]
    if selected[-1] != dates[-1]:
        selected.append(dates[-1])
    return selected


def _seasonal_lookup(data: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    return {
        (str(row.ward_id), pd.Timestamp(row.date)): float(row.dengue_cases)
        for row in data[["ward_id", "date", "dengue_cases"]].itertuples(index=False)
    }


def seasonal_naive_predictions(
    rows: pd.DataFrame, *, horizon_weeks: int,
    seasonal: dict[tuple[str, pd.Timestamp], float],
) -> pd.Series:
    """Predict from same-ward observations 52 weeks before each target date."""
    values: list[float] = []
    for row in rows.itertuples(index=False):
        target_date = pd.Timestamp(row.date) + pd.Timedelta(weeks=horizon_weeks)
        seasonal_date = target_date - pd.Timedelta(weeks=52)
        value = seasonal.get((str(row.ward_id), seasonal_date))
        if value is None:
            raise ValueError(
                f"Seasonal-naive value is unavailable for ward {row.ward_id!r} at {seasonal_date.date()}"
            )
        values.append(float(value))
    return pd.Series(values, index=rows.index, name="prediction")


def eligible_training_rows(
    dataset: pd.DataFrame, *, horizon: int, information_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return only supervised rows whose labels existed by an issuance cutoff.

    The strict origin-date check is intentionally redundant with target-date
    alignment: it prevents any row originating on/after the simulated cutoff
    from entering training, even if malformed target dates are supplied.
    """
    target_date = f"target_date_{horizon}w"
    target = f"target_{horizon}w"
    if target_date not in dataset or target not in dataset:
        raise ValueError(f"Dataset has no target columns for {horizon} weeks")
    cutoff = pd.Timestamp(information_date)
    return dataset.loc[
        dataset[target].notna()
        & (dataset["date"] < cutoff)
        & (dataset[target_date] <= cutoff)
    ].copy()


def _prediction_rows_for_origin(
    *, data: pd.DataFrame, dataset: pd.DataFrame, horizon: int,
    issue_date: pd.Timestamp, information_delay_weeks: int,
    model_names: tuple[str, ...], seed: int, iterations: int,
    seasonal: dict[tuple[str, pd.Timestamp], float],
) -> list[dict[str, Any]]:
    """Fit at one origin and score the same wards, targets and baselines."""
    feature_date = issue_date - pd.Timedelta(weeks=information_delay_weeks)
    model_lead = horizon + information_delay_weeks
    target_col = f"target_{model_lead}w"
    target_date_col = f"target_date_{model_lead}w"
    train = eligible_training_rows(dataset, horizon=model_lead, information_date=feature_date)
    evaluation = dataset.loc[
        (dataset["date"] == feature_date)
        & (dataset[target_date_col] == issue_date + pd.Timedelta(weeks=horizon))
    ].copy()
    if train.empty:
        raise ValueError(f"No eligible training labels at information date {feature_date.date()}")
    if evaluation.empty:
        return []

    target_date = issue_date + pd.Timedelta(weeks=horizon)
    seasonal_date = target_date - pd.Timedelta(weeks=52)
    history = {
        str(ward): group.loc[group["date"] <= feature_date, "dengue_cases"]
        for ward, group in data.groupby("ward_id", sort=False, observed=True)
    }
    rows: list[dict[str, Any]] = []
    for index, sample in evaluation.iterrows():
        seasonal_value = seasonal.get((str(sample["ward_id"]), seasonal_date))
        if seasonal_value is None:
            continue
        common = {
            "ward_id": str(sample["ward_id"]), "city": str(sample["city"]),
            "forecast_origin_date": issue_date.date().isoformat(),
            "feature_information_date": feature_date.date().isoformat(),
            "training_label_cutoff": feature_date.date().isoformat(),
            "target_date": target_date.date().isoformat(),
            "actual_target": float(sample[target_col]), "horizon_weeks": horizon,
            "information_delay_weeks": information_delay_weeks,
            "target_month": int(target_date.month), "target_year": int(target_date.year),
        }
        ward_history = history[str(sample["ward_id"])]
        predictions: dict[str, float] = {
            "persistence": float(sample["dengue_cases"]),
            "seasonal_naive": float(seasonal_value),
        }
        risk_reference = ward_history
        for model_name, prediction in predictions.items():
            display = display_case_count(prediction)
            predicted_risk = classify_risk(display, risk_reference)["level"]
            actual_risk = classify_risk(float(sample[target_col]), risk_reference)["level"]
            rows.append({**common, "model": model_name, "prediction": prediction,
                         "predicted_risk": predicted_risk, "actual_risk": actual_risk})

    for family in model_names:
        fitted = fit_candidate(
            build_candidate(family, seed=seed, iterations=iterations), family,
            train.loc[:, FEATURE_COLUMNS], train[target_col].astype(float),
        )
        predicted = predict_candidate(fitted, family, evaluation.loc[:, FEATURE_COLUMNS])
        by_index = predicted.to_dict()
        for index, sample in evaluation.iterrows():
            seasonal_value = seasonal.get((str(sample["ward_id"]), seasonal_date))
            if seasonal_value is None:
                continue
            prediction = float(by_index[index])
            ward_history = history[str(sample["ward_id"])]
            rows.append({
                "ward_id": str(sample["ward_id"]), "city": str(sample["city"]),
                "forecast_origin_date": issue_date.date().isoformat(),
                "feature_information_date": feature_date.date().isoformat(),
                "training_label_cutoff": feature_date.date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "actual_target": float(sample[target_col]), "horizon_weeks": horizon,
                "information_delay_weeks": information_delay_weeks,
                "target_month": int(target_date.month), "target_year": int(target_date.year),
                "model": family, "prediction": prediction,
                "predicted_risk": classify_risk(display_case_count(prediction), ward_history)["level"],
                "actual_risk": classify_risk(float(sample[target_col]), ward_history)["level"],
            })
    return rows


def _metrics_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    group_columns = ["horizon_weeks", "model"]
    if "partition" in predictions:
        group_columns.append("partition")
    for group_key, group in predictions.groupby(group_columns, sort=True):
        if len(group_columns) == 3:
            horizon, model, partition = group_key
        else:
            horizon, model = group_key
            partition = "unspecified"
        metrics = calculate_metrics(group.actual_target, group.prediction)
        metric_rows.append({
            "horizon_weeks": int(horizon), "model": model, "partition": partition, "rows": len(group),
            "forecast_origins": group.forecast_origin_date.nunique(),
            "origin_start": group.forecast_origin_date.min(), "origin_end": group.forecast_origin_date.max(),
            "target_start": group.target_date.min(), "target_end": group.target_date.max(),
            "june_october_rows": int(group.target_month.between(6, 10).sum()), **metrics,
        })
        actual_idx = group.actual_risk.map(risk_level_index)
        predicted_idx = group.predicted_risk.map(risk_level_index)
        actual_alert = actual_idx >= risk_level_index("HIGH")
        predicted_alert = predicted_idx >= risk_level_index("HIGH")
        tp = int((actual_alert & predicted_alert).sum())
        risk_rows.append({
            "horizon_weeks": int(horizon), "model": model, "partition": partition, "rows": len(group),
            "exact_tier_accuracy": float((actual_idx == predicted_idx).mean()),
            "within_one_tier_accuracy": float(((actual_idx - predicted_idx).abs() <= 1).mean()),
            "high_critical_precision": float(tp / predicted_alert.sum()) if predicted_alert.sum() else float("nan"),
            "high_critical_recall": float(tp / actual_alert.sum()) if actual_alert.sum() else float("nan"),
            "critical_recall": float(((actual_idx == 3) & (predicted_idx == 3)).sum() / (actual_idx == 3).sum()) if (actual_idx == 3).sum() else float("nan"),
            "missed_critical_count": int(((actual_idx == 3) & (predicted_idx != 3)).sum()),
            "actual_high_critical_count": int(actual_alert.sum()), "predicted_high_critical_count": int(predicted_alert.sum()),
            "warning": "Experimental ward-relative tiers; not clinically validated alerts.",
        })
        stratified = group.copy()
        stratified["target_season"] = np.where(
            stratified.target_month.between(6, 10), "June-October", "Other months",
        )
        for dimension, key in (("city", "city"), ("ward", "ward_id"),
                               ("season", "target_season"), ("year", "target_year")):
            for value, subset in stratified.groupby(key, sort=True):
                sub_metrics = calculate_metrics(subset.actual_target, subset.prediction)
                grouped_rows.append({"horizon_weeks": int(horizon), "model": model, "partition": partition,
                                     "dimension": dimension, "group": str(value),
                                     "rows": len(subset), **sub_metrics})
    return pd.DataFrame(metric_rows), pd.DataFrame(risk_rows), pd.DataFrame(grouped_rows)


def run_training(
    input_path: Path = DEFAULT_INPUT, output_dir: Path | None = None,
    horizons: tuple[int, ...] = SUPPORTED_HORIZONS, seed: int = 42,
    iterations: int = 60, config: SplitConfig | None = None,
    reporting_delay_sensitivity: bool = True,
) -> Path:
    """Select models by validation rolling origins, test once, then fit artifacts."""
    config = config or SplitConfig(*(pd.Timestamp(value) for value in
        ("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31")))
    config.validate()
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed input CSV does not exist: {input_path}")
    if not horizons or any(h not in SUPPORTED_HORIZONS for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError(f"Forecast horizons must be a non-empty subset of {SUPPORTED_HORIZONS}")
    if seed < 0 or iterations < 1:
        raise ValueError("seed must be non-negative and iterations must be positive")

    data = validate_dataset(pd.read_csv(input_path))
    validate_feature_schema(data)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    for name in CATEGORICAL_FEATURES:
        data[name] = data[name].astype(str)
    data = data.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)
    target_horizons = sorted(set(horizons) | ({h + 1 for h in horizons} if reporting_delay_sensitivity else set()))
    dataset = construct_targets(data, target_horizons)
    seasonal = _seasonal_lookup(data)
    max_lead = max(horizons)
    validation_origins = _origin_dates(data, config.validation_start, config.validation_end, max_lead, config.origin_step_weeks)
    test_origins = _origin_dates(data, config.test_start, config.test_end, max_lead, config.origin_step_weeks)
    if not validation_origins or not test_origins:
        raise ValueError("Rolling-origin validation and test periods must contain targetable weekly origins")

    output_dir = output_dir or (DEFAULT_OUTPUT_PARENT / datetime.now(timezone.utc).strftime("aedisight_%Y%m%dT%H%M%S_%fZ"))
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Run output already exists; choose a new directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        model_dir, result_dir = staging / "models", staging / "results"
        model_dir.mkdir(); result_dir.mkdir()
        all_validation: list[pd.DataFrame] = []
        all_test: list[pd.DataFrame] = []
        sensitivity_rows: list[dict[str, Any]] = []
        selected_by_horizon: dict[int, str] = {}
        metadata: dict[str, Any] = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "evaluation_method": EVALUATION_METHOD,
            "run_created_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": _portable_input_label(input_path),
            "input_rows": len(data), "ward_count": int(data.ward_id.nunique()), "city_count": int(data.city.nunique()),
            "input_date_min": str(data.date.min().date()), "input_date_max": str(data.date.max().date()),
            "training_data_cutoff": str(data.date.max().date()),
            "target_definition": TARGET_DEFINITION,
            "dengue_cases_semantics": "Weekly new dengue cases for the ward and associated observation week; not cumulative.",
            "reporting_period_convention": "The dataset observation date identifies the seven-day reporting period; week start/end convention is unspecified.",
            "feature_names": list(FEATURE_COLUMNS), "categorical_features": list(CATEGORICAL_FEATURES),
            "random_seed": seed, "iterations": iterations,
            "origin_step_weeks": config.origin_step_weeks,
            "validation_origins": [d.date().isoformat() for d in validation_origins],
            "test_origins": [d.date().isoformat() for d in test_origins],
            "feature_availability_assumption": "At an issuance date, the latest weekly observation on or before that date is available; the primary evaluation assumes zero publication delay. A one-week delayed-information sensitivity is also evaluated.",
            "selection_metric": "Pooled validation rolling-origin MAE; final test metrics are not used for selection.",
            "baseline_definitions": {
                "persistence": "Last observed weekly new-case count available at the forecast origin.",
                "seasonal_naive": "Observed value for the same ward and target-date weekday 52 weeks before the target date.",
            },
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                         "scikit_learn": sklearn.__version__, "catboost": __import__("catboost").__version__, "joblib": joblib.__version__},
            "horizons": {},
        }

        for horizon in horizons:
            print(f"Horizon {horizon}w: rolling-origin validation ({len(validation_origins)} origins)", flush=True)
            validation_rows: list[dict[str, Any]] = []
            for origin in validation_origins:
                validation_rows.extend(_prediction_rows_for_origin(
                    data=data, dataset=dataset, horizon=horizon, issue_date=origin,
                    information_delay_weeks=0, model_names=DEFAULT_CANDIDATES,
                    seed=seed, iterations=iterations, seasonal=seasonal,
                ))
            validation = pd.DataFrame(validation_rows)
            if validation.empty:
                raise ValueError(f"No validation predictions produced for {horizon} weeks")
            validation["partition"] = "validation"
            all_validation.append(validation)
            val_metrics, _, _ = _metrics_table(validation)
            candidates = []
            for _, row in val_metrics.loc[val_metrics.horizon_weeks.eq(horizon)].iterrows():
                candidates.append((float(row.mae), str(row.model), None, row.to_dict()))
            selected_mae, selected_name, _, selected_validation_metrics = select_candidate(candidates)
            selected_by_horizon[horizon] = selected_name

            print(f"Horizon {horizon}w: rolling-origin test ({len(test_origins)} origins), selected {selected_name}", flush=True)
            test_rows: list[dict[str, Any]] = []
            for origin in test_origins:
                test_rows.extend(_prediction_rows_for_origin(
                    data=data, dataset=dataset, horizon=horizon, issue_date=origin,
                    information_delay_weeks=0, model_names=DEFAULT_CANDIDATES,
                    seed=seed, iterations=iterations, seasonal=seasonal,
                ))
            test = pd.DataFrame(test_rows)
            if test.empty:
                raise ValueError(f"No test predictions produced for {horizon} weeks")
            test["partition"] = "test"
            all_test.append(test)

            selected_rows = test.loc[test.model.eq(selected_name), [
                "ward_id", "city", "forecast_origin_date", "target_date", "actual_target", "prediction",
            ]].rename(columns={"prediction": "ml_prediction"})
            base_rows = test.loc[test.model.isin(BASELINES)].pivot(
                index=["ward_id", "city", "forecast_origin_date", "target_date", "actual_target"],
                columns="model", values="prediction",
            ).reset_index()
            legacy_rows = selected_rows.merge(base_rows, on=[
                "ward_id", "city", "forecast_origin_date", "target_date", "actual_target",
            ], validate="one_to_one")
            legacy_rows["horizon_weeks"] = horizon
            legacy_rows.to_csv(result_dir / f"test_predictions_{horizon}w.csv", index=False)

            if reporting_delay_sensitivity:
                delay_rows: list[dict[str, Any]] = []
                for origin in test_origins:
                    delay_rows.extend(_prediction_rows_for_origin(
                        data=data, dataset=dataset, horizon=horizon, issue_date=origin,
                        information_delay_weeks=1, model_names=(selected_name,) if selected_name in DEFAULT_CANDIDATES else (),
                        seed=seed, iterations=iterations, seasonal=seasonal,
                    ))
                delay_frame = pd.DataFrame(delay_rows)
                if not delay_frame.empty:
                    delay_metrics, _, _ = _metrics_table(delay_frame)
                    sensitivity_rows.extend({**row.to_dict(), "assumed_reporting_delay_weeks": 1,
                                             "selected_validation_model": selected_name}
                                            for _, row in delay_metrics.iterrows())

            valid_training = dataset.loc[
                dataset[f"target_{horizon}w"].notna()
                & (dataset[f"target_date_{horizon}w"] <= data.date.max())
                & (dataset.date < data.date.max())
            ].copy()
            final_rows = data.loc[data.date.eq(data.date.max())].copy()
            sample = final_rows.loc[:, list(FEATURE_COLUMNS)]
            if selected_name in DEFAULT_CANDIDATES:
                final_model = fit_candidate(
                    build_candidate(selected_name, seed=seed, iterations=iterations), selected_name,
                    valid_training.loc[:, FEATURE_COLUMNS], valid_training[f"target_{horizon}w"].astype(float),
                )
                expected_sample = predict_candidate(final_model, selected_name, sample)
            elif selected_name == "persistence":
                final_model = None
                expected_sample = predict_candidate(None, "persistence", sample)
            elif selected_name == "seasonal_naive":
                final_model = None
                expected_sample = seasonal_naive_predictions(
                    final_rows, horizon_weeks=horizon, seasonal=seasonal,
                )
            else:
                raise ValueError(f"Unsupported selected model family {selected_name!r}")
            bundle = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "model": final_model, "model_family": selected_name,
                "horizon_weeks": horizon, "feature_names": list(FEATURE_COLUMNS),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "target_definition": TARGET_DEFINITION,
                "target_column": f"target_{horizon}w",
                "training_data_cutoff": str(data.date.max().date()),
                "feature_availability_assumption": metadata["feature_availability_assumption"],
                "prediction_constraint": "continuous non-negative output for evaluation; rounded half-up to integer only for display and risk classification",
            }
            artifact_rel = f"models/model_{horizon}w.joblib"
            artifact_path = staging / artifact_rel
            temporary_artifact = artifact_path.with_suffix(".joblib.tmp")
            joblib.dump(bundle, temporary_artifact)
            os.replace(temporary_artifact, artifact_path)
            reloaded = joblib.load(artifact_path)
            if selected_name == "seasonal_naive":
                reloaded_prediction = seasonal_naive_predictions(
                    final_rows, horizon_weeks=horizon, seasonal=seasonal,
                )
            else:
                reloaded_prediction = predict_candidate(reloaded["model"], selected_name, sample)
            if not np.allclose(expected_sample.to_numpy(), reloaded_prediction.to_numpy()):
                raise RuntimeError(f"Artifact reload check failed for {horizon}w")
            importance = _feature_importances(final_model, selected_name)
            importance.to_csv(result_dir / f"feature_importance_{horizon}w.csv", index=False)
            metadata["horizons"][str(horizon)] = {
                "model_artifact": artifact_rel, "model_family": selected_name,
                "horizon_weeks": horizon, "feature_names": list(FEATURE_COLUMNS),
                "feature_order": list(FEATURE_COLUMNS), "categorical_features": list(CATEGORICAL_FEATURES),
                "target_column": f"target_{horizon}w", "target_definition": TARGET_DEFINITION,
                "training_data_cutoff": str(data.date.max().date()),
                "training_rows": int(len(valid_training)),
                "validation_selection_mae": selected_mae,
                "selection_metrics": selected_validation_metrics,
                "artifact_reload_prediction_check": "passed",
            }

        validation_all = pd.concat(all_validation, ignore_index=True)
        test_all = pd.concat(all_test, ignore_index=True)
        validation_all.to_csv(result_dir / "validation_predictions.csv", index=False)
        test_all.to_csv(result_dir / "test_predictions.csv", index=False)
        validation_metrics, validation_risk, validation_grouped = _metrics_table(validation_all)
        test_metrics, test_risk, test_grouped = _metrics_table(test_all)
        validation_metrics["selected"] = validation_metrics.apply(
            lambda row: row.model == selected_by_horizon.get(int(row.horizon_weeks)), axis=1,
        )
        test_metrics["selected"] = test_metrics.apply(
            lambda row: row.model == selected_by_horizon.get(int(row.horizon_weeks)), axis=1,
        )
        for horizon, model_name in selected_by_horizon.items():
            selected_test = test_metrics.loc[
                test_metrics.horizon_weeks.eq(horizon) & test_metrics.model.eq(model_name)
            ]
            metadata["horizons"][str(horizon)]["test_metrics"] = (
                selected_test.iloc[0].to_dict() if not selected_test.empty else {}
            )
        validation_metrics.to_csv(result_dir / "validation_model_comparison.csv", index=False)
        test_metrics.to_csv(result_dir / "final_test_metrics.csv", index=False)
        pd.concat([validation_risk.assign(partition="validation"), test_risk.assign(partition="test")], ignore_index=True).to_csv(result_dir / "risk_metrics.csv", index=False)
        pd.concat([validation_grouped.assign(partition="validation"), test_grouped.assign(partition="test")], ignore_index=True).to_csv(result_dir / "stratified_metrics.csv", index=False)
        pd.DataFrame(sensitivity_rows).to_csv(result_dir / "reporting_delay_sensitivity.csv", index=False)
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        (staging / "README.txt").write_text(
            "Leakage-aware expanding-window run. Targets are weekly new ward cases for the target observation week. "
            "Validation selects models; final test origins are independent. See metadata.json and results/.\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"Run output appeared during training: {output_dir}")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weekly dengue models using expanding-window rolling-origin validation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="New unique run directory; never overwrites an existing run")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(SUPPORTED_HORIZONS), choices=SUPPORTED_HORIZONS)
    parser.add_argument("--validation-start", type=_date, default=pd.Timestamp("2024-01-01"))
    parser.add_argument("--validation-end", type=_date, default=pd.Timestamp("2024-12-31"))
    parser.add_argument("--test-start", type=_date, default=pd.Timestamp("2025-01-01"))
    parser.add_argument("--test-end", type=_date, default=pd.Timestamp("2025-12-31"))
    parser.add_argument("--origin-step-weeks", type=int, default=4, help="Forecast-origin sampling interval; default is approximately monthly")
    parser.add_argument("--iterations", type=int, default=60, help="Tree count; kept modest for local CPU use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-reporting-delay-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.origin_step_weeks < 1:
        parser.error("--iterations and --origin-step-weeks must be positive")
    return args


def main() -> int:
    args = parse_args()
    config = SplitConfig(pd.Timestamp("2023-12-31"), args.validation_start, args.validation_end,
                         args.test_start, args.test_end, args.origin_step_weeks)
    run_dir = run_training(args.input, args.output, tuple(args.horizons), args.seed,
                           args.iterations, config, not args.no_reporting_delay_sensitivity)
    print(f"Training run completed: {run_dir}")
    print(pd.read_csv(run_dir / "results" / "final_test_metrics.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

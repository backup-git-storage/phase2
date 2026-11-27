"""Behavior-focused tests for the offline dengue training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataset
from src.feature_engineering import FEATURE_COLUMNS, construct_targets, validate_feature_schema
from src.preprocessing import build_candidate, fit_candidate, predict_candidate
from src.risk_engine import classify_risk, risk_thresholds
from src.train import (
    SplitConfig, calculate_metrics, display_case_count, eligible_training_rows, run_training,
    seasonal_naive_predictions, select_candidate,
)
from src.validation import validate_dataset
from src import predict as forecast_service


def make_raw_frame(*, start: str = "2022-10-03", weeks: int = 170, wards: int = 3) -> pd.DataFrame:
    dates = pd.date_range(start, periods=weeks, freq="7D")
    rows: list[dict[str, object]] = []
    for ward in range(wards):
        for index, date in enumerate(dates):
            cases = (index * (ward + 2) + index // 9 + ward) % 23
            rows.append({
                "date": date.date().isoformat(), "city": f"City-{ward % 2}",
                "ward_id": f"ward-{ward}", "population_density_sqkm": 1200 + ward * 100,
                "water_logging_risk_index": 0.3 + ward * 0.1,
                "sanitation_score": 0.7 - ward * 0.1,
                "avg_temp_c": 25.0 + 0.02 * index + ward * 0.1,
                "max_temp_c": 30.0 + 0.02 * index + ward * 0.1,
                "min_temp_c": 20.0 + 0.02 * index + ward * 0.1,
                "rainfall_mm": float(index + ward * 100),
                "relative_humidity_pct": 60.0 + index % 20,
                "dengue_cases": int(cases),
            })
    return pd.DataFrame(rows)


def clean_frame(tmp_path: Path, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "synthetic_raw.csv"
    output_path = tmp_path / "synthetic_processed.csv"
    (raw if raw is not None else make_raw_frame()).to_csv(input_path, index=False)
    clean_dataset(input_path, output_path)
    return pd.read_csv(output_path)


def test_validation_rejects_schema_dates_numeric_duplicates_ward_and_gaps() -> None:
    original = make_raw_frame(weeks=8, wards=2)
    with pytest.raises(ValueError, match="Missing required source columns"):
        validate_dataset(original.drop(columns="city"))
    invalid_date = original.copy()
    invalid_date.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="Invalid date"):
        validate_dataset(invalid_date)
    invalid_number = original.copy()
    invalid_number.loc[0, "rainfall_mm"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset(invalid_number)
    duplicate = pd.concat([original, original.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset(duplicate)
    inconsistent = original.copy()
    inconsistent.loc[inconsistent.index[inconsistent.ward_id == "ward-0"][0], "city"] = "Other City"
    with pytest.raises(ValueError, match="multiple cities"):
        validate_dataset(inconsistent)
    gap = original.drop(index=2)
    with pytest.raises(ValueError, match="continuity"):
        validate_dataset(gap)
    invalid_target = original.copy()
    invalid_target.loc[0, "dengue_cases"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_dataset(invalid_target)
    fractional_target = original.copy()
    fractional_target["dengue_cases"] = fractional_target["dengue_cases"].astype(float)
    fractional_target.loc[0, "dengue_cases"] = 1.25
    with pytest.raises(ValueError, match="non-integer"):
        validate_dataset(fractional_target)
    case_inconsistent = original.copy()
    case_inconsistent.loc[0, "ward_id"] = "WARD-0"
    with pytest.raises(ValueError, match="case-inconsistent"):
        validate_dataset(case_inconsistent)


def test_cleaner_features_are_ordered_historical_and_ward_local(tmp_path: Path) -> None:
    raw = make_raw_frame(weeks=12, wards=2).sample(frac=1, random_state=7)
    processed = clean_frame(tmp_path, raw)
    assert len(processed) == 16  # Four early observations per ward lack history.
    assert processed.columns[-1] == "dengue_cases"
    assert processed.groupby("ward_id").date.apply(lambda values: values.is_monotonic_increasing).all()
    ward = processed.loc[processed.ward_id == "ward-0"].reset_index(drop=True)
    # Cleaner omits the first four rows; row 0 is origin week 4.
    assert ward.loc[0, "rain_lag_1w"] == pytest.approx(3.0)
    assert ward.loc[0, "rain_sum_4w"] == pytest.approx(sum(range(4)))
    assert ward.loc[0, "dengue_cases_lag_2w"] == int(raw.loc[(raw.ward_id == "ward-0")].sort_values("date").iloc[2].dengue_cases)
    ward_one = processed.loc[processed.ward_id == "ward-1"].reset_index(drop=True)
    assert ward_one.loc[0, "rain_lag_1w"] == pytest.approx(103.0)
    assert ward_one.loc[0, "rain_sum_4w"] == pytest.approx(sum(range(100, 104)))
    changed = raw.copy()
    cutoff = pd.Timestamp("2022-10-03") + pd.Timedelta(weeks=7)
    changed.loc[pd.to_datetime(changed.date) > cutoff, "rainfall_mm"] += 10000
    changed_processed = clean_frame(tmp_path / "changed", changed)
    before = processed.loc[pd.to_datetime(processed.date) <= cutoff].reset_index(drop=True)
    changed_before = changed_processed.loc[pd.to_datetime(changed_processed.date) <= cutoff].reset_index(drop=True)
    pd.testing.assert_series_equal(before["rain_lag_1w"], changed_before["rain_lag_1w"])
    pd.testing.assert_series_equal(before["rain_sum_4w"], changed_before["rain_sum_4w"])


def test_target_construction_horizons_dates_and_ward_boundaries(tmp_path: Path) -> None:
    processed = clean_frame(tmp_path, make_raw_frame(weeks=14, wards=2))
    validate_feature_schema(processed)
    targets = construct_targets(processed)
    for horizon in (2, 3, 4):
        ward = targets.loc[targets.ward_id == "ward-0"].reset_index(drop=True)
        assert ward.loc[0, f"target_{horizon}w"] == ward.loc[horizon, "dengue_cases"]
        assert ward.loc[0, f"target_date_{horizon}w"] == ward.loc[horizon, "date"]
        assert ward[f"target_{horizon}w"].iloc[-horizon:].isna().all()
    last = targets.loc[targets.ward_id == "ward-0"].iloc[-1]
    assert pd.isna(last["target_2w"])
    assert targets.loc[targets.ward_id == "ward-1", "target_2w"].notna().sum() == len(processed.loc[processed.ward_id == "ward-1"]) - 2


def test_rolling_origin_training_labels_are_available_by_cutoff(tmp_path: Path) -> None:
    processed = clean_frame(tmp_path, make_raw_frame(weeks=30, wards=2))
    data = construct_targets(processed, (2,))
    cutoff = pd.Timestamp("2022-12-12")
    eligible = eligible_training_rows(data, horizon=2, information_date=cutoff)
    assert not eligible.empty
    assert (eligible.date < cutoff).all()
    assert (eligible.target_date_2w <= cutoff).all()
    assert eligible.target_date_2w.max() <= cutoff
    with pytest.raises(ValueError, match="no target columns"):
        eligible_training_rows(data, horizon=5, information_date=cutoff)


def test_candidate_selection_includes_persistence_and_prefers_it_on_ties() -> None:
    candidates = [
        (0.8, "catboost", object(), {"mae": 0.8}),
        (0.7, "persistence", None, {"mae": 0.7}),
    ]
    assert select_candidate(candidates)[1] == "persistence"
    tied = [(0.7, "catboost", object(), {}), (0.7, "persistence", None, {})]
    assert select_candidate(tied)[1] == "persistence"


def test_display_case_count_keeps_nonnegative_half_up_rounding_and_rejects_nonfinite() -> None:
    for prediction, expected in (
        (0.15, 0), (0.42, 0), (0.50, 1), (0.80, 1),
        (1.20, 1), (10.87, 11), (-0.8, 0),
    ):
        assert display_case_count(prediction) == expected
    for prediction in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            display_case_count(prediction)


def test_replay_default_prefers_august_18_for_each_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        forecast_service,
        "replay_origins",
        lambda horizon: ["2025-08-04", "2025-08-18", "2025-09-15"],
    )
    for horizon in (2, 3, 4):
        bounds = forecast_service.replay_origin_bounds(horizon)
        assert bounds["default"] == "2025-08-18"
        assert "2025-08-18" in bounds["available"]


def test_replay_default_uses_nearest_valid_origin_when_preferred_date_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        forecast_service,
        "replay_origins",
        lambda _horizon: ["2025-08-04", "2025-09-01"],
    )
    assert forecast_service.replay_origin_bounds(3)["default"] == "2025-08-04"


def test_available_horizons_depend_on_saved_models_not_test_prediction_csvs(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[int] = []

    def load_saved_bundle(horizon: int) -> dict:
        loaded.append(horizon)
        return {"horizon_weeks": horizon}

    monkeypatch.setattr(forecast_service, "load_model_bundle", load_saved_bundle)
    monkeypatch.setattr(
        forecast_service,
        "replay_origins",
        lambda _horizon: (_ for _ in ()).throw(AssertionError("flexible mode must not require saved test origins")),
    )
    assert forecast_service.get_available_horizons() == [2, 3, 4]
    assert loaded == [2, 3, 4]


def test_seasonal_naive_uses_target_date_minus_52_weeks() -> None:
    rows = pd.DataFrame({
        "ward_id": ["ward-1"], "date": [pd.Timestamp("2025-01-06")],
    })
    lookup = {("ward-1", pd.Timestamp("2024-02-05")): 7.0}
    prediction = seasonal_naive_predictions(rows, horizon_weeks=4, seasonal=lookup)
    assert prediction.tolist() == [7.0]
    with pytest.raises(ValueError, match="unavailable"):
        seasonal_naive_predictions(rows, horizon_weeks=3, seasonal=lookup)


def test_risk_tiers_thresholds_and_invalid_values() -> None:
    history = pd.Series([0] + list(range(1, 21)))
    thresholds = risk_thresholds(history)
    assert classify_risk(0, history)["level"] == "LOW"
    assert classify_risk(thresholds["low_max"], history)["level"] == "LOW"
    assert classify_risk(thresholds["low_max"] + 0.5, history)["level"] == "MEDIUM"
    assert classify_risk(thresholds["medium_max"] + 0.5, history)["level"] == "HIGH"
    assert classify_risk(thresholds["high_max"] + 0.5, history)["level"] == "CRITICAL"
    with pytest.raises(ValueError, match="finite, non-negative"):
        classify_risk(float("nan"), history)
    with pytest.raises(ValueError, match="without historical"):
        classify_risk(1, pd.Series(dtype=float))


def test_metrics_baseline_clipping_unknown_categories_and_reproducibility(tmp_path: Path) -> None:
    metrics = calculate_metrics([0, 2, 4], [0, 1, 6])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["r2"] == pytest.approx(1 - 5 / 8)
    assert np.isnan(calculate_metrics([2, 2], [1, 3])["r2"])
    with pytest.raises(ValueError, match="non-finite"):
        calculate_metrics([1.0], [np.nan])
    with pytest.raises(ValueError, match="non-empty"):
        calculate_metrics([], [])
    processed = clean_frame(tmp_path, make_raw_frame(weeks=20, wards=3))
    validate_feature_schema(processed)
    data = construct_targets(processed, (2,))
    features = data.loc[data.target_2w.notna(), list(FEATURE_COLUMNS)]
    target = data.loc[data.target_2w.notna(), "target_2w"]
    model_a = fit_candidate(build_candidate("gradient_boosting", seed=42, iterations=5), "gradient_boosting", features.iloc[:30], target.iloc[:30])
    model_b = fit_candidate(build_candidate("gradient_boosting", seed=42, iterations=5), "gradient_boosting", features.iloc[:30], target.iloc[:30])
    unknown = features.iloc[[30]].copy()
    unknown["ward_id"] = "unseen-ward"
    p_a = predict_candidate(model_a, "gradient_boosting", unknown)
    p_b = predict_candidate(model_b, "gradient_boosting", unknown)
    assert (p_a >= 0).all()
    assert np.allclose(p_a, p_b)
    assert p_a.iloc[0] == pytest.approx(p_b.iloc[0])

    class NegativeEstimator:
        def predict(self, matrix: pd.DataFrame) -> np.ndarray:
            return np.full(len(matrix), -3.0)

    clipped = predict_candidate(NegativeEstimator(), "gradient_boosting", unknown)
    assert clipped.eq(0.0).all()
    persistence = predict_candidate(None, "persistence", unknown)
    assert persistence.tolist() == unknown["dengue_cases"].astype(float).tolist()


def test_end_to_end_smoke_outputs_and_artifact_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    make_raw_frame(start="2022-01-03", weeks=208, wards=3).to_csv(raw_path, index=False)
    clean_dataset(raw_path, processed_path)
    run_path = run_training(
        processed_path, tmp_path / "run", horizons=(2, 3, 4), seed=17, iterations=3,
        config=SplitConfig(*(pd.Timestamp(day) for day in
            ("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31")),
            origin_step_weeks=52),
    )
    expected = {
        "metadata.json", "models/model_2w.joblib", "models/model_3w.joblib", "models/model_4w.joblib",
        "results/validation_model_comparison.csv", "results/final_test_metrics.csv",
        "results/test_predictions_2w.csv", "results/test_predictions_3w.csv", "results/test_predictions_4w.csv",
        "results/risk_metrics.csv", "results/reporting_delay_sensitivity.csv",
    }
    assert {str(path.relative_to(run_path)) for path in run_path.rglob("*") if path.is_file()} >= expected
    monkeypatch.setattr(forecast_service, "DATA_PATH", processed_path)
    monkeypatch.setattr(forecast_service, "RUNS_DIR", tmp_path)
    forecast_service.load_raw_data.cache_clear()
    forecast_service.load_model_bundle.cache_clear()
    forecast_service._load_replay_predictions.cache_clear()


    metadata = json.loads((run_path / "metadata.json").read_text())
    assert metadata["input_path"] == processed_path.name
    assert not Path(metadata["input_path"]).is_absolute()
    processed = pd.read_csv(processed_path)
    comparisons = pd.read_csv(run_path / "results/validation_model_comparison.csv")
    validation_predictions = pd.read_csv(run_path / "results/validation_predictions.csv")
    metrics = pd.read_csv(run_path / "results/final_test_metrics.csv")
    assert comparisons.loc[comparisons.partition == "validation"].groupby("horizon_weeks").selected.sum().eq(1).all()
    for horizon in (2, 3, 4):
        group = comparisons.loc[(comparisons.horizon_weeks == horizon) & (comparisons.partition == "validation")]
        assert group.loc[group.selected, "model"].iloc[0] == group.sort_values(["mae", "model"]).iloc[0].model
        horizon_rows = validation_predictions.loc[validation_predictions.horizon_weeks == horizon]
        assert (pd.to_datetime(horizon_rows.training_label_cutoff) <= pd.to_datetime(horizon_rows.forecast_origin_date)).all()
        assert pd.to_datetime(horizon_rows.forecast_origin_date).eq(
            pd.to_datetime(horizon_rows.feature_information_date)
        ).all()
        samples = horizon_rows.groupby("model").apply(
            lambda rows: set(zip(rows.ward_id, rows.forecast_origin_date, rows.target_date, rows.actual_target)),
            include_groups=False,
        )
        assert len(samples) >= 5
        assert all(sample_keys == samples.iloc[0] for sample_keys in samples.iloc[1:])
        result = pd.read_csv(run_path / f"results/test_predictions_{horizon}w.csv")
        assert (pd.to_datetime(result.target_date) > pd.to_datetime(result.forecast_origin_date)).all()
        assert (pd.to_datetime(result.target_date) - pd.to_datetime(result.forecast_origin_date)).dt.days.eq(horizon * 7).all()
        assert (result.ml_prediction >= 0).all()
        artifact = __import__("joblib").load(run_path / f"models/model_{horizon}w.joblib")
        assert artifact["feature_names"] == metadata["horizons"][str(horizon)]["feature_order"] == list(FEATURE_COLUMNS)
        assert metadata["feature_names"] == artifact["feature_names"]
        assert artifact["categorical_features"] == metadata["horizons"][str(horizon)]["categorical_features"]
        sample = result.iloc[0]
        model_input = processed.loc[
            (processed.ward_id == sample.ward_id)
            & (processed.date == sample.forecast_origin_date),
            list(FEATURE_COLUMNS),
        ]
        reloaded_prediction = predict_candidate(artifact["model"], artifact["model_family"], model_input)
        assert np.isfinite(reloaded_prediction.to_numpy()).all()
        assert metrics.loc[metrics.horizon_weeks == horizon, "partition"].iloc[0] == "test"
        live = forecast_service.generate_forecast("City-0", "ward-0", horizon, result.iloc[0].forecast_origin_date)
        assert live["rawPredictedCases"] == pytest.approx(result.iloc[0].ml_prediction, abs=1e-10)
        assert live["modelTrainingDataCutoff"] == result.iloc[0].forecast_origin_date
        assert live["horizonWeeks"] == horizon
        assert live["model"].startswith(artifact["model_family"])
        assert live["targetDate"] == (pd.Timestamp(live["observationDate"]) + pd.Timedelta(weeks=horizon)).date().isoformat()
    forecast_service.load_raw_data.cache_clear()
    forecast_service.load_model_bundle.cache_clear()
    forecast_service._load_replay_predictions.cache_clear()


def test_seasonal_naive_selected_artifact_reloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import train as training_service

    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    make_raw_frame(start="2022-01-03", weeks=208, wards=3).to_csv(raw_path, index=False)
    clean_dataset(raw_path, processed_path)
    monkeypatch.setattr(
        training_service,
        "select_candidate",
        lambda candidates: next(candidate for candidate in candidates if candidate[1] == "seasonal_naive"),
    )
    run_path = run_training(
        processed_path, tmp_path / "seasonal_run", horizons=(2,), seed=23, iterations=2,
        config=SplitConfig(*(pd.Timestamp(day) for day in
            ("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31")),
            origin_step_weeks=52),
        reporting_delay_sensitivity=False,
    )
    metadata = json.loads((run_path / "metadata.json").read_text())
    artifact = __import__("joblib").load(run_path / "models/model_2w.joblib")
    assert metadata["horizons"]["2"]["model_family"] == "seasonal_naive"
    assert artifact["model_family"] == "seasonal_naive"
    assert artifact["model"] is None
    assert artifact["artifact_schema_version"] == metadata["artifact_schema_version"]


def test_inference_uses_artifact_schema_origin_and_real_ward_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = clean_frame(tmp_path, make_raw_frame(start="2022-10-03", weeks=16, wards=2))
    data_path = tmp_path / "synthetic_processed.csv"
    processed.to_csv(data_path, index=False)
    monkeypatch.setattr(forecast_service, "DATA_PATH", data_path)
    run_dir = tmp_path / "synthetic_run"
    (run_dir / "results").mkdir(parents=True)
    origin = (pd.Timestamp("2022-10-03") + pd.Timedelta(weeks=11)).date().isoformat()
    target_date = (pd.Timestamp(origin) + pd.Timedelta(weeks=2)).date().isoformat()
    actual_target = processed.loc[
        (processed.ward_id == "ward-0") & (processed.date == target_date), "dengue_cases"
    ].iloc[0]
    pd.DataFrame([{
        "horizon_weeks": 2, "model": "test-estimator", "partition": "test",
        "ward_id": "ward-0", "forecast_origin_date": origin,
        "feature_information_date": origin, "training_label_cutoff": origin,
        "target_date": target_date, "actual_target": actual_target, "prediction": 2.5,
    }]).to_csv(run_dir / "results" / "test_predictions.csv", index=False)
    monkeypatch.setattr(forecast_service, "_run_directory", lambda horizon: run_dir)
    forecast_service.load_raw_data.cache_clear()
    forecast_service.load_model_bundle.cache_clear()
    forecast_service._load_replay_predictions.cache_clear()

    class FixedModel:
        inputs: list[pd.DataFrame] = []

        def predict(self, matrix: pd.DataFrame) -> np.ndarray:
            assert list(matrix.columns) == list(FEATURE_COLUMNS)
            self.inputs.append(matrix.copy())
            return np.array([float(matrix["dengue_cases_lag_1w"].iloc[0])])

    original_load_model_bundle = forecast_service.load_model_bundle
    monkeypatch.setattr(
        forecast_service,
        "load_model_bundle",
        lambda horizon: {
            "model": FixedModel(),
            "model_family": "test-estimator",
            "horizon_weeks": horizon,
            "feature_names": list(FEATURE_COLUMNS),
            "categorical_features": ["city", "ward_id"],
            "training_data_cutoff": str(pd.to_datetime(processed.date).max().date()),
            "artifact_schema_version": 2,
            "target_definition": "Weekly new dengue cases for the ward and associated observation week; not cumulative.",
        },
    )
    try:
        areas = forecast_service.get_supported_areas()
        assert areas == {"City-0": ["ward-0"], "City-1": ["ward-1"]}
        date_bounds = forecast_service.get_supported_date_bounds()
        ward_dates = pd.to_datetime(processed.loc[processed.ward_id == "ward-0", "date"]).sort_values().reset_index(drop=True)
        assert date_bounds["areas"]["City-0"]["ward-0"] == {
            "min": ward_dates.iloc[7].date().isoformat(),
            "max": ward_dates.iloc[-1].date().isoformat(),
        }

        requested_date = (pd.Timestamp(origin) + pd.Timedelta(days=2)).date().isoformat()
        result = forecast_service.generate_forecast("City-0", "ward-0", 2, requested_date)
        origin_features = processed.loc[
            (processed.ward_id == "ward-0") & (processed.date == origin),
            list(FEATURE_COLUMNS),
        ].iloc[0]
        expected_raw = float(origin_features["dengue_cases_lag_1w"])
        assert result["predictedCases"] == display_case_count(expected_raw)
        assert result["rawPredictedCases"] == expected_raw
        assert result["expectedPredictedCases"] == expected_raw
        assert result["rawPredictionIsClipped"] is False
        assert result["modelInputFeatureCount"] == result["expectedFeatureCount"] == len(FEATURE_COLUMNS)
        assert result["missingFeatureCount"] == 0
        assert result["horizonWeeks"] == 2
        assert result["requestedAsOf"] == requested_date
        assert result["observationDate"] == origin
        assert result["targetDate"] == (pd.Timestamp(origin) + pd.Timedelta(weeks=2)).date().isoformat()
        assert result["actualCases"] is not None
        assert result["dataIsHistorical"] is True
        assert result["isHistoricalReplay"] is False
        assert result["isRetrospectiveEstimate"] is True
        assert "not a leakage-safe historical backtest" in result["predictionWarning"]
        assert result["historicalReplayPartition"] is None
        assert result["modelTrainingDataThrough"] == pd.to_datetime(processed.date).max().date().isoformat()
        assert result["modelTrainingDataCutoff"] == pd.to_datetime(processed.date).max().date().isoformat()
        assert "not clinically validated" in result["riskWarning"]
        ward_history = processed.loc[
            (processed.ward_id == "ward-0") & (pd.to_datetime(processed.date) <= pd.Timestamp(origin)),
            "dengue_cases",
        ]
        assert result["riskThresholds"] == risk_thresholds(ward_history)
        assert result["dataFreshness"] == "stale"
        assert len(FixedModel.inputs[-1]) == 1
        assert list(FixedModel.inputs[-1].columns) == list(FEATURE_COLUMNS)

        strict = forecast_service.generate_forecast(
            "City-0", "ward-0", 2, origin, strict_evaluation=True,
        )
        assert strict["rawPredictedCases"] == 2.5
        assert strict["isHistoricalReplay"] is True
        assert strict["isRetrospectiveEstimate"] is False
        assert strict["historicalReplayPartition"] == "test"
        assert strict["modelTrainingDataThrough"] == origin

        future_modified = processed.copy()
        future_mask = pd.to_datetime(future_modified.date) > pd.Timestamp(origin)
        future_modified["dengue_cases"] = pd.to_numeric(future_modified.dengue_cases)
        future_modified["rainfall_mm"] = pd.to_numeric(future_modified.rainfall_mm)
        future_modified.loc[future_mask, "dengue_cases"] += 10
        future_modified.loc[future_mask, "rainfall_mm"] += 100
        future_path = tmp_path / "future_modified_processed.csv"
        future_modified.to_csv(future_path, index=False)
        monkeypatch.setattr(forecast_service, "DATA_PATH", future_path)
        forecast_service.load_raw_data.cache_clear()
        future_independent = forecast_service.generate_forecast(
            "City-0", "ward-0", 2, requested_date,
        )
        assert future_independent["rawPredictedCases"] == expected_raw
        assert future_independent["observationDate"] == origin

        earliest_valid = date_bounds["areas"]["City-0"]["ward-0"]["min"]
        earliest_forecast = forecast_service.generate_forecast(
            "City-0", "ward-0", 2, earliest_valid,
        )
        assert earliest_forecast["observationDate"] == earliest_valid
        latest_valid = date_bounds["areas"]["City-0"]["ward-0"]["max"]
        latest_forecast = forecast_service.generate_forecast(
            "City-0", "ward-0", 2, latest_valid,
        )
        assert latest_forecast["observationDate"] == latest_valid
        assert latest_forecast["actualCases"] is None
        assert latest_forecast["isRetrospectiveEstimate"] is False

        other_ward_forecast = forecast_service.generate_forecast(
            "City-1", "ward-1", 2, requested_date,
        )
        assert other_ward_forecast["city"] == "City-1"
        assert other_ward_forecast["wardId"] == "ward-1"
        assert other_ward_forecast["observationDate"] == origin

        with pytest.raises(forecast_service.ForecastUnavailableError, match="Current forecasting is disabled"):
            forecast_service.generate_forecast("City-0", "ward-0", 2, today=pd.Timestamp("2026-11-26").date())
        with pytest.raises(ValueError, match="insufficient feature history"):
            forecast_service.generate_forecast("City-0", "ward-0", 2, "2022-10-04")
        with pytest.raises(ValueError, match="after the latest ward observation"):
            forecast_service.generate_forecast("City-0", "ward-0", 2, "2025-12-31")
        unsupported_test_origin = (pd.Timestamp("2022-10-03") + pd.Timedelta(weeks=12)).date().isoformat()
        with pytest.raises(ValueError, match="saved held-out rolling-origin test weeks"):
            forecast_service.generate_forecast("City-0", "ward-0", 2, unsupported_test_origin, strict_evaluation=True)
        last_origin = (pd.Timestamp("2022-10-03") + pd.Timedelta(weeks=15)).date().isoformat()
        with pytest.raises(ValueError, match="target week is not in the dataset"):
            forecast_service.generate_forecast("City-0", "ward-0", 2, last_origin, strict_evaluation=True)

        with pytest.raises(LookupError, match="Unknown ward"):
            forecast_service.generate_forecast("City-0", "unknown", 2)
        with pytest.raises(LookupError, match="does not belong"):
            forecast_service.generate_forecast("City-1", "ward-0", 2)
        with pytest.raises(ValueError, match="Unsupported horizon"):
            forecast_service.generate_forecast("City-0", "ward-0", 5)
    finally:
        forecast_service.load_raw_data.cache_clear()
        original_load_model_bundle.cache_clear()
        forecast_service._load_replay_predictions.cache_clear()

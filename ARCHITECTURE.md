# AEDISIGHT ML Architecture and Validation

## Scope

This document describes the Bangladesh ward-level forecasting pipeline and its authenticated FastAPI dashboard integration. The dashboard loads locally trained run artifacts; forecasts use historical inputs and do not establish a clinically validated or operational warning system.

## Data flow

```text
raw ward CSV
    │
    ▼
src/clean_data.py ── validate source values, IDs, dates, duplicate keys,
    │                  and seven-day continuity within each ward
    │                  drop/report missing source and history rows
    ▼
processed CSV ────── atomic temporary write and row/schema verification
    │
    ▼
src/validation.py ─ validate processed source columns and values again
src/feature_engineering.py ─ validate cleaner feature schema; grouped targets
    │
    ▼
target-date chronological split (train / validation / final test)
    │
    ├── fit CatBoost, GradientBoosting, HistGradientBoosting on train
    ├── compare validation MAE including persistence, select one per horizon
    ├── refit selected learned model on train + validation when selected
    ├── evaluate once on untouched test and compare persistence baseline
    ▼
new runs/<run>/ directory: models, metadata, validation/test reports
    │
    ▼
src/predict.py ─ load compatible artifact once; enforce feature order
    │
    ▼
POST /api/predict ─ hospital session + CSRF → ward forecast response
    │
    ▼
Jinja dashboard and JavaScript render actual output or explicit errors
```

The cleaner's default input and output paths are based on the repository location, not the shell's current directory. The raw CSV is never an output target. The processed CSV write uses a temporary file in the destination directory; the cleaner verifies its header, complete rows, and row count before atomically replacing the destination. Training refuses to reuse an existing run output directory. Each model file is written to a temporary filename and reloaded before the full staging directory is atomically published.

## Validation and missing data

`src/clean_data.py` requires the documented source columns and checks ISO-compatible dates, finite numeric values, non-negative dengue/rainfall/population values, positive population density, `[0, 1]` risk/sanitation scores, `[0, 100]` humidity, plausible temperatures, ordered minimum/average/maximum temperatures, and integer dengue counts. It rejects duplicate rows and duplicate `(date, ward_id)` keys, whitespace/case-inconsistent labels, wards associated with multiple city labels, and any within-ward date interval other than seven days.

Missing source rows are explicitly counted and dropped before feature generation. Since cleaner features need prior history, initial rows for each ward can also be incomplete; they are counted and dropped before processed export. The training validator rejects missing values and enforces the same source-value rules on the processed dataset. It does not require a fixed row count, fixed ward count, or fixed number of cities.

## Feature contract and time availability

`src/feature_engineering.py` contains the ordered feature schema consumed by each estimator. It does not repeat the cleaner's computations. This prevents the training and comparison path from silently constructing features differently from `src/clean_data.py`.

`src/compare_splits.py` is retained as a compatibility command that delegates directly to `src.train`; it no longer contains its own feature code or uses test metrics to rank candidates.

The contract contains:

- Categorical `city` and `ward_id`.
- Static/context values: population density, water-logging risk, and sanitation score.
- Current-row observations: average/minimum/maximum temperature, rainfall, relative humidity, and dengue value.
- Per-ward lags for rainfall, dengue, average temperature, and humidity at one through four weekly observations.
- Rainfall sums for the preceding two and four observations.
- Average temperature and relative humidity means across the four observations preceding the current row.
- Calendar week and month.

The cleaner sorts each ward chronologically before using past slices and never mixes ward groups. The current-row dengue value is also the persistence baseline. This pipeline assumes the current observation is available after that week's reporting is complete. It cannot verify reporting delay, retrospective correction, or whether a feature was available on a real forecast issuance date. Dengue case semantics remain unknown: values are preserved without interpreting them as incidence or cumulative totals.

## Forecast targets

For each ward, a grouped negative shift constructs `target_2w`, `target_3w`, and `target_4w`; corresponding `target_date_*` fields use the same shift on dates. Before construction, rows are sorted by ward and date and the training validator enforces exact seven-day continuity. Thus the shift counts weekly observations, and no target can cross a ward boundary. Missing future targets at the end of each ward's series are excluded explicitly and their counts are stored per horizon in metadata.

## Chronological partitions and selection

The partition is assigned from the future target date:

| Partition | Inclusive target-date range |
| --- | --- |
| Train | Through 2023-12-31 |
| Validation | 2024-01-01 through 2024-12-31 |
| Final test | 2025-01-01 through 2025-12-31 |

All periods are configurable in `src/train.py` or with CLI date flags. Every requested horizon must have a non-empty train, validation, and test partition. Random splitting is not used.

Each learned candidate is fitted to training examples only. Candidate comparison includes the persistence forecast, and selection uses validation MAE; persistence wins ties. The test partition is not passed to candidate fitting, candidate prediction, parameter selection, or feature selection. A selected learned model is refitted using the combined train and validation examples. If persistence wins, the artifact records that method without an estimator. The held-out 2025 examples are then evaluated for final reporting only.

Chronological partitions are better aligned with forecasting future dates than random row splits and make the evaluation periods explicit. Limitations include a single validation year and test year, changes in data collection or outbreak patterns between years, adjacent forecasts with overlapping target periods, and unknown source publication delays. Performance must not be generalized beyond this backtest without further temporal/geographic validation.

## Candidate models and preprocessing

The three repository model families are retained:

- `CatBoostRegressor` with explicit categorical feature names, fixed seed, and configured iteration count.
- `GradientBoostingRegressor` behind a scikit-learn pipeline.
- `HistGradientBoostingRegressor` behind the same type of pipeline.
- Persistence baseline carrying forecast-origin dengue cases forward unchanged.

Scikit-learn candidates one-hot encode `city` and `ward_id` with `handle_unknown="ignore"`, then pass numeric columns through. CatBoost categorical values are normalized to strings for fitting and prediction. All families use the common fixed-order feature list. `--iterations` controls their tree/iteration budget, and `--seed` controls applicable randomization.

Negative model predictions are clipped to zero because the reported quantity is a non-negative dengue value. Observed actual values are not transformed. Non-finite predictions and invalid targets fail the run rather than being silently excluded or changed.

## Evaluation and persistence baseline

Validation comparisons report candidate MAE, RMSE, R², baseline validation MAE, and the selected flag. Final test metrics report the selected model's MAE, RMSE, and R² alongside the persistence baseline's corresponding metrics and relative MAE improvement:

`(baseline MAE - model MAE) / baseline MAE`

Positive improvement means a lower MAE than persistence; negative means worse. A zero baseline MAE makes relative improvement undefined. R² is NaN for a constant actual target. Test outcomes are not used to choose the selected model. A higher R² alone does not establish an improvement over the baseline.

The baseline predicts the observed `dengue_cases` value at the forecast origin for every requested future horizon. The metrics are calculated separately for each horizon and partition.

## Reproducibility and run outputs

The cleaner resolves its default paths from the repository and can be called from any current directory. Run training from the repository root so `src.train` imports the project package:

```bash
python /path/to/repository/src/clean_data.py
cd /path/to/repository
python -m src.train
```

For a reproducible explicit command, including all configuration:

```bash
python -m src.train \
  --input data/processed/dengue_dataset.csv \
  --output runs/aedisight_reproducible_run \
  --horizons 2 3 4 \
  --train-through 2023-12-31 \
  --validation-start 2024-01-01 --validation-end 2024-12-31 \
  --test-start 2025-01-01 --test-end 2025-12-31 \
  --seed 42 --iterations 200
```

The output path must be new. A run directory contains one selected model artifact per requested horizon, `metadata.json`, validation model comparison, final test metrics, test predictions, and supported-model feature importances. Each prediction row records ward, city, forecast origin date, target date, actual target, non-negative ML prediction, persistence prediction, and horizon.

Inference reads the same processed feature table and shared `FEATURE_COLUMNS` contract used by training. It discovers a compatible run from `runs/` (or uses `AEDISIGHT_MODEL_RUN`), verifies each artifact's horizon and ordered feature metadata, then caches the model per horizon. City and ward selectors are generated from records in the validated processed dataset. The API validates the selected city/ward, horizon, optional origin date, session and CSRF token. The API returns an explicit error when data or model artifacts are unavailable; it does not substitute a heuristic prediction.

Risk categories use only case history through the selected forecast origin. Active-week percentile bands and sparse-history fallback values are display heuristics, not clinical thresholds or probabilities. Driver text is a summary of observed current values against ward history, not SHAP attribution.

Metadata includes dataset/date/ward summary, horizon and target, dropped no-target row count, feature names and order, categorical fields, chosen family and parameters, split boundaries and partition sizes, seed, library versions, selected validation metrics, final test metrics, baseline definition, feature availability assumption, and successful artifact reload checks. Artifact bundles include the estimator and feature schema. Reload using `joblib.load(path)` and pass a DataFrame with the recorded feature names in the recorded order.

## Tests and verification boundaries

Synthetic tests in `tests/test_ml_pipeline.py` exercise source validation failures, cleaner lag/rolling calculations, chronological ordering, ward isolation, no future-derived history, targets/dates, target-date splits, metrics, unseen categorical values, fixed-seed behavior, a complete 2/3/4-week smoke run, and artifact reloads. Run them with:

```bash
python -m pytest -q tests/test_ml_pipeline.py
```

The complete repository suite is:

```bash
python -m pytest -q
```

Passing synthetic tests verifies code behavior on their fixtures; it is not a model-performance claim and does not replace real-data evaluation. The model training process is offline and does not alter FastAPI routes, authentication, database configuration, login/registration, or dashboard inference behavior.

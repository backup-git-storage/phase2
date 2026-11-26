# AEDISIGHT Change Report

## Executive summary

This continuation completes the existing FastAPI AEDISIGHT application with a leakage-aware weekly forecasting pipeline, horizon-specific trained models, held-out historical replay, real evaluation reports, and expanded regression coverage. Training completed successfully for 2-, 3-, and 4-week horizons. The application uses a historical competition dataset ending 2025-12-29; it does not provide live 2026 surveillance.

## Repository comparison

- **Original baseline:** `Abdullah-Tawsif/Dengue-Outbreak-Prediction-ML`, refreshed `main` at `fb6c4e6bf5cd4d482c4bb3236c70845e64c16d27` (2026-11-22). The existing FastAPI application, hospital-user authentication, dashboard, and initial CatBoost workflow were retained and extended.
- **Completed local implementation:** the existing AEDISIGHT worktree, with the changes listed below and a completed run at `runs/aedisight_20261125T232745_977823Z/`.
- **Destination before publication:** `backup-git-storage/phase2`, refreshed `main` at `188ff9edb95e3aa94b1a9bef4f7555cd04d146ab`. It contained only a 17-byte root `README.md`. The file is preserved unchanged; no project README or architecture document is part of the application payload.

Files changed relative to the refreshed original application include `main.py`; `src/clean_data.py`, `src/compare_splits.py`, `src/feature_engineering.py`, `src/predict.py`, `src/preprocessing.py`, `src/risk_engine.py`, `src/train.py`, and `src/validation.py`; `static/css/user/user_dashboard.css`; `static/js/user/user.js`; `templates/home/index.html`; `templates/user/user_dashboard.html`; and `tests/test_app.py`. `tests/test_ml_pipeline.py` and the completed run artifacts are new. `src/compare_splits.py` now delegates to the rolling-origin validation implementation instead of the original target-date split implementation. The existing authentication modules and other shared pages remain part of the application; they are not described as new functionality.

The original-only `MODEL_INTEGRATION.md` and `environment.yml` are not in the publication payload. Legacy root `models/catboost_model_*.joblib` files and root `results/` outputs are incompatible with the completed artifact contract and are excluded. Local `dengue.db`, `.env`, `CODEX_HANDOFF.md`, root `README.md`, and root `ARCHITECTURE.md` are excluded. The root README and architecture file had pre-existing local worktree edits; this continuation did not edit them and does not publish them.

## Final architecture and functionality

- FastAPI remains the application server, with the existing hospital-user login, session, password-hash, and protected-route structure. Prediction routes retain authentication and CSRF protections and return explicit errors for unavailable models, stale current data, invalid wards, and unsupported requests.
- `src/train.py`, `src/validation.py`, `src/feature_engineering.py`, and `src/preprocessing.py` implement weekly target construction, origin-aware features, expanding-window evaluation, baseline comparison, per-horizon model selection, and artifact writing.
- `src/predict.py` loads only compatible versioned artifacts and uses saved held-out test predictions for historical replay. `src/risk_engine.py` supplies transparent ward-relative ordinal bands and observed-input context.
- The existing Jinja dashboard now shows model evaluation data from saved reports, forecast loading/error states, supported replay origins, and a prominent stale-data message. Historical replay and current forecasting have distinct labels and controls.

## Target definition and data

The competition data contains 31,320 raw weekly records across 60 wards and five cities; the processed input used by the run contains 31,080 records through 2025-12-29. The separate 576-record monthly `dataset.xlsx` is not used.

The project-owner working interpretation is that `dengue_cases` is the number of new cases recorded for a ward during its associated seven-day reporting period. This is an assumption, not an independently verified data-dictionary definition. The observation’s week-start/end convention is unspecified. A horizon target is one future observation from the same ward: `dengue_cases.shift(-weeks)` after chronological ward grouping. Thus 2w, 3w, and 4w each forecast one weekly count at t+2, t+3, or t+4; they do not forecast daily or cumulative cases and do not sum multiple future observations.

## ML validation, models, and measured results

Training examples at each forecast origin are limited to labels available by that origin. Features use same-ward history and weather observations available on or before the information date. The primary evaluation assumes zero publication delay; a one-week delay is also reported as a hypothetical sensitivity, not as a known reporting rule. The evaluation uses 14 expanding-window validation origins and 13 held-out test origins at four-week steps. For each horizon and origin, candidate models and both baselines were scored on the same 60-ward sample (840 rows per model/horizon in validation; 780 in test). Model selection uses pooled validation MAE only; held-out test results are reported after selection.

| Horizon | Selected on validation | Validation MAE / RMSE / R² | Held-out test MAE / RMSE / R² |
| --- | --- | ---: | ---: |
| 2 weeks | `hist_gradient_boosting` | 0.930 / 1.855 / 0.804 | 1.283 / 2.744 / 0.440 |
| 3 weeks | `hist_gradient_boosting` | 0.868 / 1.623 / 0.799 | 1.040 / 1.842 / 0.767 |
| 4 weeks | `gradient_boosting` | 1.006 / 1.808 / 0.641 | 1.315 / 2.629 / 0.489 |

Validation MAE for persistence and seasonal-naive, respectively, was 1.624/2.418 at 2w, 1.408/1.670 at 3w, and 1.625/1.654 at 4w. The selected models are not uniformly best on the held-out test set: at 2w seasonal-naive test MAE was 1.179 versus 1.283 for the validation-selected model; at 3w CatBoost test MAE was 1.087 versus 1.040 for the selected model; at 4w gradient boosting had the lowest test MAE at 1.315. Selection was not changed using test results. These are historical estimates, not a guarantee of future performance.

The full run includes schema-versioned metadata, model feature contracts, training cutoffs, selected model and validation/test metrics, predictions, risk metrics, stratified metrics, and reporting-delay sensitivity at `runs/aedisight_20261125T232745_977823Z/`. All three artifacts passed the training-time save/reload prediction check. The completed run used Python 3.12.3, pandas 3.1.1, scikit-learn 1.9.1, CatBoost 1.2.10, and 50 iterations for the candidate estimators.

## Risk bands, replay, and data freshness

Risk output is one of LOW, MEDIUM, HIGH, or CRITICAL. With at least eight nonzero historical weeks, thresholds use ward-specific 50th, 75th, and 90th percentiles among active weeks, with monotonic minimum spacing; sparse histories use explicit prototype fallback cutoffs. Historical thresholds are calculated from observations available at the selected origin. Risk categories are experimental ordinal bands, not calibrated probabilities or clinically validated alerts. Rain, temperature, humidity, and recent cases are described as observed context, not causal explanations or model attribution.

Historical replay is restricted to saved, held-out rolling-origin test samples. It verifies the origin-specific information date, label cutoff, target date, and actual value; it does not apply a final model trained through 2025 to earlier origins. Validation-period replay is unavailable because that period is used for model selection. Current forecasting is blocked when data are older than the configured 14-day freshness window. As of 2026-11-26, the dataset is 332 days old, so the product must be presented as historical replay, not live surveillance.

## Backend, frontend, and security changes

- `main.py` connects the dashboard to available compatible horizons, replay-origin bounds, and actual saved model-performance data; it maps model/freshness/validation errors to explicit HTTP responses.
- `src/predict.py` validates ward/city inputs, horizon and artifact compatibility, staleness, saved replay origins, and training cutoffs. Incompatible legacy artifacts do not silently generate predictions.
- `templates/user/user_dashboard.html`, `static/js/user/user.js`, and `static/css/user/user_dashboard.css` add the freshness banner, replay-only language, distinct current/replay controls, supported-origin handling, loading states, and readable API errors.
- Existing authentication and hospital-user data remain integrated. Regression tests exercise protected dashboard access, CSRF rejection, and prediction status handling. No public admin registration or hardcoded administrator credentials were added.

## Verification

- Training command: `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -u -m src.train --input data/processed/dengue_dataset.csv --horizons 2 3 4 --origin-step-weeks 4 --iterations 50` — completed with exit code 0; all three horizons wrote models, metadata, test predictions, and aggregate reports.
- Artifact/data audit — passed: all expected horizon artifacts exist, metadata reports save/reload checks passed, validation and test samples are paired across five model/baseline families, information dates and training-label cutoffs do not exceed origins, and target dates equal origin plus exactly 7, 14, or 21 days.
- Full suite: `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider -q` — **42 passed**, 2 dependency deprecation warnings, 21.89 seconds.
- `git diff --check` — passed.
- Browser and JavaScript-runtime checks were unavailable: no Node.js, other JavaScript runtime, or browser executable is installed. No browser/E2E result is claimed.

## Limitations and known issues

- Data end date is 2025-12-29; current operational forecasts are unavailable until recent weekly data is provided.
- The meaning of `dengue_cases` and reporting-week convention remain project assumptions/unknowns as described above.
- The rolling-origin test covers 13 sampled origins in 2025; the separate seasonal baselines sometimes score better on test MAE than a validation-selected ML model.
- Risk thresholds and tier metrics have no clinical validation; forecasts do not provide calibrated probabilities or causal explanations.
- No browser-based test was possible in the available environment.

## Publication payload

The exact 58-file payload prepared in the temporary clone is:

```text
.env.example
.gitignore
WHAT_CHANGED.md
data/processed/dengue_dataset.csv
main.py
requirements.txt
runs/aedisight_20261125T232745_977823Z/metadata.json
runs/aedisight_20261125T232745_977823Z/models/model_2w.joblib
runs/aedisight_20261125T232745_977823Z/models/model_3w.joblib
runs/aedisight_20261125T232745_977823Z/models/model_4w.joblib
runs/aedisight_20261125T232745_977823Z/results/feature_importance_2w.csv
runs/aedisight_20261125T232745_977823Z/results/feature_importance_3w.csv
runs/aedisight_20261125T232745_977823Z/results/feature_importance_4w.csv
runs/aedisight_20261125T232745_977823Z/results/final_test_metrics.csv
runs/aedisight_20261125T232745_977823Z/results/reporting_delay_sensitivity.csv
runs/aedisight_20261125T232745_977823Z/results/risk_metrics.csv
runs/aedisight_20261125T232745_977823Z/results/stratified_metrics.csv
runs/aedisight_20261125T232745_977823Z/results/test_predictions.csv
runs/aedisight_20261125T232745_977823Z/results/test_predictions_2w.csv
runs/aedisight_20261125T232745_977823Z/results/test_predictions_3w.csv
runs/aedisight_20261125T232745_977823Z/results/test_predictions_4w.csv
runs/aedisight_20261125T232745_977823Z/results/validation_model_comparison.csv
runs/aedisight_20261125T232745_977823Z/results/validation_predictions.csv
src/__init__.py
src/backend/__init__.py
src/backend/admin_cli.py
src/backend/auth.py
src/backend/database.py
src/backend/dependencies.py
src/backend/models.py
src/backend/security.py
src/clean_data.py
src/compare_splits.py
src/feature_engineering.py
src/predict.py
src/preprocessing.py
src/risk_engine.py
src/train.py
src/validation.py
static/css/admin/admin_dashboard.css
static/css/login/login.css
static/css/register/register.css
static/css/user/user_dashboard.css
static/index.css
static/js/admin/admin.js
static/js/index.js
static/js/login/login.js
static/js/register/register.js
static/js/user/user.js
templates/admin/admin_dashboard.html
templates/auth/login/login.html
templates/auth/register/register.html
templates/home/index.html
templates/user/user_dashboard.html
tests/test_app.py
tests/test_auth.py
tests/test_ml_pipeline.py
vercel.json
```

The processed CSV is copied with CRLF line endings normalized to LF in the temporary clone; its 31,080 rows and field values are unchanged. Two files also have trailing blank lines removed in that clone for a clean Git whitespace check. The payload excludes root `README.md`, root `ARCHITECTURE.md`, `CODEX_HANDOFF.md`, `.env`, local databases, raw/monthly/duplicate data, old root models/results, staging/partial runs, caches, logs, and build/virtual-environment files. The destination’s existing root README is preserved byte-for-byte.

### Follow-up: local run instructions

`RUNNING.md` was added after the initial application publication. It documents VS Code integrated-terminal setup and launch commands for Linux Bash, Windows PowerShell, and Windows Command Prompt; local `.env` and secret setup; virtual-environment creation; the local URL and stop command; and the stale-data limitation. The authorized follow-up publication adds `RUNNING.md` and updates this report. The final destination tree therefore contains the initial 58 application files plus `RUNNING.md`, along with the byte-for-byte preserved destination README.

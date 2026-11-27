# Independent reference CatBoost pipeline

The copied app selects `PREDICTION_BACKEND=reference_catboost` by default. Set it to `phase2` in `.env` to select the preserved Phase2 inference service. The reference backend requires a trained run under `runs/reference_catboost_*`; it errors clearly when a selected model is unavailable and never falls back silently. `REFERENCE_CATBOOST_RUN` can point to a specific run directory.

## Reproduced reference behavior

At the inspected reference commit, `src/feature_engineering.py` and `src/predict.py` are zero-byte files. The executable feature construction and estimator parameters are in `src/train.py`; those concrete transformations and CatBoost inference semantics (`fit` with categorical feature names, then raw `predict` on the same ordered columns) are what this copy reproduces. No behavior was inferred from the empty helper files.

`src/reference_catboost_features.py` rebuilds features from the raw columns present in the Phase2 processed CSV. It preserves the reference feature ordering and CatBoost configuration: RMSE, learning rate 0.05, depth 7, L2 leaf regularization 3, random seed 42, no file writes, and one CPU thread. The requested 50 iterations replace the reference's 700. Forecast targets are same-ward shifts of 2, 3, or 4 weekly rows. The chronological split follows the reference: target years through 2024 train the model, target year 2025 is held out for test metrics. The model evaluated on the test rows is the same model saved for inference; there is no final refit.

All lag and rolling calculations group by ward. Rolling weather and dengue summaries use only the preceding 2, 4, or 8 observations; current weather is available at the forecast origin and current dengue is used only to form trend features. The dataset contains all reference raw inputs, so no reference source features were omitted. There is no extra population, vector, or climate data beyond Phase2's existing columns.

## Evaluation distinction

This single chronological split differs from Phase2's expanding rolling-origin validation/testing. Its metrics are reported separately and must not be compared as if they used identical origins or aggregation. R² is a regression goodness-of-fit statistic, not classification accuracy.

## Historical dates

Bounds are computed from usable engineered rows for each city and ward. The browser date input has inclusive min/max bounds and no weekly step. A selected calendar day resolves to that ward's latest usable observation on or before the day, and the selected day stays visible. The forecast target follows Phase2's existing convention: resolved weekly observation date plus the selected number of weeks. Flexible history is explicitly marked retrospective if the model's target-label training cutoff is later than the origin. Strict evaluation continues to use saved Phase2 held-out predictions.

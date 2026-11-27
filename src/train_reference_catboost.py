"""Train exactly one reference-style CatBoost model for each 2/3/4-week horizon."""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.reference_catboost_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, engineer_features, usable_feature_mask

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/dengue_dataset.csv'
PARAMS = dict(loss_function='RMSE', iterations=50, learning_rate=0.05, depth=7,
              l2_leaf_reg=3, random_seed=42, verbose=False,
              allow_writing_files=False, thread_count=1)
HORIZONS = (2, 3, 4)


def run_training(input_path: Path = INPUT, output_parent: Path | None = None) -> Path:
    started = time.monotonic()
    raw = pd.read_csv(input_path)
    features = engineer_features(raw)
    for h in HORIZONS:
        features[f'target_{h}w'] = features.groupby('ward_id', sort=False, observed=True).dengue_cases.shift(-h)
        features[f'target_date_{h}w'] = features.groupby('ward_id', sort=False, observed=True).date.shift(-h)
    usable = usable_feature_mask(features)
    root = output_parent or ROOT / 'runs'
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / ('reference_catboost_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    run_dir.mkdir(exist_ok=False)
    (run_dir / 'models').mkdir(); (run_dir / 'results').mkdir()
    metadata = {
      'schema_version': 1, 'backend': 'reference_catboost', 'run_created_utc': datetime.now(timezone.utc).isoformat(),
      'dataset': str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else input_path.name,
      'dataset_rows': int(len(raw)), 'ward_count': int(raw.ward_id.nunique()),
      'dataset_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
      'source_columns': list(raw.columns), 'fit_count': 3,
      'input_date_min': str(pd.to_datetime(raw.date).min().date()), 'input_date_max': str(pd.to_datetime(raw.date).max().date()),
      'feature_names': FEATURE_COLUMNS, 'categorical_features': CATEGORICAL_FEATURES,
      'parameters': PARAMS, 'target_definition': 'dengue_cases at the same ward after h weekly observations',
      'split': 'Chronological target-date split matching reference: train target year <= 2024; test target year == 2025.',
      'evaluation_method': 'single_chronological_train_test_split', 'horizons': {},
      'differences_from_reference': [
        'At the inspected reference commit, src/feature_engineering.py and src/predict.py are empty; the concrete feature logic and inference contract were taken from src/train.py.',
        'Phase2 processed CSV has cleaner-generated features, but this pipeline rebuilds reference features from available raw columns to guarantee matching train/inference transformations.',
        'Reference target years and CatBoost parameters are preserved; iterations are set to the requested 50.',
        'The Phase2 source provides weekly new-case counts and ward/city/weather/context columns; no additional source columns are inferred.'
      ]
    }
    for h in HORIZONS:
        target_date = pd.to_datetime(features[f'target_date_{h}w'])
        valid = usable & features[f'target_{h}w'].notna() & target_date.notna()
        train_mask = valid & (target_date.dt.year <= 2024)
        test_mask = valid & (target_date.dt.year == 2025)
        train, test = features.loc[train_mask].copy(), features.loc[test_mask].copy()
        if train.empty or test.empty:
            raise ValueError(f'{h}w chronological split is empty (train={len(train)}, test={len(test)})')
        X_train = train.loc[:, FEATURE_COLUMNS].copy(); X_test = test.loc[:, FEATURE_COLUMNS].copy()
        for col in CATEGORICAL_FEATURES:
            X_train[col] = X_train[col].astype(str); X_test[col] = X_test[col].astype(str)
        model = CatBoostRegressor(**PARAMS)
        model.fit(X_train, train[f'target_{h}w'].astype(float), cat_features=list(CATEGORICAL_FEATURES))
        prediction = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        if len(prediction) != len(test) or not np.isfinite(prediction).all():
            raise RuntimeError(f'{h}w model generated invalid test predictions')
        if model.tree_count_ != 50:
            raise RuntimeError(f'{h}w CatBoost tree count is {model.tree_count_}, expected 50')
        model_name = f'catboost_model_{h}w.cbm'
        model.save_model(run_dir / 'models' / model_name)
        metrics = {'mae': float(mean_absolute_error(test[f'target_{h}w'], prediction)),
                   'rmse': float(np.sqrt(mean_squared_error(test[f'target_{h}w'], prediction))),
                   'r2': float(r2_score(test[f'target_{h}w'], prediction))}
        out = test[['city','ward_id','date',f'target_date_{h}w',f'target_{h}w']].copy()
        out = out.rename(columns={f'target_date_{h}w':'target_date',f'target_{h}w':'actual'})
        out['horizon_weeks'] = h; out['prediction'] = prediction
        out.to_csv(run_dir / 'results' / f'test_predictions_{h}w.csv', index=False)
        train_end = pd.to_datetime(train.date)
        test_dates = pd.to_datetime(test.date)
        target_dates = pd.to_datetime(test.target_date if 'target_date' in test else test[f'target_date_{h}w'])
        metadata['horizons'][str(h)] = {
          'horizon_weeks': h, 'model_artifact': f'models/{model_name}', 'tree_count': int(model.tree_count_),
          'feature_count': len(FEATURE_COLUMNS), 'feature_names': FEATURE_COLUMNS,
          'training_rows': len(train), 'test_rows': len(test),
          'training_origin_start': str(train.date.min().date()), 'training_origin_end': str(train.date.max().date()),
          'training_target_start': str(pd.to_datetime(train[f'target_date_{h}w']).min().date()),
          'training_target_end': str(pd.to_datetime(train[f'target_date_{h}w']).max().date()),
          'test_origin_start': str(test_dates.min().date()), 'test_origin_end': str(test_dates.max().date()),
          'test_target_start': str(pd.to_datetime(test[f'target_date_{h}w']).min().date()),
          'test_target_end': str(pd.to_datetime(test[f'target_date_{h}w']).max().date()),
          'training_cutoff': str(pd.to_datetime(train[f'target_date_{h}w']).max().date()),
          'metrics': metrics
        }
        print(f'{h}w: train={len(train)} test={len(test)} MAE={metrics["mae"]:.4f} RMSE={metrics["rmse"]:.4f} R2={metrics["r2"]:.4f}', flush=True)
    metadata['training_duration_seconds'] = round(time.monotonic() - started, 3)
    (run_dir / 'feature_names.json').write_text(json.dumps({'feature_names': FEATURE_COLUMNS,
        'categorical_features': CATEGORICAL_FEATURES}, indent=2) + '\n', encoding='utf-8')
    (run_dir / 'catboost_parameters.json').write_text(json.dumps(PARAMS, indent=2) + '\n', encoding='utf-8')
    metric_rows=[]
    for h,info in metadata['horizons'].items():
        metric_rows.append({'horizon_weeks':int(h), **info['metrics'], 'training_rows':info['training_rows'],
            'test_rows':info['test_rows'], 'training_origin_start':info['training_origin_start'],
            'training_origin_end':info['training_origin_end'], 'training_target_start':info['training_target_start'],
            'training_target_end':info['training_target_end'], 'test_origin_start':info['test_origin_start'],
            'test_origin_end':info['test_origin_end'], 'test_target_start':info['test_target_start'],
            'test_target_end':info['test_target_end'], 'training_cutoff':info['training_cutoff'],
            'iterations':info['tree_count']})
    pd.DataFrame(metric_rows).to_csv(run_dir / 'results' / 'evaluation_metrics.csv', index=False)
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'RUN_DIR={run_dir}', flush=True)
    print(f'TRAINING_DURATION_SECONDS={metadata["training_duration_seconds"]}', flush=True)
    return run_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output-parent', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    run_training(args.input, args.output_parent)

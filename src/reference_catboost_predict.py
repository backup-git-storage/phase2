"""Inference service for the independent reference-style CatBoost run."""
from __future__ import annotations
import json, os
from datetime import date
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.reference_catboost_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, engineer_features, usable_feature_mask
from src.feature_engineering import SUPPORTED_HORIZONS
from src.predict import (DATA_PATH, ForecastUnavailableError, ModelUnavailableError,
                         MAX_CURRENT_DATA_AGE_DAYS, data_freshness as _legacy_freshness)
from src.risk_engine import classify_risk, describe_drivers
from src.train import display_case_count

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / 'runs'


def _run_dir() -> Path:
    configured = os.getenv('REFERENCE_CATBOOST_RUN', '').strip()
    if configured:
        path = Path(configured).expanduser()
        path = path if path.is_absolute() else (ROOT / path).resolve()
        if not path.is_dir():
            raise ModelUnavailableError(f'Configured reference CatBoost run does not exist: {path}')
        return path
    runs = sorted((p for p in RUNS_DIR.glob('reference_catboost_*') if p.is_dir()), key=lambda p: p.name)
    if not runs:
        raise ModelUnavailableError('No trained reference CatBoost experiment is configured')
    return runs[-1]


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise ForecastUnavailableError('Processed ward data is unavailable')
    try:
        frame = engineer_features(pd.read_csv(DATA_PATH))
        mask = usable_feature_mask(frame)
        frame = frame.loc[mask].copy().sort_values(['ward_id','date'], kind='stable').reset_index(drop=True)
        if frame.empty:
            raise ValueError('No complete reference feature rows')
        return frame
    except Exception as exc:
        raise ForecastUnavailableError('Processed data cannot produce usable reference CatBoost features') from exc


@lru_cache(maxsize=3)
def load_model(horizon: int) -> tuple[CatBoostRegressor, dict, Path]:
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f'Unsupported horizon; choose one of {SUPPORTED_HORIZONS} weeks')
    run = _run_dir()
    try:
        metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
        info = metadata['horizons'][str(horizon)]
        if metadata.get('backend') != 'reference_catboost' or info.get('horizon_weeks') != horizon:
            raise ValueError('horizon/backend metadata mismatch')
        if info.get('feature_names') != FEATURE_COLUMNS or metadata.get('feature_names') != FEATURE_COLUMNS:
            raise ValueError('ordered feature list mismatch')
        artifact = (run / info['model_artifact']).resolve()
        if run.resolve() not in artifact.parents or not artifact.is_file():
            raise ValueError('model artifact missing or outside experiment directory')
        model = CatBoostRegressor()
        model.load_model(str(artifact))
        if model.tree_count_ != 50 or list(model.feature_names_) != list(FEATURE_COLUMNS):
            raise ValueError('model tree count or ordered feature names mismatch')
        return model, info, run
    except Exception as exc:
        raise ModelUnavailableError(f'Reference CatBoost model for {horizon} weeks is unavailable or incompatible') from exc


def get_available_horizons() -> list[int]:
    result=[]
    for h in SUPPORTED_HORIZONS:
        try: load_model(h); result.append(h)
        except (ModelUnavailableError, ValueError, OSError): pass
    return result


def get_supported_areas() -> dict[str,list[str]]:
    data=load_data()
    return {str(city): sorted(group.ward_id.astype(str).unique().tolist()) for city,group in data.groupby('city',sort=True,observed=True)}


def get_supported_date_bounds() -> dict:
    data=load_data(); areas={}
    for (city,ward), group in data.groupby(['city','ward_id'],sort=True,observed=True):
        dates=group.date.sort_values()
        if dates.empty: continue
        areas.setdefault(str(city),{})[str(ward)]={'min':dates.iloc[0].date().isoformat(),'max':dates.iloc[-1].date().isoformat()}
    if not areas: raise ForecastUnavailableError('No ward has usable historical reference features')
    all_bounds=[b for wards in areas.values() for b in wards.values()]
    return {'min':min(x['min'] for x in all_bounds),'max':max(x['max'] for x in all_bounds),'areas':areas}


def get_data_through() -> str:
    return load_data().date.max().date().isoformat()


def data_freshness(today: date | None = None) -> dict:
    data_through=load_data().date.max().date(); today=today or date.today()
    age=max(0,(today-data_through).days); stale=age>MAX_CURRENT_DATA_AGE_DAYS
    return {'dataThrough':data_through.isoformat(),'ageDays':age,'status':'stale' if stale else 'current',
            'currentForecastAvailable':not stale,'maxAgeDays':MAX_CURRENT_DATA_AGE_DAYS,
            'message':(f'Data ends {data_through.isoformat()} ({age} days old). Current forecasting is disabled; select a historical date.' if stale else 'The latest weekly data is within the configured 14-day freshness window.')}


def resolve_feature_row(ward: pd.DataFrame, requested_date: str | date | pd.Timestamp) -> tuple[pd.Timestamp, pd.Series]:
    """Resolve a calendar date to the latest usable same-ward row on or before it."""
    try:
        requested = pd.Timestamp(date.fromisoformat(requested_date) if isinstance(requested_date, str) else requested_date)
    except (TypeError, ValueError) as exc:
        raise ValueError('Forecast date must be a valid YYYY-MM-DD calendar date') from exc
    if ward.empty:
        raise ValueError('This ward has no usable historical feature rows')
    first, last = pd.Timestamp(ward.date.min()), pd.Timestamp(ward.date.max())
    if requested < first:
        raise ValueError(f'This ward has no usable feature row before {first.date().isoformat()}')
    if requested > last:
        raise ValueError(f'Forecast date is after the latest ward observation ({last.date().isoformat()})')
    candidates = ward.loc[ward.date.le(requested)]
    if candidates.empty:
        raise ValueError('No usable feature observation exists on or before the selected date')
    row = candidates.sort_values('date', kind='stable').iloc[-1]
    return pd.Timestamp(row.date), row


def generate_forecast(city: str, ward_id: str, horizon_weeks: int, as_of: str | None = None,
                      today: date | None = None, *, strict_evaluation: bool = False) -> dict:
    if strict_evaluation:
        # Strict evaluation remains bound to saved held-out predictions and the original Phase2 contract.
        from src.predict import generate_forecast as legacy_generate
        result = legacy_generate(city, ward_id, horizon_weeks, as_of, today, strict_evaluation=True)
        result['predictionBackend'] = 'phase2_strict_saved_evaluation'
        return result
    if horizon_weeks not in SUPPORTED_HORIZONS:
        raise ValueError(f'Unsupported horizon; choose one of {SUPPORTED_HORIZONS} weeks')
    data=load_data(); areas=get_supported_areas()
    if city not in areas: raise LookupError('Unknown city')
    if ward_id not in areas[city]:
        if ward_id in {w for values in areas.values() for w in values}: raise LookupError('Ward does not belong to the selected city')
        raise LookupError('Unknown ward')
    ward=data.loc[data.ward_id.astype(str).eq(str(ward_id))].sort_values('date')
    earliest=ward.date.min(); latest=ward.date.max(); historical=as_of is not None
    freshness=data_freshness(today)
    if historical:
        requested=pd.Timestamp(date.fromisoformat(as_of))
        origin, row = resolve_feature_row(ward, requested)
    else:
        if freshness['status']=='stale': raise ForecastUnavailableError(freshness['message'])
        requested=None; origin=pd.Timestamp(latest)
    if not historical:
        row=ward.loc[ward.date.eq(origin)].iloc[-1]
    model,info,run=load_model(horizon_weeks)
    matrix=row.loc[list(FEATURE_COLUMNS)].to_frame().T
    for col in CATEGORICAL_FEATURES: matrix[col]=matrix[col].astype(str)
    numeric=[c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
    matrix[numeric]=matrix[numeric].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(matrix[numeric].to_numpy(dtype=float)).all(): raise ForecastUnavailableError('Resolved feature row contains missing or non-finite values')
    raw=float(np.asarray(model.predict(matrix),dtype=float).reshape(-1)[0])
    if not np.isfinite(raw): raise ModelUnavailableError('CatBoost produced a non-finite prediction')
    target=origin+pd.Timedelta(weeks=horizon_weeks)
    target_rows=ward.loc[ward.date.eq(target)]
    actual=float(target_rows.iloc[0].dengue_cases) if not target_rows.empty else None
    clipped=max(0.0,raw); shown=display_case_count(clipped)
    history=ward.loc[ward.date.le(origin)]
    risk=classify_risk(shown,history.dengue_cases)
    retrospective=str(info['training_cutoff'])>origin.date().isoformat()
    warning='Retrospective historical estimate — not a leakage-safe historical backtest.' if historical and retrospective else None
    warnings=[risk['warning']]
    if freshness['status']=='stale': warnings.insert(0,freshness['message'])
    if warning: warnings.insert(0,warning)
    metrics=info.get('metrics',{})
    perf={'model':'CatBoost','testMae':metrics.get('mae'),'testRmse':metrics.get('rmse'),'testR2':metrics.get('r2'),
          'testRows':info.get('test_rows'),'originStart':info.get('test_origin_start'),'originEnd':info.get('test_origin_end'),
          'targetStart':info.get('test_target_start'),'targetEnd':info.get('test_target_end')}
    return {
      'city':str(city),'wardId':str(ward_id),'horizonWeeks':horizon_weeks,'targetDefinition':'dengue_cases at the same ward after h weekly observations',
      'predictedCases':shown,'rawPredictedCases':raw,'expectedPredictedCases':clipped,'rawPredictionIsClipped':False,
      'riskCategory':risk['level'],'riskMethod':risk['method'],'riskThresholds':risk['thresholds'],'riskWarning':risk['warning'],
      'requestedAsOf':requested.date().isoformat() if requested is not None else None,
      'observationDate':origin.date().isoformat(),'forecastOrigin':origin.date().isoformat(),'targetDate':target.date().isoformat(),'targetReportingWeek':target.date().isoformat(),
      'actualCases':actual,'dataThrough':data.date.max().date().isoformat(),'dataFreshness':freshness['status'],'dataAgeDays':freshness['ageDays'],
      'mode':'flexible_historical_prediction' if historical else 'current_forecast','dataIsHistorical':historical,'warnings':warnings,
      'isHistoricalReplay':False,'isRetrospectiveEstimate':bool(historical and retrospective),'dataFreshnessStatus':freshness['status'],'predictionWarning':warning,
      'drivers':describe_drivers(history,row),'rainfallTiming':{'target_minus_three_weeks':None,'interpretation':'Observed timing context only; not a causal or feature-attribution claim.'},
      'observedFeatures':{n:float(row[n]) for n in ('avg_temp_C','relative_humidity_pct','rainfall_mm','population_density_sqkm','water_logging_risk_index','sanitation_score') if n in row and pd.notna(row[n])},
      'history':[{'date':r.date.date().isoformat(),'cases':float(r.dengue_cases)} for r in history.tail(16).itertuples()],
      'forecast':[{'date':target.date().isoformat(),'cases':shown}],
      'model':'CatBoost','experimentId':run.name,'modelInputFeatureCount':len(FEATURE_COLUMNS),'expectedFeatureCount':len(FEATURE_COLUMNS),'missingFeatureCount':0,
      'modelTrainingDataCutoff':info['training_cutoff'],'modelTrainingDataThrough':info['training_cutoff'],'historicalReplayPartition':None,
      'modelPerformance':perf,'modelHorizonWeeks':horizon_weeks,'predictionBackend':'reference_catboost',
      'modelArtifact':info['model_artifact'],'modelTreeCount':info['tree_count'],'selectedDate':requested.date().isoformat() if requested is not None else None,
    }

"""Non-fitting checks for reference features, date resolution, and backend wiring."""
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.reference_catboost_features import FEATURE_COLUMNS, engineer_features, usable_feature_mask
from src import reference_catboost_predict as ref


def raw_rows(wards=('A',), start='2025-06-02', weeks=20):
    rows=[]
    dates=pd.date_range(start, periods=weeks, freq='7D')
    for wi,ward in enumerate(wards):
        for i,d in enumerate(dates):
            rows.append({'date':d,'city':'City','ward_id':ward,
                'population_density_sqkm':100+wi,'water_logging_risk_index':.5,'sanitation_score':.6,
                'avg_temp_C':20+i,'max_temp_c':25+i,'min_temp_c':15+i,
                'rainfall_mm':float(i+wi*100),'relative_humidity_pct':50+i,
                'dengue_cases':float(i+wi*100)})
    return pd.DataFrame(rows)


def test_reference_features_are_ordered_and_have_supported_columns():
    data=engineer_features(raw_rows())
    assert list(data.loc[:, FEATURE_COLUMNS].columns)==FEATURE_COLUMNS
    assert {'dengue_change_1w','dengue_growth_ratio_2w','dengue_mean_prev_8w',
            'rainfall_sum_prev_8w','avg_temp_mean_prev_8w','humidity_mean_prev_8w',
            'week_sin','month_cos'} <= set(FEATURE_COLUMNS)
    assert usable_feature_mask(data).any()


def test_lags_and_rolls_are_ward_grouped():
    data=engineer_features(raw_rows(('A','B'),weeks=12))
    b=data.loc[data.ward_id.eq('B')].iloc[8]
    assert b.dengue_cases_lag_1w == 107
    assert b.rainfall_sum_prev_2w == 213


def test_features_do_not_use_future_rows():
    base=raw_rows(weeks=15)
    changed=base.copy()
    changed.loc[changed.date > pd.Timestamp('2025-07-28'), 'dengue_cases']=999999
    left=engineer_features(base).set_index('date').loc[pd.Timestamp('2025-07-28'), FEATURE_COLUMNS]
    right=engineer_features(changed).set_index('date').loc[pd.Timestamp('2025-07-28'), FEATURE_COLUMNS]
    pd.testing.assert_series_equal(left,right)


def test_missing_source_columns_fail_clearly():
    with pytest.raises(ValueError, match='missing reference source columns'):
        engineer_features(pd.DataFrame({'date': []}))


def test_daily_date_resolution_is_on_or_before_and_rejects_outside():
    ward=engineer_features(raw_rows(weeks=20))
    origin,row=ref.resolve_feature_row(ward,'2025-08-14')
    assert origin.date().isoformat()=='2025-08-11'
    assert row.date.date().isoformat()=='2025-08-11'
    with pytest.raises(ValueError,match='no usable feature row'):
        ref.resolve_feature_row(ward,'2025-05-01')
    with pytest.raises(ValueError,match='after the latest'):
        ref.resolve_feature_row(ward,'2025-12-31')


def test_date_resolution_is_ward_specific():
    frame=engineer_features(raw_rows(('A','B'),weeks=20))
    frame=frame.loc[~(frame.ward_id.eq('B') & (frame.date < pd.Timestamp('2025-07-28')))]
    a=frame.loc[frame.ward_id.eq('A')]; b=frame.loc[frame.ward_id.eq('B')]
    assert ref.resolve_feature_row(a,'2025-07-30')[0].date().isoformat()=='2025-07-28'
    assert ref.resolve_feature_row(b,'2025-07-30')[0].date().isoformat()=='2025-07-28'
    assert b.date.min()>a.date.min()


class FakeModel:
    tree_count_=50
    feature_count_=len(FEATURE_COLUMNS)
    def predict(self, matrix):
        assert list(matrix.columns)==FEATURE_COLUMNS
        return np.array([4.375])


@pytest.mark.parametrize('horizon',[2,3,4])
def test_all_horizons_return_raw_decimal_and_target_date(monkeypatch,horizon):
    frame=engineer_features(raw_rows(weeks=22))
    meta={'training_cutoff':'2025-09-01','tree_count':50,'feature_count':len(FEATURE_COLUMNS),
          'model_artifact':f'models/{horizon}w.cbm','metrics':{},'test_rows':1}
    monkeypatch.setattr(ref,'load_data',lambda:frame)
    monkeypatch.setattr(ref,'load_model',lambda h:(FakeModel(),meta,Path('/runs/reference_catboost_demo')))
    result=ref.generate_forecast('City','A',horizon,'2025-08-14')
    assert result['requestedAsOf']=='2025-08-14'
    assert result['observationDate']=='2025-08-11'
    assert result['targetDate']==(pd.Timestamp('2025-08-11')+pd.Timedelta(weeks=horizon)).date().isoformat()
    assert result['rawPredictedCases']==4.375
    assert result['predictionBackend']=='reference_catboost'
    assert result['modelHorizonWeeks']==horizon


def test_retrospective_warning_uses_model_training_cutoff(monkeypatch):
    frame=engineer_features(raw_rows(weeks=22))
    meta={'training_cutoff':'2025-09-01','tree_count':50,'feature_count':len(FEATURE_COLUMNS),
          'model_artifact':'models/2w.cbm','metrics':{},'test_rows':1}
    monkeypatch.setattr(ref,'load_data',lambda:frame)
    monkeypatch.setattr(ref,'load_model',lambda h:(FakeModel(),meta,Path('/runs/reference_catboost_demo')))
    response=ref.generate_forecast('City','A',2,'2025-08-14')
    assert response['isRetrospectiveEstimate'] is True
    assert response['predictionWarning']=='Retrospective historical estimate — not a leakage-safe historical backtest.'


def test_ward_bounds_use_first_and_last_usable_rows(monkeypatch):
    frame=engineer_features(raw_rows(('A','B'),weeks=20))
    frame=frame.loc[~(frame.ward_id.eq('B') & frame.date.eq(frame.loc[frame.ward_id.eq('B'),'date'].min()))]
    monkeypatch.setattr(ref,'load_data',lambda:frame)
    bounds=ref.get_supported_date_bounds()
    for ward in ('A','B'):
        dates=frame.loc[frame.ward_id.eq(ward),'date']
        assert bounds['areas']['City'][ward]=={'min':dates.min().date().isoformat(),'max':dates.max().date().isoformat()}


def test_strict_evaluation_stays_on_legacy_saved_predictions(monkeypatch):
    from src import predict as legacy
    monkeypatch.setattr(legacy,'generate_forecast',lambda *args,**kwargs:{'mode':'strict_evaluation_replay'})
    result=ref.generate_forecast('City','A',2,'2025-08-11',strict_evaluation=True)
    assert result['mode']=='strict_evaluation_replay'
    assert result['predictionBackend']=='phase2_strict_saved_evaluation'


def test_frontend_date_input_allows_daily_dates_and_horizon_keeps_selection():
    root=Path(__file__).resolve().parents[1]
    html=(root/'templates/user/user_dashboard.html').read_text()
    js=(root/'static/js/user/user.js').read_text()
    assert 'type="date"' in html and 'step=' not in html
    assert '/static/js/user/user.js?v=10' in html
    assert 'A date between weekly observations resolves to the latest observation on or before your selected date.' in html
    assert 'asOfDate.min = bounds.min;' in js and 'asOfDate.max = bounds.max;' in js
    city_start=js.index('citySelect.addEventListener("change"')
    ward_start=js.index('wardSelect.addEventListener("change"')
    horizon_start=js.index('forecastForm.querySelectorAll', ward_start)
    current_start=js.index('currentMode.addEventListener', horizon_start)
    assert 'updateDateBounds();' in js[city_start:ward_start]
    assert 'updateDateBounds();' in js[ward_start:horizon_start]
    horizon_listener=js[horizon_start:current_start]
    assert 'updateDateBounds' not in horizon_listener and 'asOfDate.value' not in horizon_listener
    assert 'timeZone: "UTC"' in js
    assert 'const FORECAST_TIMEOUT_MS = 30000;' in js
    assert 'invalidatePendingForecast();' in js
    assert 'controller.abort()' in js and 'window.clearTimeout(timeout)' in js
    assert 'citySelect.disabled = isLoading' not in js

"""Dashboard and prediction-handler checks isolated from persistent databases."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SESSION_SECRET"] = "app-tests-only-session-secret-at-least-32-characters"
os.environ["COOKIE_SECURE"] = "false"

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

import main


def _request_with_session(session: dict) -> Request:
    return Request({
        "type": "http", "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/user_dashboard", "raw_path": b"/user_dashboard",
        "query_string": b"", "headers": [], "client": ("127.0.0.1", 1),
        "server": ("test", 80), "root_path": "", "session": session,
    })


def test_dashboard_template_and_prediction_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    user = SimpleNamespace(full_name="Test User", hospital_name="Test Hospital", email="test@example.invalid")
    monkeypatch.setattr(main, "get_supported_areas", lambda: {"Example City": ["WARD_01"]})
    monkeypatch.setattr(
        main,
        "get_supported_date_bounds",
        lambda: {"min": "2025-01-01", "max": "2025-12-29", "areas": {"Example City": {"WARD_01": {"min": "2025-02-01", "max": "2025-12-29"}}}},
    )
    monkeypatch.setattr(
        main,
        "generate_forecast",
        lambda city, ward, horizon, as_of, **_kwargs: {
            "city": city, "wardId": ward, "horizonWeeks": horizon,
            "predictedCases": 2.5, "riskCategory": "MEDIUM",
            "observationDate": "2025-12-01", "targetDate": "2025-12-15",
            "dataThrough": "2025-12-29", "drivers": {"rainfall": "observed context"},
            "history": [{"date": "2025-11-24", "cases": 1.0}, {"date": "2025-12-01", "cases": 1.0}],
            "forecast": [{"date": "2025-12-15", "cases": 2.5}],
            "riskWarning": "Prototype only", "model": "test-estimator",
        },
    )
    session: dict = {}
    request = _request_with_session(session)
    response = asyncio.run(main.user_dashboard(request, user))
    html = response.body.decode("utf-8")
    assert response.status_code == 200
    assert "Example City" in html and "WARD_01" in html
    assert 'value="2025-12-29"' in html
    assert "data-date-bounds" in html
    assert 'name="csrf-token"' in html
    assert '/static/css/user/user_dashboard.css' in html
    assert Path("static/css/user/user_dashboard.css").is_file()
    assert Path("static/js/user/user.js").is_file()

    token = session["_csrf_token"]
    payload = main.ForecastRequest(city="Example City", ward="WARD_01", horizonWeeks=3)
    result = asyncio.run(main.predict_ward(payload, request, token, user))
    assert result["predictedCases"] == 2.5
    assert result["horizonWeeks"] == 3

    with pytest.raises(ValidationError):
        main.ForecastRequest(city="Example City", ward="WARD_01", horizonWeeks=5)
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.predict_ward(payload, request, "", user))
    assert error.value.status_code == 403

    for failure, expected_status in (
        (main.ForecastUnavailableError("stale observations"), 503),
        (main.ModelUnavailableError("missing artifact"), 503),
        (ValueError("invalid replay origin"), 422),
        (LookupError("unknown ward"), 422),
    ):
        def fail_forecast(*_args, error=failure, **_kwargs):
            raise error

        monkeypatch.setattr(main, "generate_forecast", fail_forecast)
        with pytest.raises(HTTPException) as mapped_error:
            asyncio.run(main.predict_ward(payload, request, token, user))
        assert mapped_error.value.status_code == expected_status


def test_registered_prediction_route_enforces_auth_csrf_and_error_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = SimpleNamespace(full_name="Test User", hospital_name="Test Hospital", email="test@example.invalid")
    with TestClient(main.app) as client:
        protected = client.get("/user_dashboard", follow_redirects=False)
        assert protected.status_code == 303
        assert protected.headers["location"] == "/login"

        main.app.dependency_overrides[main.get_current_hospital_user] = lambda: user
        try:
            login_page = client.get("/login")
            token = re.search(r'name="csrf_token"\s+value="([^"]+)', login_page.text).group(1)
            payload = {"city": "Dhaka", "ward": "ward-1", "horizonWeeks": 2}

            csrf_failure = client.post("/api/predict", json=payload, headers={"x-csrf-token": "invalid"})
            assert csrf_failure.status_code == 403

            def stale_forecast(*_args, **_kwargs):
                raise main.ForecastUnavailableError("Current forecasting is disabled")

            monkeypatch.setattr(main, "generate_forecast", stale_forecast)
            stale = client.post("/api/predict", json=payload, headers={"x-csrf-token": token})
            assert stale.status_code == 503

            def invalid_ward(*_args, **_kwargs):
                raise LookupError("Unknown ward")

            monkeypatch.setattr(main, "generate_forecast", invalid_ward)
            invalid = client.post("/api/predict", json=payload, headers={"x-csrf-token": token})
            assert invalid.status_code == 422
        finally:
            main.app.dependency_overrides.pop(main.get_current_hospital_user, None)


def test_auth_pages_are_not_cached_and_include_fresh_csrf_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = SimpleNamespace(full_name="Test User", hospital_name="Test Hospital", email="test@example.invalid", is_admin=False)
    monkeypatch.setitem(main.app.dependency_overrides, main.get_current_hospital_user, lambda: user)
    with TestClient(main.app) as client:
        for path in ("/login", "/register"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store, private"
            assert 'name="csrf_token"' in response.text

        dashboard = client.get("/user_dashboard")
        assert dashboard.status_code == 200
        assert dashboard.headers["cache-control"] == "no-store, private"

        expired = client.post(
            "/login",
            data={
                "email": "diagnostic@example.invalid",
                "password": "not-a-real-password",
                "csrf_token": "expired-form-token",
            },
        )
        assert expired.status_code == 200
        assert expired.history and expired.history[0].status_code == 303
        assert "Your login page expired. Please try again." in expired.text
        assert 'name="csrf_token"' in expired.text


def test_live_historical_prediction_api_raw_rounded_horizons_and_small_values() -> None:
    user = SimpleNamespace(full_name="Forecast Test", hospital_name="Isolated test", email="forecast@example.invalid")
    main.app.dependency_overrides[main.get_current_hospital_user] = lambda: user
    try:
        with TestClient(main.app) as client:
            login = client.get("/login")
            token_match = re.search(r'name="csrf_token"\s+value="([^"]+)', login.text)
            assert token_match
            headers = {"x-csrf-token": token_match.group(1)}

            expected = {
                2: ("catboost", 10.094583339849212, 10, "2025-09-01"),
                3: ("gradient_boosting", 10.869329483670965, 11, "2025-09-08"),
                4: ("gradient_boosting", 9.506004642200926, 10, "2025-09-15"),
            }
            for horizon, (model, raw, rounded, target_date) in expected.items():
                response = client.post(
                    "/api/predict",
                    json={"city": "Chattogram", "ward": "CCC_Ward_01", "horizonWeeks": horizon, "asOf": "2025-08-18", "strictEvaluation": True},
                    headers=headers,
                )
                assert response.status_code == 200
                body = response.json()
                assert body["city"] == "Chattogram"
                assert body["wardId"] == "CCC_Ward_01"
                assert body["forecastOrigin"] == "2025-08-18"
                assert body["targetDate"] == target_date
                assert body["model"] == model
                assert body["rawPredictedCases"] == pytest.approx(raw, abs=1e-12)
                assert body["expectedPredictedCases"] == pytest.approx(raw, abs=1e-12)
                assert body["predictedCases"] == rounded
                assert body["mode"] == "strict_evaluation_replay"
                assert body["isHistoricalReplay"] is True
                assert body["isRetrospectiveEstimate"] is False
                assert body["requestedAsOf"] == "2025-08-18"
                assert body["modelTrainingDataThrough"] == "2025-08-18"
                assert body["dataFreshness"] == "stale"
                assert body["experimentId"] == "aedisight_200_iter_20261126_215231"
                assert body["requestId"]

            small = client.post(
                "/api/predict",
                json={"city": "Chattogram", "ward": "CCC_Ward_01", "horizonWeeks": 3, "asOf": "2025-01-06", "strictEvaluation": True},
                headers=headers,
            )
            assert small.status_code == 200
            small_body = small.json()
            assert small_body["rawPredictedCases"] == pytest.approx(0.11121739967171428)
            assert small_body["expectedPredictedCases"] == pytest.approx(0.11121739967171428)
            assert small_body["predictedCases"] == 0

            for horizon, target_date in ((2, "2025-09-01"), (3, "2025-09-08"), (4, "2025-09-15")):
                flexible = client.post(
                    "/api/predict",
                    json={"city": "Chattogram", "ward": "CCC_Ward_01", "horizonWeeks": horizon, "asOf": "2025-08-19"},
                    headers=headers,
                )
                assert flexible.status_code == 200
                flexible_body = flexible.json()
                assert flexible_body["requestedAsOf"] == "2025-08-19"
                assert flexible_body["observationDate"] == "2025-08-18"
                assert flexible_body["targetDate"] == target_date
                assert flexible_body["horizonWeeks"] == horizon
                assert flexible_body["mode"] == "flexible_historical_prediction"
                assert flexible_body["isHistoricalReplay"] is False
                assert flexible_body["isRetrospectiveEstimate"] is True
                assert flexible_body["modelTrainingDataThrough"] == "2025-12-29"
                assert "not a leakage-safe historical backtest" in flexible_body["predictionWarning"]
                assert flexible_body["historicalReplayPartition"] is None
                assert flexible_body["model"] == expected[horizon][0]

            stale_current = client.post(
                "/api/predict",
                json={"city": "Chattogram", "ward": "CCC_Ward_01", "horizonWeeks": 3},
                headers=headers,
            )
            assert stale_current.status_code == 503
            assert "Current forecasting is disabled" in stale_current.json()["detail"]
    finally:
        main.app.dependency_overrides.pop(main.get_current_hospital_user, None)


def test_dashboard_contains_continuous_and_rounded_prediction_display() -> None:
    template = Path("templates/user/user_dashboard.html").read_text(encoding="utf-8")
    script = Path("static/js/user/user.js").read_text(encoding="utf-8")
    assert 'id="expectedPredictedCases"' in template
    assert 'id="roundedPredictedCases"' in template
    assert 'id="predictionModel"' in template
    assert "result.rawPredictedCases" in script
    assert "result.expectedPredictedCases" in script
    assert 'getElementById("expectedPredictedCases")' in script
    assert 'getElementById("roundedPredictedCases")' in script
    assert "city: selection.city" in script
    assert "ward: selection.ward" in script
    assert "horizonWeeks: selection.horizonWeeks" in script
    assert "asOf: selection.asOf || null" in script
    assert "data-date-bounds" in template
    assert "function updateDateBounds()" in script
    assert "Choose a supported held-out replay origin" not in script
    assert "Your dashboard session expired or changed" in script

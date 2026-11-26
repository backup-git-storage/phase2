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
        "generate_forecast",
        lambda city, ward, horizon, as_of: {
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
        def fail_forecast(*_args, error=failure):
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

            def stale_forecast(*_args):
                raise main.ForecastUnavailableError("Current forecasting is disabled")

            monkeypatch.setattr(main, "generate_forecast", stale_forecast)
            stale = client.post("/api/predict", json=payload, headers={"x-csrf-token": token})
            assert stale.status_code == 503

            def invalid_ward(*_args):
                raise LookupError("Unknown ward")

            monkeypatch.setattr(main, "generate_forecast", invalid_ward)
            invalid = client.post("/api/predict", json=payload, headers={"x-csrf-token": token})
            assert invalid.status_code == 422
        finally:
            main.app.dependency_overrides.pop(main.get_current_hospital_user, None)

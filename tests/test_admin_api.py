"""Admin API tests for authorization, filtering, audit history, and health data."""

from __future__ import annotations

import os
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SESSION_SECRET", "admin-api-tests-session-secret-at-least-32-characters")
os.environ.setdefault("COOKIE_SECURE", "false")

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from src.backend.database import Base, get_db
from src.backend.models import AuditLog, User


@pytest.fixture
def admin_api_client(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_sessions = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    def override_db():
        db = test_sessions()
        try:
            yield db
        finally:
            db.close()

    with test_sessions() as db:
        admin = User(
            full_name="Platform Admin", hospital_name="AEDISIGHT",
            email="admin@example.com", hashed_password="not-rendered",
            is_admin=True, is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        active = User(
            full_name="Central Hospital", hospital_name="Central Hospital",
            email="central@example.com", hashed_password="hospital-hash",
            is_admin=False, is_active=True,
            created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        inactive = User(
            full_name="North Hospital", hospital_name="North Hospital",
            email="north@example.com", hashed_password="inactive-hash",
            is_admin=False, is_active=False,
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        db.add_all([admin, active, inactive])
        db.commit()
        db.add(AuditLog(
            event_type="user_registered",
            description="A hospital account was registered.",
            actor_user_id=active.id,
        ))
        db.commit()

    main.app.dependency_overrides[get_db] = override_db
    main.app.dependency_overrides[main.get_current_admin_user] = lambda: SimpleNamespace(
        id=1, full_name="Platform Admin", is_admin=True
    )
    monkeypatch.setattr(main, "get_available_horizons", lambda: [2, 3, 4])
    monkeypatch.setattr(main, "data_freshness", lambda: {
        "dataThrough": "2025-12-29", "status": "stale", "ageDays": 332,
    })

    with TestClient(main.app) as client:
        yield client

    main.app.dependency_overrides.pop(get_db, None)
    main.app.dependency_overrides.pop(main.get_current_admin_user, None)
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_admin_summary_uses_database_records(admin_api_client):
    response = admin_api_client.get("/api/admin/dashboard/summary")
    assert response.status_code == 200
    assert response.json() == {
        "total_users": 3,
        "active_hospital_accounts": 1,
        "administrator_accounts": 1,
        "inactive_accounts": 1,
    }


def test_admin_user_api_search_role_and_status_filters(admin_api_client):
    response = admin_api_client.get(
        "/api/admin/users", params={"search": "NORTH", "role": "hospital", "status": "inactive"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["users"][0]["email"] == "north@example.com"
    assert "hashed_password" not in payload["users"][0]


def test_admin_activity_returns_persisted_audit_records(admin_api_client):
    response = admin_api_client.get("/api/admin/activity")
    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["event_type"] == "user_registered"
    assert events[0]["description"] == "A hospital account was registered."


def test_admin_system_status_reports_observed_artifacts_and_freshness(admin_api_client):
    response = admin_api_client.get("/api/admin/system-status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "available"
    assert payload["database"] == "available"
    assert payload["available_horizons"] == [2, 3, 4]
    assert payload["data_freshness"] == "stale"


def test_admin_api_is_not_available_to_unauthenticated_visitors():
    with TestClient(main.app) as client:
        response = client.get("/api/admin/dashboard/summary", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_hospital_user_cannot_access_admin_api():
    from src.backend.dependencies import get_current_user

    main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=99, full_name="Hospital User", is_admin=False, is_active=True
    )
    try:
        with TestClient(main.app) as client:
            response = client.get("/api/admin/dashboard/summary", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/user_dashboard"
    finally:
        main.app.dependency_overrides.pop(get_current_user, None)

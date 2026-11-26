"""Database-backed admin dashboard rendering tests."""

from __future__ import annotations

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault(
    "SESSION_SECRET",
    "admin-dashboard-tests-session-secret-at-least-32-characters",
)
os.environ.setdefault("COOKIE_SECURE", "false")

from datetime import datetime, timezone
from types import SimpleNamespace
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from src.backend.database import Base, get_db
from src.backend.models import User


@pytest.fixture
def admin_dashboard_context():
    """Provide an isolated database and an authenticated admin dashboard client."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    with TestSessionLocal() as db:
        db.add_all([
            User(
                full_name="Platform Administrator",
                hospital_name="AEDISIGHT",
                email="admin@example.com",
                hashed_password="test-hash-not-a-real-credential",
                is_admin=True,
                is_active=True,
                terms_accepted=True,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            User(
                full_name="Active Hospital",
                hospital_name="Central Hospital",
                email="active@example.com",
                hashed_password="test-hash",
                is_admin=False,
                is_active=True,
                terms_accepted=True,
                created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            ),
            User(
                full_name="Inactive Hospital",
                hospital_name="North Hospital",
                email="inactive@example.com",
                hashed_password="test-hash",
                is_admin=False,
                is_active=False,
                terms_accepted=True,
                created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
            ),
            User(
                full_name="Inactive Administrator",
                hospital_name="AEDISIGHT",
                email="inactive-admin@example.com",
                hashed_password="test-hash",
                is_admin=True,
                is_active=False,
                terms_accepted=True,
                created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        db.commit()

    # Keep this test focused on database-backed rendering; authentication
    # dependency behavior is covered by the existing authorization tests.
    with TestClient(main.app) as client:
        main.app.dependency_overrides[get_db] = override_get_db
        main.app.dependency_overrides[main.get_current_admin_user] = (
            lambda: SimpleNamespace(full_name="Platform Administrator", is_admin=True)
        )
        try:
            yield client
        finally:
            main.app.dependency_overrides.pop(get_db, None)
            main.app.dependency_overrides.pop(main.get_current_admin_user, None)

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_admin_dashboard_renders_live_database_statistics_and_accounts(
    admin_dashboard_context,
):
    client = admin_dashboard_context

    response = client.get("/admin_dashboard")

    assert response.status_code == 200
    html = response.text

    # Four actual records were inserted into the isolated test database.
    assert 'Total registered users' in html
    assert 'Current platform count' in html
    assert 'Active hospital accounts' in html
    assert 'Current active count' in html
    assert 'Administrator accounts' in html
    assert 'Inactive accounts' in html

    # Values are rendered in the statistic cards, not the previous placeholders.
    rendered_stats = re.findall(
        r'<strong class="stat-value"[^>]*>\s*(\d+)\s*</strong>', html
    )
    assert rendered_stats == ["4", "1", "2", "2"]

    # All records appear in the existing table, newest registration first.
    assert 'active@example.com' in html
    assert 'inactive@example.com' in html
    assert 'inactive-admin@example.com' in html
    assert html.index('inactive-admin@example.com') < html.index('inactive@example.com')

    # Password hashes must never be rendered into the HTML response.
    assert 'test-hash' not in html


def test_admin_dashboard_remains_protected_for_unauthenticated_requests():
    with TestClient(main.app) as client:
        response = client.get("/admin_dashboard", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"

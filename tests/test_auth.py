"""Integration tests for authentication and account security."""

import os
from urllib.parse import parse_qs, urlparse

import pytest

# Set an isolated test database before importing application modules.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SESSION_SECRET"] = (
    "unit-test-only-session-secret-at-least-32-characters"
)

from fastapi import Depends, FastAPI, Request
from src.backend.dependencies import (
    get_current_hospital_user,
    get_current_admin_user,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.middleware.sessions import SessionMiddleware

from src.backend.auth import router as auth_router
from src.backend.database import Base, build_engine, get_db
from src.backend.models import User
from src.backend.security import (
    get_csrf_token,
    hash_password,
    verify_password,
)


TEST_SESSION_SECRET = (
    "unit-test-only-session-secret-at-least-32-characters"
)

@pytest.fixture
def auth_context():
    """Create an isolated FastAPI app and SQLite database for each test."""

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

    app = FastAPI()

    app.add_middleware(
        SessionMiddleware,
        secret_key=TEST_SESSION_SECRET,
        session_cookie="test_session",
        same_site="lax",
        https_only=False,
    )

    app.include_router(auth_router)

    def override_get_db():
        db = TestSessionLocal()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Test-only endpoint to obtain a valid CSRF token.
    @app.get("/_test/csrf")
    def test_csrf(request: Request):
        return {
            "csrf_token": get_csrf_token(request)
        }

    # Test-only endpoint to inspect the authenticated session.
    @app.get("/_test/session")
    def test_session(request: Request):
        return {
            "user_id": request.session.get("user_id")
        }

    # Test-only hospital-protected endpoint.
    @app.get("/_test/hospital")
    def test_hospital_access(
        current_user: User = Depends(get_current_hospital_user),
    ):
        return {
            "user_id": current_user.id,
            "is_admin": current_user.is_admin,
        }

    # Test-only administrator-protected endpoint.
    @app.get("/_test/admin")
    def test_admin_access(
        current_user: User = Depends(get_current_admin_user),
    ):
        return {
            "user_id": current_user.id,
            "is_admin": current_user.is_admin,
        }
    try:
        with TestClient(app) as client:
            yield client, TestSessionLocal

    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def get_test_csrf(client):
    """Obtain a CSRF token while preserving the session cookie."""

    response = client.get("/_test/csrf")

    assert response.status_code == 200

    return response.json()["csrf_token"]


def get_redirect_error(response):
    """Extract the error query parameter from a redirect response."""

    location = response.headers["location"]

    return parse_qs(urlparse(location).query).get("error", [None])[0]


def create_test_user(
    session_factory,
    *,
    email="hospital@example.com",
    password="StrongPassword123!",
    is_active=True,
    is_admin=False,
):
    """Create a database user for authentication tests."""

    with session_factory() as db:

        user = User(
            full_name="Test Hospital User",
            hospital_name="Test Hospital",
            email=email,
            hashed_password=hash_password(password),
            is_admin=is_admin,
            is_active=is_active,
            terms_accepted=True,
        )

        db.add(user)
        db.commit()

        return user.id


def registration_form(token, **overrides):
    """Build a valid registration form with optional overrides."""

    data = {
        "full_name": "john doe",
        "hospital_name": "Test Hospital",
        "email": "newhospital@example.com",
        "password": "StrongPassword123!",
        "confirm_password": "StrongPassword123!",
        "terms_accepted": "on",
        "csrf_token": token,
    }

    data.update(overrides)

    return data


# --------------------------------------------------
# REGISTRATION TESTS
# --------------------------------------------------


def test_registration_creates_hospital_user(auth_context):
    """A valid registration should create a normal hospital account."""

    client, session_factory = auth_context

    csrf_token = get_test_csrf(client)

    password = "StrongPassword123!"

    response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            password=password,
            confirm_password=password,
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?")

    with session_factory() as db:

        user = (
            db.query(User)
            .filter(User.email == "newhospital@example.com")
            .one_or_none()
        )

        assert user is not None
        assert user.full_name == "John Doe"
        assert user.hospital_name == "Test Hospital"
        assert user.is_admin is False
        assert user.is_active is True
        assert user.terms_accepted is True

        assert user.hashed_password != password
        assert verify_password(password, user.hashed_password)


def test_registered_user_can_log_in(auth_context):
    """A newly registered account can immediately use the login route."""

    client, session_factory = auth_context
    password = "StrongPassword123!"
    csrf_token = get_test_csrf(client)

    registration_response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            password=password,
            confirm_password=password,
        ),
        follow_redirects=False,
    )

    assert registration_response.status_code == 303
    assert registration_response.headers["location"].startswith("/login?")

    login_response = client.post(
        "/login",
        data={
            "email": "NEWHOSPITAL@example.com",
            "password": password,
            "csrf_token": get_test_csrf(client),
        },
        follow_redirects=False,
    )

    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/user_dashboard"

    with session_factory() as db:
        user = (
            db.query(User)
            .filter(User.email == "newhospital@example.com")
            .one()
        )
        assert client.get("/_test/session").json()["user_id"] == user.id


def test_production_database_rejects_sqlite(monkeypatch):
    """Production auth storage must be persistent across app instances."""

    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(RuntimeError, match="persistent PostgreSQL"):
        build_engine("sqlite:///:memory:")


def test_registration_rejects_invalid_csrf(auth_context):
    """Invalid CSRF tokens must prevent account creation."""

    client, session_factory = auth_context

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            csrf_token="invalid-token",
        ),
        follow_redirects=False,
    )

    assert response.status_code == 403

    with session_factory() as db:
        assert db.query(User).count() == 0


def test_registration_rejects_password_mismatch(auth_context):
    """Registration must reject different password values."""

    client, session_factory = auth_context

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            confirm_password="DifferentPassword123!",
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303

    assert (
        get_redirect_error(response)
        == "Passwords do not match."
    )

    with session_factory() as db:
        assert db.query(User).count() == 0


def test_registration_requires_terms_acceptance(auth_context):
    """Registration must reject users who have not accepted the terms."""

    client, session_factory = auth_context

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            terms_accepted="",
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303

    assert (
        get_redirect_error(response)
        == "You must accept the platform terms."
    )

    with session_factory() as db:
        assert db.query(User).count() == 0


def test_registration_rejects_duplicate_email(auth_context):
    """Email duplication must be checked case-insensitively."""

    client, session_factory = auth_context

    create_test_user(
        session_factory,
        email="existing@example.com",
    )

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            email="EXISTING@example.com",
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303

    assert (
        get_redirect_error(response)
        == "An account with this email already exists."
    )

    with session_factory() as db:
        assert db.query(User).count() == 1


# --------------------------------------------------
# LOGIN TESTS
# --------------------------------------------------


def test_login_creates_hospital_user_session(auth_context):
    """Successful login should establish the hospital user session."""

    client, session_factory = auth_context

    user_id = create_test_user(session_factory)

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/login",
        data={
            "email": "HOSPITAL@example.com",
            "password": "StrongPassword123!",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/user_dashboard"

    session_response = client.get("/_test/session")

    assert session_response.status_code == 200
    assert session_response.json()["user_id"] == user_id


def test_login_rejects_invalid_password(auth_context):
    """Incorrect credentials must not establish a session."""

    client, session_factory = auth_context

    create_test_user(session_factory)

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "WrongPassword123!",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303

    assert (
        get_redirect_error(response)
        == "Invalid email or password."
    )

    session_response = client.get("/_test/session")

    assert session_response.json()["user_id"] is None


def test_login_rejects_inactive_account(auth_context):
    """Inactive users must not be allowed to log in."""

    client, session_factory = auth_context

    create_test_user(
        session_factory,
        is_active=False,
    )

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303

    assert (
        get_redirect_error(response)
        == "This account is currently inactive."
    )

    session_response = client.get("/_test/session")

    assert session_response.json()["user_id"] is None


# --------------------------------------------------
# LOGOUT TESTS
# --------------------------------------------------


def test_logout_clears_authenticated_session(auth_context):
    """Logout must remove the authenticated user from the session."""

    client, session_factory = auth_context

    create_test_user(session_factory)

    csrf_token = get_test_csrf(client)

    login_response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert login_response.status_code == 303

    session_response = client.get("/_test/session")

    assert session_response.json()["user_id"] is not None

    # Login rotates the CSRF token, so retrieve the new one.
    logout_csrf_token = get_test_csrf(client)

    logout_response = client.post(
        "/logout",
        data={
            "csrf_token": logout_csrf_token,
        },
        follow_redirects=False,
    )

    assert logout_response.status_code == 303
    assert logout_response.headers["location"] == "/login"

    session_response = client.get("/_test/session")

    assert session_response.json()["user_id"] is None


# --------------------------------------------------
# ADDITIONAL STAGE 8 SECURITY AND ACCESS TESTS
# --------------------------------------------------


def test_registration_rejects_invalid_email(auth_context):
    """Invalid email addresses must not create accounts."""

    client, session_factory = auth_context
    token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(token, email="not-an-email"),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert get_redirect_error(response) == "Enter a valid email address."

    with session_factory() as db:
        assert db.query(User).count() == 0


def test_public_registration_cannot_create_admin(auth_context):
    """Extra admin fields must never grant public admin privileges."""

    client, session_factory = auth_context
    token = get_test_csrf(client)

    data = registration_form(token)
    data["is_admin"] = "true"

    response = client.post(
        "/register",
        data=data,
        follow_redirects=False,
    )

    assert response.status_code == 303

    with session_factory() as db:
        user = (
            db.query(User)
            .filter(User.email == "newhospital@example.com")
            .one()
        )

        assert user.is_admin is False


def test_login_rejects_invalid_csrf(auth_context):
    """Login rejects stale CSRF but recovers with a fresh uncached page."""

    client, session_factory = auth_context
    create_test_user(session_factory)

    response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": "invalid-token",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?error=")
    assert client.get("/_test/session").json()["user_id"] is None


def test_logout_rejects_invalid_csrf_and_keeps_session(auth_context):
    """Invalid logout tokens must not terminate or change the session."""

    client, session_factory = auth_context
    user_id = create_test_user(session_factory)

    token = get_test_csrf(client)

    login_response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": token,
        },
        follow_redirects=False,
    )

    assert login_response.status_code == 303

    response = client.post(
        "/logout",
        data={"csrf_token": "invalid-token"},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert client.get("/_test/session").json()["user_id"] == user_id


def test_admin_login_redirects_to_admin_dashboard(auth_context):
    """Administrators must be redirected to their own dashboard."""

    client, session_factory = auth_context

    user_id = create_test_user(
        session_factory,
        is_admin=True,
    )

    token = get_test_csrf(client)

    response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/admin_dashboard"

    assert client.get("/_test/session").json()["user_id"] == user_id


def test_unauthenticated_users_cannot_access_protected_routes(auth_context):
    """Unauthenticated visitors must be redirected to login."""

    client, _ = auth_context

    for path in ["/_test/hospital", "/_test/admin"]:
        response = client.get(
            path,
            follow_redirects=False,
        )

        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_hospital_user_cannot_access_admin_route(auth_context):
    """Hospital users must not access administrator-only resources."""

    client, session_factory = auth_context
    create_test_user(session_factory)

    token = get_test_csrf(client)

    login_response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": token,
        },
        follow_redirects=False,
    )

    assert login_response.status_code == 303

    response = client.get(
        "/_test/admin",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/user_dashboard"


def test_admin_cannot_access_hospital_only_route(auth_context):
    """Administrators must be redirected away from hospital-only resources."""

    client, session_factory = auth_context

    create_test_user(
        session_factory,
        is_admin=True,
    )

    token = get_test_csrf(client)

    login_response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": token,
        },
        follow_redirects=False,
    )

    assert login_response.status_code == 303

    response = client.get(
        "/_test/hospital",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/admin_dashboard"


def test_inactive_account_session_is_invalidated(auth_context):
    """An account deactivated after login must lose protected access."""

    client, session_factory = auth_context
    user_id = create_test_user(session_factory)

    token = get_test_csrf(client)

    login_response = client.post(
        "/login",
        data={
            "email": "hospital@example.com",
            "password": "StrongPassword123!",
            "csrf_token": token,
        },
        follow_redirects=False,
    )

    assert login_response.status_code == 303

    with session_factory() as db:
        user = db.get(User, user_id)
        user.is_active = False
        db.commit()

    response = client.get(
        "/_test/hospital",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/login"

    assert client.get("/_test/session").json()["user_id"] is None


@pytest.mark.parametrize(
    "weak_password, expected_error",
    [
        ("12345678", "Password must contain at least one letter."),
        ("abcdefgh!", "Password must contain at least one number."),
        ("Password123", "Password must contain at least one special character."),
    ],
)
def test_registration_rejects_weak_passwords(
    auth_context,
    weak_password,
    expected_error,
):
    """Passwords need a letter, a number and a special character."""

    client, session_factory = auth_context

    csrf_token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(
            csrf_token,
            password=weak_password,
            confirm_password=weak_password,
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert get_redirect_error(response) == expected_error

    with session_factory() as db:
        assert db.query(User).count() == 0


@pytest.mark.parametrize(
    "email",
    [
        "cscr@gmail.co",
        "user@example.xyz",
        "user@hospital.bd",
    ],
)
def test_registration_rejects_unsupported_email_domain(auth_context, email):
    """Email addresses must end in an allowed domain such as .com or .edu."""

    client, session_factory = auth_context
    token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(token, email=email),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "Use an email ending in" in get_redirect_error(response)

    with session_factory() as db:
        assert db.query(User).count() == 0


@pytest.mark.parametrize(
    "email",
    [
        "doctor@hospital.com",
        "admin@dmc.edu.bd",
        "staff@health.gov.bd",
    ],
)
def test_registration_accepts_allowed_email_domain(auth_context, email):
    """Allowed domain endings must still register successfully."""

    client, session_factory = auth_context
    token = get_test_csrf(client)

    response = client.post(
        "/register",
        data=registration_form(token, email=email),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")

    with session_factory() as db:
        assert db.query(User).count() == 1

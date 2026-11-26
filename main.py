
"""FastAPI application entry point for DengueGuard."""

import os
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Resolve paths independently of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables before importing database configuration.
load_dotenv(PROJECT_ROOT / ".env", override=False)

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
# --------------------------------------------------
# APPLICATION ENVIRONMENT AND SESSION CONFIGURATION
# --------------------------------------------------

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

if APP_ENV not in {"development", "test", "production"}:
    raise RuntimeError(
        "APP_ENV must be development, test, or production."
    )

SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()

if (
    not SESSION_SECRET
    or SESSION_SECRET == "replace_with_a_random_secret"
    or len(SESSION_SECRET) < 32
):
    raise RuntimeError(
        "SESSION_SECRET is missing or too short. "
        "Generate a secure secret and add it to your .env file."
    )

try:
    SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", "3600"))
except ValueError as exc:
    raise RuntimeError("SESSION_MAX_AGE must be an integer.") from exc

if SESSION_MAX_AGE <= 0:
    raise RuntimeError("SESSION_MAX_AGE must be greater than zero.")

# Secure cookies default to enabled in production.
cookie_secure_raw = os.getenv("COOKIE_SECURE")

if cookie_secure_raw is None:
    COOKIE_SECURE = APP_ENV == "production"
else:
    cookie_secure_value = cookie_secure_raw.strip().lower()

    if cookie_secure_value not in {
        "true", "false", "1", "0", "yes", "no", "on", "off"
    }:
        raise RuntimeError(
            "COOKIE_SECURE must be a valid boolean value."
        )

    COOKIE_SECURE = cookie_secure_value in {
        "true", "1", "yes", "on"
    }

if APP_ENV == "production" and not COOKIE_SECURE:
    raise RuntimeError(
        "COOKIE_SECURE must be true in production. "
        "Production session cookies must only be sent over HTTPS."
    )

from mangum import Mangum
from src.backend.database import Base, engine, get_db
from src.backend.models import User
from src.backend.auth import router as auth_router
from src.backend.security import get_csrf_token, verify_csrf_token

from src.backend.dependencies import (
    get_current_hospital_user,
    get_current_admin_user,
)
from src.predict import (
    ForecastUnavailableError,
    ModelUnavailableError,
    data_freshness,
    generate_forecast,
    get_available_horizons,
    get_data_through,
    replay_origin_bounds,
    get_supported_areas,
)
from src.feature_engineering import SUPPORTED_HORIZONS


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="DengueGuard",
    description="Urban Dengue Early Warning System",
    version="1.0.0",
)


# --------------------------------------------------
# SESSION MIDDLEWARE
# --------------------------------------------------

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="dengue_session",
    max_age=SESSION_MAX_AGE,
    same_site="lax",
    https_only=COOKIE_SECURE,
)


# --------------------------------------------------
# STATIC FILES AND TEMPLATES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=str(PROJECT_ROOT / "templates")
)


class ForecastRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    ward: str = Field(min_length=1, max_length=100)
    horizonWeeks: int = Field(ge=2, le=4)
    asOf: Optional[date] = None


@app.post("/api/predict")
async def predict_ward(
    payload: ForecastRequest,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
):
    """Serve an authenticated forecast from the selected trained model run."""
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    if payload.horizonWeeks not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported forecast horizon; choose one of {SUPPORTED_HORIZONS} weeks.",
        )
    try:
        return generate_forecast(
            payload.city,
            payload.ward,
            payload.horizonWeeks,
            payload.asOf.isoformat() if payload.asOf else None,
        )
    except (FileNotFoundError, ForecastUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) if isinstance(exc, ForecastUnavailableError) else "A compatible forecasting model or required data is unavailable.",
        ) from exc
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed because the model or processed data could not be used.",
        ) from exc


# --------------------------------------------------
# AUTHENTICATION ROUTER
# --------------------------------------------------

app.include_router(auth_router)


# --------------------------------------------------
# PUBLIC HOME PAGE
# --------------------------------------------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home/index.html",
        context={},
    )


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

@app.get("/login")
async def login_page(request: Request):

    csrf_token = get_csrf_token(request)

    return templates.TemplateResponse(
        request=request,
        name="auth/login/login.html",
        context={
            "csrf_token": csrf_token,
        },
    )


# --------------------------------------------------
# REGISTRATION PAGE
# --------------------------------------------------

@app.get("/register")
async def register_page(request: Request):

    csrf_token = get_csrf_token(request)

    return templates.TemplateResponse(
        request=request,
        name="auth/register/register.html",
        context={
            "csrf_token": csrf_token,
        },
    )


# --------------------------------------------------
# HOSPITAL USER DASHBOARD
# --------------------------------------------------

@app.get("/user_dashboard")
async def user_dashboard(
    request: Request,
    current_user: User = Depends(get_current_hospital_user),
):

    csrf_token = get_csrf_token(request)
    data_error = None
    model_horizons = get_available_horizons()
    data_through = None
    freshness = None
    replay_origins = {}
    try:
        areas = get_supported_areas()
        data_through = get_data_through()
        freshness = data_freshness()
        replay_origins = {
            horizon: replay_origin_bounds(horizon) for horizon in model_horizons
        }
    except (FileNotFoundError, ValueError, ForecastUnavailableError):
        areas = {}
        data_error = "Processed ward data is unavailable or invalid. Check the server configuration."

    return templates.TemplateResponse(
        request=request,
        name="user/user_dashboard.html",
        context={
            "current_user": current_user,
            "csrf_token": csrf_token,
            "areas": areas,
            "data_error": data_error,
            "model_horizons": model_horizons,
            "data_through": data_through,
            "freshness": freshness,
            "replay_origins": replay_origins,
        },
    )


# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------

@app.get("/admin_dashboard")
async def admin_dashboard(
    request: Request,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Render the admin dashboard using live account records from the database."""

    csrf_token = get_csrf_token(request)

    # Keep the dashboard statistics aligned with the same users table shown below.
    # Hospital accounts exclude administrators; administrator count includes both
    # active and inactive administrator accounts.
    total_users = db.query(func.count(User.id)).scalar() or 0
    active_hospital_accounts = (
        db.query(func.count(User.id))
        .filter(User.is_admin.is_(False), User.is_active.is_(True))
        .scalar()
        or 0
    )
    administrator_accounts = (
        db.query(func.count(User.id))
        .filter(User.is_admin.is_(True))
        .scalar()
        or 0
    )
    inactive_accounts = (
        db.query(func.count(User.id))
        .filter(User.is_active.is_(False))
        .scalar()
        or 0
    )

    # Do not expose password hashes or other internal fields to the template.
    users = (
        db.query(User)
        .order_by(User.created_at.desc(), User.id.desc())
        .all()
    )

    return templates.TemplateResponse(
        request=request,
        name="admin/admin_dashboard.html",
        context={
            "current_user": current_user,
            "csrf_token": csrf_token,
            "total_users": total_users,
            "active_hospital_accounts": active_hospital_accounts,
            "administrator_accounts": administrator_accounts,
            "inactive_accounts": inactive_accounts,
            "users": users,
        },
    )


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# VERCEL ASGI HANDLER
# --------------------------------------------------

handler = Mangum(app)

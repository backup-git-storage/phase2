
"""FastAPI application entry point for DengueGuard."""

import os
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Resolve paths independently of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables before importing database configuration.
load_dotenv(PROJECT_ROOT / ".env", override=False)

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from sqlalchemy import func, text
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

from src.backend.database import Base, engine, get_db
from src.backend.models import AuditLog, User
from src.backend.auth import router as auth_router
from src.backend.security import get_csrf_token, verify_csrf_token

from src.backend.dependencies import (
    get_current_hospital_user,
    get_current_admin_user,
)
PREDICTION_BACKEND = os.getenv("PREDICTION_BACKEND", "reference_catboost").strip().lower()
if PREDICTION_BACKEND == "reference_catboost":
    from src.reference_catboost_predict import (
        ForecastUnavailableError, ModelUnavailableError, data_freshness,
        generate_forecast, get_available_horizons, get_data_through,
        get_supported_date_bounds, get_supported_areas, DATA_PATH,
    )
elif PREDICTION_BACKEND == "phase2":
    from src.predict import (
        ForecastUnavailableError, ModelUnavailableError, data_freshness,
        generate_forecast, get_available_horizons, get_data_through,
        get_supported_date_bounds, get_supported_areas, DATA_PATH,
    )
else:
    raise RuntimeError("PREDICTION_BACKEND must be 'reference_catboost' or 'phase2'.")
from src.feature_engineering import SUPPORTED_HORIZONS

logger = logging.getLogger(__name__)


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
    strictEvaluation: bool = False


@app.post("/api/predict")
async def predict_ward(
    payload: ForecastRequest,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
):
    """Serve an authenticated forecast from the selected trained model run."""
    request_id = uuid.uuid4().hex
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    if payload.horizonWeeks not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported forecast horizon; choose one of {SUPPORTED_HORIZONS} weeks.",
        )
    try:
        result = generate_forecast(
            payload.city,
            payload.ward,
            payload.horizonWeeks,
            payload.asOf.isoformat() if payload.asOf else None,
            strict_evaluation=payload.strictEvaluation,
        )
        result["requestId"] = request_id
        logger.info(
            "forecast request_id=%s city=%s ward=%s horizon_weeks=%s origin=%s mode=%s "
            "model=%s experiment=%s features=%s expected_features=%s missing_features=%s "
            "raw_prediction=%s expected_prediction=%s rounded_prediction=%s freshness=%s",
            request_id,
            result.get("city"),
            result.get("wardId"),
            result.get("horizonWeeks"),
            result.get("forecastOrigin"),
            result.get("mode"),
            result.get("model"),
            result.get("experimentId"),
            result.get("modelInputFeatureCount"),
            result.get("expectedFeatureCount"),
            result.get("missingFeatureCount"),
            result.get("rawPredictedCases"),
            result.get("expectedPredictedCases"),
            result.get("predictedCases"),
            result.get("dataFreshness"),
        )
        return result
    except (FileNotFoundError, ForecastUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) if isinstance(exc, ForecastUnavailableError) else "A compatible forecasting model or required data is unavailable.",
        ) from exc
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "forecast request failed request_id=%s city=%s ward=%s horizon_weeks=%s origin=%s",
            request_id, payload.city, payload.ward, payload.horizonWeeks,
            payload.asOf.isoformat() if payload.asOf else None,
        )
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
        headers={"Cache-Control": "no-store, private"},
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
        headers={"Cache-Control": "no-store, private"},
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
    date_bounds = {}
    try:
        areas = get_supported_areas()
        data_through = get_data_through()
        freshness = data_freshness()
        date_bounds = get_supported_date_bounds()
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
            "date_bounds": date_bounds,
        },
        headers={"Cache-Control": "no-store, private"},
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
        headers={"Cache-Control": "no-store, private"},
    )


# --------------------------------------------------
# ADMIN-ONLY JSON API
# --------------------------------------------------


def _user_payload(user: User) -> dict:
    """Return only fields intended for the administrator user table."""
    return {
        "id": user.id,
        "full_name": user.full_name,
        "hospital_name": user.hospital_name,
        "email": user.email,
        "is_admin": bool(user.is_admin),
        "is_active": bool(user.is_active),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@app.get("/api/admin/dashboard/summary")
async def admin_dashboard_summary(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Return live account totals for the administrator dashboard."""
    return {
        "total_users": db.query(func.count(User.id)).scalar() or 0,
        "active_hospital_accounts": (
            db.query(func.count(User.id))
            .filter(User.is_admin.is_(False), User.is_active.is_(True))
            .scalar() or 0
        ),
        "administrator_accounts": (
            db.query(func.count(User.id))
            .filter(User.is_admin.is_(True))
            .scalar() or 0
        ),
        "inactive_accounts": (
            db.query(func.count(User.id))
            .filter(User.is_active.is_(False))
            .scalar() or 0
        ),
    }


@app.get("/api/admin/users")
async def admin_list_users(
    search: str = Query(default="", max_length=255),
    role: str = Query(default="", pattern="^(|admin|hospital)$"),
    status_filter: str = Query(default="", alias="status", pattern="^(|active|inactive)$"),
    limit: int = Query(default=500, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Search and filter user accounts without exposing password hashes."""
    query = db.query(User)
    term = search.strip().casefold()
    if term:
        pattern = f"%{term}%"
        query = query.filter(
            func.lower(User.full_name).like(pattern)
            | func.lower(User.hospital_name).like(pattern)
            | func.lower(User.email).like(pattern)
        )
    if role == "admin":
        query = query.filter(User.is_admin.is_(True))
    elif role == "hospital":
        query = query.filter(User.is_admin.is_(False))
    if status_filter == "active":
        query = query.filter(User.is_active.is_(True))
    elif status_filter == "inactive":
        query = query.filter(User.is_active.is_(False))

    total = query.count()
    records = (
        query.order_by(User.created_at.desc(), User.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "users": [_user_payload(user) for user in records],
    }


@app.get("/api/admin/activity")
async def admin_recent_activity(
    limit: int = Query(default=20, ge=1, le=50),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Return persisted, non-sensitive audit events newest first."""
    events = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(limit)
        .all()
    )
    return {
        "events": [
            {
                "id": event.id,
                "event_type": event.event_type,
                "description": event.description,
                "timestamp": event.created_at.isoformat(),
            }
            for event in events
        ]
    }


@app.get("/api/admin/system-status")
async def admin_system_status(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Report observed database, dataset, and model-artifact availability."""
    try:
        db.execute(text("SELECT 1"))
        database_status = "available"
    except Exception:
        database_status = "unavailable"

    processed_status = "missing"
    data_through = None
    freshness_status = "unknown"
    if DATA_PATH.is_file():
        processed_status = "available"
        try:
            freshness = data_freshness()
            data_through = freshness.get("dataThrough")
            freshness_status = freshness.get("status", "unknown")
        except Exception:
            processed_status = "invalid"
            freshness_status = "invalid"

    try:
        available_horizons = get_available_horizons()
    except Exception:
        available_horizons = []

    return {
        "backend": "available",
        "database": database_status,
        "raw_dataset": "not_configured",
        "processed_dataset": processed_status,
        "data_through": data_through,
        "data_freshness": freshness_status,
        "model_artifacts": "available" if available_horizons else "unavailable",
        "available_horizons": available_horizons,
        "prediction_service": "available" if available_horizons else "unavailable",
        "application_version": app.version,
    }


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

Base.metadata.create_all(bind=engine)

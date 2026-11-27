"""Authentication endpoints for registration, login and logout."""

import logging
import re
from urllib.parse import urlencode

from email_validator import EmailNotValidError, validate_email

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from src.backend.database import get_db
from src.backend.models import User
from src.backend.audit import record_audit_event

from src.backend.security import (
    hash_password,
    verify_password,
    verify_csrf_token,
    rotate_csrf_token,
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])

def redirect_with_message(
    path: str,
    message: str | None = None,
    error: str | None = None,
):
    """Redirect to a fixed authentication page with a message."""

    parameters = {}

    if message:
        parameters["message"] = message

    if error:
        parameters["error"] = error

    if parameters:
        path = f"{path}?{urlencode(parameters)}"

    return RedirectResponse(
        url=path,
        status_code=status.HTTP_303_SEE_OTHER,
    )

def normalize_email(email: str) -> str:
    """Validate and normalize an email address."""

    if not email or not email.strip():
        raise ValueError("Email address is required.")

    try:
        result = validate_email(
            email.strip(),
            check_deliverability=False,
        )

        normalized = result.normalized.casefold()

        if len(normalized) > 255:
            raise ValueError("Email address is too long.")

        return normalized

    except EmailNotValidError as exc:
        raise ValueError("Enter a valid email address.") from exc

# Email endings accepted for new registrations. Edit this list to allow more.
ALLOWED_EMAIL_DOMAIN_ENDINGS = (
    ".com",
    ".edu",
    ".org",
    ".net",
    ".gov",
    ".com.bd",
    ".edu.bd",
    ".org.bd",
    ".net.bd",
    ".gov.bd",
    ".ac.bd",
)


def validate_email_domain(email: str) -> str | None:
    """
    Return an error message if the email domain ending is not allowed.

    Used for registration only, so existing accounts can still log in.
    Returns None when the domain is acceptable.
    """

    domain = email.rsplit("@", 1)[-1].casefold()

    if not domain.endswith(ALLOWED_EMAIL_DOMAIN_ENDINGS):
        return (
            "Use an email ending in .com, .edu, .org, .net, .gov "
            "or a Bangladeshi domain such as .com.bd or .edu.bd."
        )

    return None


def validate_password_strength(password: str) -> str | None:
    """
    Return an error message if the password is too weak.

    Requires at least one letter, one number and one special character.
    Returns None when the password is acceptable.
    """

    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter."

    if not re.search(r"[0-9]", password):
        return "Password must contain at least one number."

    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must contain at least one special character."

    return None


@router.post("/register", name="register")
def register_user(
    request: Request,

    full_name: str = Form(...),
    hospital_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    terms_accepted: str | None = Form(default=None),
    csrf_token: str = Form(...),

    db: Session = Depends(get_db),
):
    """Register a new hospital user."""

    # 1. Check CSRF token.
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(
            status_code=403,
            detail="Invalid security token. Reload the page and try again.",
        )

    # 2. Clean user information.
    full_name = " ".join(full_name.split()).title()
    hospital_name = hospital_name.strip()

    # 3. Validate required fields.
    if not full_name or not hospital_name:
        return redirect_with_message(
            "/register",
            error="Full name and hospital name are required.",
        )

    if len(full_name) > 150 or len(hospital_name) > 255:
        return redirect_with_message(
            "/register",
            error="One or more fields exceed the allowed length.",
        )

    # 4. Normalize email.
    try:
        email = normalize_email(email)

    except ValueError as exc:
        return redirect_with_message(
            "/register",
            error=str(exc),
        )

    domain_error = validate_email_domain(email)

    if domain_error:
        return redirect_with_message(
            "/register",
            error=domain_error,
        )

    # 5. Validate password.
    if len(password) < 8:
        return redirect_with_message(
            "/register",
            error="Password must contain at least 8 characters.",
        )

    if len(password) > 128:
        return redirect_with_message(
            "/register",
            error="Password cannot exceed 128 characters.",
        )

    strength_error = validate_password_strength(password)

    if strength_error:
        return redirect_with_message(
            "/register",
            error=strength_error,
        )

    if password != confirm_password:
        return redirect_with_message(
            "/register",
            error="Passwords do not match.",
        )

    # 6. Validate terms acceptance.
    accepted = (
        terms_accepted or ""
    ).strip().lower() in {"on", "true", "1", "yes"}

    if not accepted:
        return redirect_with_message(
            "/register",
            error="You must accept the platform terms.",
        )

    # 7. Check whether the email already exists.
    existing_user = (
        db.query(User)
        .filter(func.lower(User.email) == email)
        .first()
    )

    if existing_user:
        return redirect_with_message(
            "/register",
            error="An account with this email already exists.",
        )

    # 8. Create the user.
    # Public registration must NEVER assign administrator access.
    new_user = User(
        full_name=full_name,
        hospital_name=hospital_name,
        email=email,
        hashed_password=hash_password(password),
        is_admin=False,
        is_active=True,
        terms_accepted=True,
    )

    try:
        db.add(new_user)
        db.commit()

    except IntegrityError:
        db.rollback()

        return redirect_with_message(
            "/register",
            error="An account with this email already exists.",
        )

    except SQLAlchemyError:
        db.rollback()

        logger.exception("Database error during registration.")

        return redirect_with_message(
            "/register",
            error="Registration could not be completed. Please try again.",
        )

    record_audit_event(
        db, "user_registered", "A hospital account was registered.", new_user.id
    )

    # 9. Reset session state and rotate CSRF token.
    request.session.clear()
    rotate_csrf_token(request)

    # 10. Redirect to login.
    return redirect_with_message(
        "/login",
        message="Registration successful. Please log in.",
    )

@router.post("/login", name="login")
def login_user(
    request: Request,

    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),

    db: Session = Depends(get_db),
):
    """Authenticate a user and establish a session."""

    # 1. Verify CSRF token.
    if not verify_csrf_token(request, csrf_token):
        # Browser form submissions can outlive their signed session cookie
        # (for example, when a stale tab or cached login page is submitted).
        # Reject the request, then return the browser to a page with a fresh
        # token instead of displaying a raw JSON error document.
        return redirect_with_message(
            "/login",
            error="Your login page expired. Please try again.",
        )

    # 2. Normalize email.
    try:
        email = normalize_email(email)

    except ValueError:
        return redirect_with_message(
            "/login",
            error="Invalid email or password.",
        )

    # 3. Find the user.
    user = (
        db.query(User)
        .filter(func.lower(User.email) == email)
        .first()
    )

    # 4. Verify password.
    # Use the same generic message for invalid credentials.
    if user is None or not verify_password(
        password,
        user.hashed_password,
    ):
        return redirect_with_message(
            "/login",
            error="Invalid email or password.",
        )

    # 5. Reject disabled accounts.
    if not user.is_active:
        return redirect_with_message(
            "/login",
            error="This account is currently inactive.",
        )

    # 6. Clear the old session state.
    request.session.clear()

    # 7. Save only the user ID.
    request.session["user_id"] = user.id

    # 8. Generate a fresh CSRF token.
    rotate_csrf_token(request)

    record_audit_event(
        db,
        "user_login",
        "Administrator signed in." if user.is_admin else "Hospital user signed in.",
        user.id,
    )

    # 9. Determine dashboard using the database value.
    if user.is_admin:
        return RedirectResponse(
            url="/admin_dashboard",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return RedirectResponse(
        url="/user_dashboard",
        status_code=status.HTTP_303_SEE_OTHER,
    )

@router.post("/logout", name="logout")
def logout_user(
    request: Request,
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    """Log out the authenticated user."""

    # Verify CSRF token before changing session state.
    if not verify_csrf_token(request, csrf_token):
        raise HTTPException(
            status_code=403,
            detail="Invalid security token.",
        )

    # Record the authenticated account before clearing its session.
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if type(user_id) is int else None
    if user is not None:
        record_audit_event(
            db,
            "user_logout",
            "Administrator signed out." if user.is_admin else "Hospital user signed out.",
            user.id,
        )

    # Remove all session data.
    request.session.clear()

    # Redirect to login.
    return RedirectResponse(
        url="/login",
        status_code=status.HTTP_303_SEE_OTHER,
    )

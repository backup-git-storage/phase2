
"""Reusable authentication and authorization dependencies."""

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.backend.database import get_db
from src.backend.models import User


def _redirect(location: str) -> None:
    """Redirect a browser to the appropriate page."""
    raise HTTPException(
        status_code=status.HTTP_303_SEE_OTHER,
        detail="Redirecting",
        headers={"Location": location},
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """Return the currently authenticated, active user."""

    user_id = request.session.get("user_id")

    if type(user_id) is not int:
        request.session.clear()
        _redirect("/login")

    user = db.get(User, user_id)

    if user is None or not user.is_active:
        request.session.clear()
        _redirect("/login")

    return user


def get_current_hospital_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Allow hospital users, but not administrators, into the user dashboard."""

    if current_user.is_admin:
        _redirect("/admin_dashboard")

    return current_user


def get_current_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Allow administrators only into administrator pages."""

    if not current_user.is_admin:
        _redirect("/user_dashboard")

    return current_user

"""Password hashing and CSRF security utilities."""

import secrets

from pwdlib import PasswordHash
from starlette.requests import Request


# Use Argon2 as the recommended password hashing algorithm.
password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """
    Convert a plain-text password into a secure Argon2 hash.

    The original password is never stored.
    """

    if not isinstance(password, str) or not password:
        raise ValueError("Password cannot be empty.")

    return password_hasher.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    """
    Check whether a supplied password matches its stored Argon2 hash.

    Returns True when the password is correct.
    Returns False when it is incorrect or the stored hash is invalid.
    """

    if not isinstance(password, str):
        return False

    if not isinstance(hashed_password, str) or not hashed_password:
        return False

    try:
        return password_hasher.verify(
            password,
            hashed_password,
        )

    except Exception:
        return False


# Name used to store the CSRF token inside the session.
CSRF_SESSION_KEY = "_csrf_token"


def get_csrf_token(request: Request) -> str:
    """
    Retrieve the existing CSRF token or generate a new one.

    The token is stored in the user's session.
    """

    token = request.session.get(CSRF_SESSION_KEY)

    if not isinstance(token, str) or len(token) < 32:

        token = secrets.token_urlsafe(32)

        request.session[CSRF_SESSION_KEY] = token

    return token


def verify_csrf_token(
    request: Request,
    submitted_token: str,
) -> bool:
    """
    Verify a submitted CSRF token against the session token.
    """

    expected_token = request.session.get(CSRF_SESSION_KEY)

    if not isinstance(expected_token, str):
        return False

    if not isinstance(submitted_token, str):
        return False

    if not submitted_token:
        return False

    return secrets.compare_digest(
        expected_token,
        submitted_token,
    )


def rotate_csrf_token(request: Request) -> str:
    """
    Generate and store a fresh CSRF token.

    The old token becomes invalid.
    """

    token = secrets.token_urlsafe(32)

    request.session[CSRF_SESSION_KEY] = token

    return token
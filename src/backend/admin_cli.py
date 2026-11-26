"""Protected CLI for promoting an existing user to administrator."""

import argparse
import sys

from email_validator import EmailNotValidError, validate_email
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from src.backend.database import SessionLocal
from src.backend.models import User


def normalize_email(raw_email: str) -> str:
    """Validate and normalize an email address."""

    try:
        result = validate_email(
            raw_email.strip(),
            check_deliverability=False,
        )
        return result.normalized.casefold()

    except EmailNotValidError as exc:
        raise ValueError("Enter a valid email address.") from exc


def promote_user(email: str) -> int:
    """Promote an existing active user to administrator."""

    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(func.lower(User.email) == email)
            .one_or_none()
        )

        if user is None:
            print("ERROR: No registered user found with that email.")
            return 1

        if not user.is_active:
            print("ERROR: Cannot promote an inactive account.")
            return 1

        if user.is_admin:
            print("This account is already an administrator.")
            return 0

        user.is_admin = True
        db.commit()

        print("\nAdministrator promotion successful.")
        print(f"Name: {user.full_name}")
        print(f"Email: {user.email}")
        print(f"Administrator: {user.is_admin}")

        return 0

    except SQLAlchemyError:
        db.rollback()
        print(
            "ERROR: Database operation failed.",
            file=sys.stderr,
        )
        return 1

    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote an existing hospital user to administrator."
    )

    parser.add_argument(
        "--email",
        required=True,
        help="Email of the existing registered user.",
    )

    args = parser.parse_args()

    try:
        email = normalize_email(args.email)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nADMINISTRATOR PROVISIONING")
    print("--------------------------")
    print(f"Target email: {email}")
    print("This operation grants administrator privileges.")

    confirmation = input(
        "\nType PROMOTE to continue: "
    ).strip()

    if confirmation != "PROMOTE":
        print("Operation cancelled.")
        return 1

    return promote_user(email)


if __name__ == "__main__":
    raise SystemExit(main())
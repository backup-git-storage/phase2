"""SQLAlchemy database models."""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    func,
    text,
)

from src.backend.database import Base


class User(Base):
    """Application user associated with a hospital."""

    __tablename__ = "users"

    # Unique user ID.
    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # User's full name.
    full_name = Column(
        String(150),
        nullable=False,
    )

    # Hospital or healthcare organization.
    hospital_name = Column(
        String(255),
        nullable=False,
    )

    # Unique login email.
    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    # Secure password hash, never the original password.
    hashed_password = Column(
        String(255),
        nullable=False,
    )

    # False = hospital user.
    # True = administrator.
    is_admin = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    # Account activation status.
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )

    # Account registration timestamp.
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Records acceptance of platform terms.
    terms_accepted = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
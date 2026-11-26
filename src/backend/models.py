"""SQLAlchemy database models."""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    ForeignKey,
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

class AuditLog(Base):
    """Non-sensitive account and platform events for the admin activity feed."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    description = Column(String(500), nullable=False)
    actor_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

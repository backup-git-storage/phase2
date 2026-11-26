"""Best-effort audit event recording for administrator visibility."""

import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.backend.models import AuditLog

logger = logging.getLogger(__name__)


def record_audit_event(
    db: Session,
    event_type: str,
    description: str,
    actor_user_id: int | None = None,
) -> None:
    """Persist a non-sensitive audit event without breaking the main request."""
    try:
        db.add(
            AuditLog(
                event_type=event_type,
                description=description,
                actor_user_id=actor_user_id,
            )
        )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Could not persist audit event %s", event_type)

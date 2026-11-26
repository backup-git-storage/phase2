"""Database configuration and reusable SQLAlchemy sessions."""

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool


# Find the project root automatically.
# database.py is inside src/backend/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Load the project's .env file.
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# Read database URL.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. "
        "Configure it in the project's .env file."
    )


def build_engine(database_url: str) -> Engine:
    """Create an SQLAlchemy engine for PostgreSQL or SQLite."""

    try:
        url = make_url(database_url)

    except Exception as exc:
        raise ValueError(
            "DATABASE_URL is not a valid SQLAlchemy URL."
        ) from exc

    backend = url.get_backend_name()

    if (
        backend == "sqlite"
        and os.getenv("APP_ENV", "development").strip().lower()
        == "production"
    ):
        raise RuntimeError(
            "SQLite is not supported in production because local database "
            "files are not shared or durable on serverless deployments. "
            "Configure DATABASE_URL with a persistent PostgreSQL database."
        )

    engine_options = {
        "pool_pre_ping": True,
    }

    if backend == "sqlite":

        database_path = url.database

        # Handle in-memory SQLite databases.
        if database_path not in (None, "", ":memory:"):

            path = Path(database_path).expanduser()

            # Resolve relative paths from the project directory.
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()

            # Create the parent directory if needed.
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            url = url.set(database=str(path))

        engine_options["connect_args"] = {
            "check_same_thread": False
        }

        # Keep in-memory SQLite available across sessions.
        if url.database in (None, "", ":memory:"):
            engine_options["poolclass"] = StaticPool

    else:

        # PostgreSQL and Neon configuration.
        engine_options["pool_recycle"] = 300

    return create_engine(
        url,
        **engine_options,
    )


# Create engine without connecting immediately.
engine = build_engine(DATABASE_URL)


# Reusable database session factory.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


# Shared metadata for ORM models.
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Provide a database session and always close it."""

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
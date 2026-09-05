"""
Database backend selection for AI Tutor.

Thin, deliberately compatibility-oriented facade: it no longer parses ``.env``
itself (that used to duplicate config.py with *different* defaults for the same
path). Every value is read once in :mod:`config` and re-exposed here for callers
that were written against this module.

Prefer ``import db_manager`` for actual access, or ``config`` for raw settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import config
from config import DB_PATH, DB_TYPE


class DatabaseConfig:
    """Backend-agnostic database settings resolved from :mod:`config`."""

    def __init__(self, db_type: str | None = None, sqlite_path: str | None = None) -> None:
        self.DB_TYPE: str = (db_type or DB_TYPE).lower()
        self.SQLITE_DB_PATH: str = sqlite_path or DB_PATH

        # PostgreSQL settings are surfaced read-only for callers that inspect them.
        self.POSTGRES_HOST: str = config.POSTGRES_HOST
        self.POSTGRES_PORT: int = config.POSTGRES_PORT
        self.POSTGRES_DB: str = config.POSTGRES_DB
        self.POSTGRES_USER: str = config.POSTGRES_USER
        self.POSTGRES_POOL_SIZE: int = config.POSTGRES_POOL_SIZE
        self.POSTGRES_MAX_OVERFLOW: int = config.POSTGRES_MAX_OVERFLOW

    @property
    def postgres_password(self) -> str:
        """Postgres password, kept private so it is not printed in reprs."""
        return config.POSTGRES_PASSWORD

    def get_connection_string(self) -> str:
        """Libpq-style connection string for the active backend."""
        if self.is_postgresql():
            return (
                f"postgresql://{self.POSTGRES_USER}:{self.postgres_password}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return f"sqlite:///{Path(self.SQLITE_DB_PATH).resolve()}"

    def get_driver_name(self) -> Literal["sqlite", "postgresql"]:
        """Return the active driver name (validated once, in config.py)."""
        return "postgresql" if self.is_postgresql() else "sqlite"

    def is_sqlite(self) -> bool:
        return self.DB_TYPE == "sqlite"

    def is_postgresql(self) -> bool:
        return self.DB_TYPE == "postgresql"


# Global configuration instance (module-level, as before)
db_config = DatabaseConfig()

# Backwards-compatible constants.
DB_TYPE = db_config.DB_TYPE
DB_PATH = db_config.SQLITE_DB_PATH
CONNECTION_STRING = db_config.get_connection_string()


def get_db_type() -> str:
    """Get current database type."""
    return db_config.DB_TYPE


def is_postgresql() -> bool:
    """Check if PostgreSQL is configured."""
    return db_config.is_postgresql()


def is_sqlite() -> bool:
    """Check if SQLite is configured."""
    return db_config.is_sqlite()

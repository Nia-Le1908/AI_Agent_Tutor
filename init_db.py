"""
Database initialization utility for AI Tutor V5.1.

Responsibilities:
1. Read schema.sql.
2. Initialize the SQLite database at ``config.DB_PATH``.
3. Apply the schema idempotently (every statement is CREATE ... IF NOT EXISTS).
4. Verify the required tables exist.

Schema bootstrap is SQLite-specific: ``schema.sql`` uses PRAGMA, AUTOINCREMENT and
SQLite trigger syntax. When DB_TYPE=postgresql this module refuses to run rather
than quietly creating a SQLite file that the application will never read.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import config
from config import PROJECT_ROOT, ensure_runtime_dirs

REQUIRED_TABLES = frozenset({"users", "questions", "history", "sessions"})

SCHEMA_SQL_PATH = PROJECT_ROOT / "schema.sql"


class SchemaBootstrapError(RuntimeError):
    """Raised when the schema cannot be applied or verified."""


def read_schema_sql(schema_path: Path = SCHEMA_SQL_PATH) -> str:
    """Load the SQL schema text and validate that it is non-empty."""
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.sql not found: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8").strip()
    if not sql:
        raise ValueError(f"schema.sql is empty: {schema_path}")
    return sql


def verify_tables(conn: sqlite3.Connection, required: frozenset[str] = REQUIRED_TABLES) -> None:
    """
    Ensure every required table was created.

    Raises:
        SchemaBootstrapError: listing whichever tables are still missing.
    """
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    missing = required - {row[0] for row in rows}
    if missing:
        raise SchemaBootstrapError(
            "Database initialization incomplete. Missing tables: " + ", ".join(sorted(missing))
        )


def initialize_database() -> Path:
    """
    Create/upgrade the SQLite database from schema.sql.

    Returns:
        Absolute path of the initialized database.

    Raises:
        SchemaBootstrapError: when the configured backend is not SQLite.
        FileNotFoundError: when schema.sql is absent.
        sqlite3.Error: when the schema cannot be applied.
    """
    if config.DB_TYPE != "sqlite":
        raise SchemaBootstrapError(
            "init_db.py only bootstraps SQLite (schema.sql uses SQLite-only DDL). "
            "For PostgreSQL, apply an equivalent migration and skip this step."
        )

    ensure_runtime_dirs()
    schema_sql = read_schema_sql(SCHEMA_SQL_PATH)

    # Read lazily from config so a runtime override (tests, tools) is honoured
    # instead of being shadowed by an import-time snapshot.
    db_path = config.DB_PATH
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        verify_tables(conn)
        conn.commit()
    finally:
        conn.close()

    return Path(db_path).resolve()


def main() -> int:
    """CLI entry point with an explicit process exit code."""
    try:
        db_path = initialize_database()
    except Exception as exc:  # noqa: BLE001 - report any failure as a clean CLI error
        print(f"[ERROR] Failed to initialize database: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Database initialized at: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

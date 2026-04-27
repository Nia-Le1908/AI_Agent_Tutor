"""
Database initialization utility for AI Tutor V5.1.

Responsibilities:
1. Read schema.sql.
2. Initialize SQLite database at config.DB_PATH.
3. Apply schema in a safe, idempotent way.
4. Verify that required Phase 1 tables exist.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from config import DB_PATH


REQUIRED_TABLES = {"users", "questions", "history", "sessions"}


def _read_schema(schema_path: Path) -> str:
    """Load SQL schema text and validate that it is non-empty."""
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.sql not found: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8").strip()
    if not sql:
        raise ValueError(f"schema.sql is empty: {schema_path}")
    return sql


def _verify_tables(conn: sqlite3.Connection) -> None:
    """Ensure all required tables were created successfully."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    existing = {row[0] for row in cursor.fetchall()}

    missing = REQUIRED_TABLES - existing
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(f"Database initialization incomplete. Missing tables: {missing_list}")


def initialize_database() -> Path:
    """Initialize database from schema.sql and return database path."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_path = Path(__file__).resolve().parent / "schema.sql"
    schema_sql = _read_schema(schema_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        _verify_tables(conn)
        conn.commit()
    finally:
        conn.close()

    return db_path.resolve()


def main() -> int:
    """CLI entry point with explicit process exit code."""
    try:
        db_path = initialize_database()
    except Exception as exc:
        print(f"[ERROR] Failed to initialize database: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Database initialized at: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

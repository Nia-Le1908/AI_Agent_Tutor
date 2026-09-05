"""
Unified database access layer for AI Tutor (SQLite and PostgreSQL).

This module owns *how* the app talks to a database:

- connection creation / pooling / teardown,
- dialect differences (placeholder style, inserted-id retrieval),
- rows converted to plain dicts so callers never depend on driver row types.

It deliberately owns no business SQL: queries live in ``sqlite_manager.py`` (the
repository layer) and ``init_db.py`` (schema bootstrap). Callers use the small
driver-agnostic wrapper returned by :func:`connect`, or the convenience helpers
:func:`fetch_all`, :func:`fetch_one` and :func:`execute`.

Portable-SQL rules for anything written through this layer:
- use ``?`` placeholders (translated to ``%s`` for PostgreSQL);
- no ``INSERT OR IGNORE`` (use :meth:`DatabaseManager.insert_ignore`);
- no ``datetime('now')`` defaults in queries (pass a timestamp from Python);
- never rely on ``cursor.rowcount`` for single-row reads.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from collections.abc import Mapping as MappingABC

from config import (
    DB_TYPE,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_POOL_SIZE,
    POSTGRES_PORT,
    POSTGRES_USER,
    SQLITE_TIMEOUT_SECONDS,
)
from config_db import db_config

logger = logging.getLogger(__name__)

# Placeholder token recognised outside of quoted literals.
_PLACEHOLDER_RE = re.compile(r"\?")
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


class DatabaseError(RuntimeError):
    """Raised for database access failures that the app layer must report."""


def to_backend_sql(sql: str, *, backend: str = DB_TYPE) -> str:
    """
    Translate a ``?``-style query into the active backend's placeholder style.

    Quoted literals are skipped so a question text containing ``?`` is never
    mistaken for a placeholder.
    """
    if backend != "postgresql":
        return sql

    out: List[str] = []
    cursor = 0
    for match in _STRING_LITERAL_RE.finditer(sql):
        out.append(_PLACEHOLDER_RE.sub("%s", sql[cursor : match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(_PLACEHOLDER_RE.sub("%s", sql[cursor:]))
    return "".join(out)


class Result:
    """Driver-agnostic result of one statement: dict rows plus write metadata."""

    __slots__ = ("rows", "rowcount", "lastrowid")

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        rowcount: int = -1,
        lastrowid: Optional[int] = None,
    ) -> None:
        self.rows = rows
        self.rowcount = rowcount
        # Cursor-level autoincrement id; None on backends that do not provide it.
        self.lastrowid = lastrowid

    def fetchone(self) -> Optional[Dict[str, Any]]:
        """First row, or None when empty (mirrors sqlite3 semantics)."""
        return self.rows[0] if self.rows else None

    def fetchall(self) -> List[Dict[str, Any]]:
        return self.rows

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


class Connection:
    """
    Minimal, backend-independent connection wrapper used by the repository layer.

    Exposes just enough of the DB-API that calling code reads like plain SQL and
    works identically on SQLite and PostgreSQL.
    """

    def __init__(self, raw: Any, backend: str) -> None:
        self._raw = raw
        self.backend = backend

    @property
    def raw(self) -> Any:
        """Underlying driver connection (sqlite3.Connection / psycopg2)."""
        return self._raw

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Result:
        """Run one statement and return dict rows."""
        translated = to_backend_sql(sql, backend=self.backend)
        args = tuple(params or ())
        try:
            cursor = self._raw.cursor()
            try:
                cursor.execute(translated, args)
                rows = _rows_as_dicts(cursor) if cursor.description else []
                return Result(rows, cursor.rowcount, getattr(cursor, "lastrowid", None))
            finally:
                cursor.close()
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQL failed on sqlite: {exc}") from exc
        except Exception as exc:  # psycopg2.Error and driver-specific failures
            raise DatabaseError(f"SQL failed on {self.backend}: {exc}") from exc

    def executemany(self, sql: str, params_list: Iterable[Sequence[Any]]) -> int:
        """Run one statement per parameter tuple; returns affected row count."""
        translated = to_backend_sql(sql, backend=self.backend)
        batch = [tuple(params) for params in params_list]
        if not batch:
            return 0
        try:
            cursor = self._raw.cursor()
            try:
                cursor.executemany(translated, batch)
                return max(int(cursor.rowcount or 0), 0)
            finally:
                cursor.close()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Batch SQL failed on sqlite: {exc}") from exc
        except Exception as exc:
            raise DatabaseError(f"Batch SQL failed on {self.backend}: {exc}") from exc

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        # PostgreSQL connections come from a pool and must be returned, not closed.
        release = getattr(self._raw, "putconn", None)
        if callable(release):
            release()
        else:
            self._raw.close()


def _rows_as_dicts(cursor: Any) -> List[Dict[str, Any]]:
    """Materialize cursor rows as dicts, tolerating sqlite3.Row and tuples."""
    columns = [description[0] for description in (cursor.description or [])]
    if not columns:
        return []

    rows = cursor.fetchall()
    if rows and isinstance(rows[0], MappingABC):
        return [dict(row) for row in rows]
    return [dict(zip(columns, row)) for row in rows]


class DatabaseManager:
    """
    Factory + pool holder for database connections.

    SQLite gets a short-lived connection per operation (the historical behaviour,
    and the safest fit for Streamlit's re-run model). PostgreSQL uses a thread-safe
    connection pool.
    """

    def __init__(self, backend: str | None = None) -> None:
        self.backend = (backend or db_config.DB_TYPE).lower()
        self._pool = None
        self._pool_lock = threading.Lock()
        self._psycopg2 = None

        if self.backend not in ("sqlite", "postgresql"):
            raise DatabaseError(f"Unsupported DB_TYPE: {self.backend!r}")

        if self.backend == "postgresql":
            self._load_psycopg2()

        logger.info("Database manager configured for %s", self.backend)

    # -- backend setup ---------------------------------------------------
    def _load_psycopg2(self) -> None:
        try:
            import psycopg2
            from psycopg2 import pool  # noqa: F401  (validates the extras module exists)
        except ImportError as exc:
            raise DatabaseError(
                "DB_TYPE=postgresql requires psycopg2-binary. "
                "Install with: pip install psycopg2-binary"
            ) from exc
        self._psycopg2 = psycopg2

    def _postgres_pool(self):
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = self._psycopg2.pool.ThreadedConnectionPool(
                        minconn=1,
                        maxconn=max(POSTGRES_POOL_SIZE, 1),
                        host=POSTGRES_HOST,
                        port=POSTGRES_PORT,
                        dbname=POSTGRES_DB,
                        user=POSTGRES_USER,
                        password=POSTGRES_PASSWORD,
                    )
                    logger.info("PostgreSQL connection pool created")
        return self._pool

    # -- connections -----------------------------------------------------
    def open_connection(self) -> Connection:
        """
        Create a connection. Prefer :meth:`connect` (context managed) in app code.

        Raises:
            DatabaseError: when the backend cannot be reached at all.
        """
        if self.backend == "postgresql":
            return Connection(self._postgres_pool().getconn(), self.backend)
        return Connection(self._open_sqlite(), self.backend)

    def _open_sqlite(self) -> sqlite3.Connection:
        from config import DB_PATH, ensure_runtime_dirs

        try:
            ensure_runtime_dirs()
            conn = sqlite3.connect(
                DB_PATH, timeout=SQLITE_TIMEOUT_SECONDS, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"Cannot open SQLite database at {DB_PATH}: {exc}") from exc
        return conn

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        """
        Yield a connection, committing on success and rolling back on error.

        Callers may still call ``commit()`` explicitly; a second commit is a no-op.
        """
        conn = self.open_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:  # pragma: no cover - teardown must not mask errors
                logger.debug("Rollback failed while unwinding a database error", exc_info=True)
            raise
        finally:
            conn.close()

    # -- convenience helpers --------------------------------------------
    def fetch_all(self, sql: str, params: Sequence[Any] | None = None) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def fetch_one(self, sql: str, params: Sequence[Any] | None = None) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """Run a write statement and return the affected row count."""
        with self.connect() as conn:
            return conn.execute(sql, params).rowcount

    def execute_many(self, sql: str, params_list: Iterable[Sequence[Any]]) -> int:
        with self.connect() as conn:
            return conn.executemany(sql, params_list)

    def insert_returning_id(self, table: str, values: Mapping[str, Any]) -> int:
        """
        Insert one row and return its primary key.

        ``lastrowid`` is SQLite-only, so PostgreSQL uses ``RETURNING id`` instead.
        """
        columns = list(values)
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

        with self.connect() as conn:
            if conn.backend == "postgresql":
                row = conn.execute(f"{sql} RETURNING id", tuple(values[c] for c in columns)).fetchone()
                if not row or row.get("id") is None:
                    raise DatabaseError(f"Insert into {table} returned no id")
                return int(row["id"])

            result = conn.execute(sql, tuple(values[c] for c in columns))
            if result.lastrowid is None:
                raise DatabaseError(f"Insert into {table} returned no id")
            return int(result.lastrowid)

    def insert_ignore(self, table: str, values: Mapping[str, Any]) -> bool:
        """
        Insert unless the row already exists (``INSERT OR IGNORE`` equivalent).

        Implemented as a check-then-insert so it behaves the same on both backends
        instead of relying on SQLite-specific conflict syntax.
        """
        if not values:
            raise ValueError("values must not be empty")

        columns = list(values)
        where = " AND ".join(f"{column} = ?" for column in columns)
        with self.connect() as conn:
            exists = conn.execute(
                f"SELECT 1 AS found FROM {table} WHERE {where} LIMIT 1",
                tuple(values[c] for c in columns),
            ).fetchone()
            if exists:
                return False
            conn.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(values[c] for c in columns),
            )
        return True

    def health_check(self) -> bool:
        """True when a trivial query round-trips successfully."""
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1 AS ok")
            return True
        except (DatabaseError, sqlite3.Error, OSError) as exc:
            logger.error("Database health check failed: %s", exc)
            return False


_manager: Optional[DatabaseManager] = None
_manager_lock = threading.Lock()


def get_manager() -> DatabaseManager:
    """Return the process-wide :class:`DatabaseManager` (created on first use)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DatabaseManager()
    return _manager


def reset_manager_for_tests(manager: DatabaseManager | None = None) -> None:
    """Replace the cached manager (used by tests to point at a scratch DB)."""
    global _manager
    _manager = manager


# -- module-level convenience API (kept for backwards compatibility) --------
def get_db_connection() -> Connection:
    """
    Single-connection accessor.

    Prefer ``with db_manager.connect() as conn`` so commits/rollbacks and teardown
    are automatic; this helper is retained for scripts that already call it and
    manage the connection themselves.
    """
    return get_manager().open_connection()


def execute_query(
    query: str,
    params: Optional[Tuple[Any, ...]] = None,
    fetch: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Run ``query``, returning dict rows when ``fetch`` is True."""
    with get_manager().connect() as conn:
        result = conn.execute(query, params)
        return result.fetchall() if fetch else None


def fetch_all(query: str, params: Sequence[Any] | None = None) -> List[Dict[str, Any]]:
    return get_manager().fetch_all(query, params)


def fetch_one(query: str, params: Sequence[Any] | None = None) -> Optional[Dict[str, Any]]:
    return get_manager().fetch_one(query, params)


def execute(query: str, params: Sequence[Any] | None = None) -> int:
    return get_manager().execute(query, params)


def execute_many(query: str, params_list: Iterable[Sequence[Any]]) -> int:
    return get_manager().execute_many(query, params_list)


def insert_returning_id(table: str, values: Mapping[str, Any]) -> int:
    return get_manager().insert_returning_id(table, values)


def insert_ignore(table: str, values: Mapping[str, Any]) -> bool:
    return get_manager().insert_ignore(table, values)


def health_check() -> bool:
    """Check database connectivity."""
    return get_manager().health_check()

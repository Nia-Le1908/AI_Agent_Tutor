"""
Adaptive difficulty logic for AI Tutor V5.1 (Phase 3).

Specification implemented:
- get_next_difficulty(uid) -> int
- Rule 1: if user answers correctly 3 times in a row => level + 1
- Rule 2: if user answers incorrectly 2 times in a row => level - 1
- Always clamp level to [1, 5]

Data source:
- Reads `history` table ordered by most recent attempts.
- Persists updated level back to `users.level` to keep state consistent.
"""

from __future__ import annotations

import sqlite3
from typing import List

from config import DB_PATH


MIN_LEVEL = 1
MAX_LEVEL = 5


def _clamp_level(level: int) -> int:
    """Clamp integer level to the allowed [1, 5] range."""
    return max(MIN_LEVEL, min(MAX_LEVEL, level))


def _get_connection() -> sqlite3.Connection:
    """Create a database connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _fetch_user_level(conn: sqlite3.Connection, uid: int) -> int:
    """Read current user level; return default level 1 when user is not found."""
    row = conn.execute("SELECT level FROM users WHERE id = ?", (uid,)).fetchone()
    if row is None:
        return MIN_LEVEL
    return _clamp_level(int(row["level"]))


def _fetch_recent_outcomes(conn: sqlite3.Connection, uid: int, limit: int = 10) -> List[int]:
    """
    Fetch recent correctness flags as integers (1 correct / 0 incorrect).

    We order by timestamp DESC and rowid DESC to make ordering deterministic when
    multiple inserts share the same timestamp second.
    """
    rows = conn.execute(
        """
        SELECT is_correct
        FROM history
        WHERE uid = ?
        ORDER BY timestamp DESC, rowid DESC
        LIMIT ?
        """,
        (uid, limit),
    ).fetchall()

    return [int(row["is_correct"]) for row in rows]


def _count_consecutive(values: List[int], expected: int) -> int:
    """Count consecutive values from the start of the list equal to `expected`."""
    count = 0
    for value in values:
        if value == expected:
            count += 1
        else:
            break
    return count


def get_next_difficulty(uid: int) -> int:
    """
    Compute and persist next difficulty level for a user.

    Args:
        uid: User ID.

    Returns:
        int: next difficulty level in [1, 5].

    Raises:
        ValueError: if uid is invalid.
        sqlite3.Error: for database errors.
    """
    if not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid must be a positive integer")

    conn = _get_connection()
    try:
        current_level = _fetch_user_level(conn, uid)
        outcomes = _fetch_recent_outcomes(conn, uid)

        # No history yet: keep current level (or default 1).
        if not outcomes:
            next_level = current_level
        else:
            correct_streak = _count_consecutive(outcomes, expected=1)
            wrong_streak = _count_consecutive(outcomes, expected=0)

            if correct_streak >= 3:
                next_level = current_level + 1
            elif wrong_streak >= 2:
                next_level = current_level - 1
            else:
                next_level = current_level

        next_level = _clamp_level(next_level)

        # Persist for future calls and UI consistency.
        conn.execute("UPDATE users SET level = ? WHERE id = ?", (next_level, uid))
        conn.commit()

        return next_level
    finally:
        conn.close()

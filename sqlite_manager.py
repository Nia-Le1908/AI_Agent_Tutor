"""
SQLite data access helpers for AI Tutor V5.1 (Phase 1).

Required interfaces from spec:
- save_history(uid, qid, is_correct)
- get_question_by_diff(level) -> List[dict]
- get_all_subjects() -> List[str]
- get_questions_filtered(level, subject, exclude_uid) -> List[dict]
- get_weak_topics(uid) -> dict

This module keeps SQL concentrated in one place to simplify maintenance.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List

from config import DB_PATH

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection configured for dictionary-like row access.

    Returning sqlite3.Row helps convert query outputs to plain dict objects.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def save_history(uid: int, qid: int, is_correct: bool) -> None:
    """
    Insert one answer event into history.

    Args:
        uid: User ID.
        qid: Question ID.
        is_correct: True if user's answer is correct.

    Raises:
        ValueError: on invalid argument types/values.
        sqlite3.Error: on database constraint or connectivity errors.
    """
    if not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid must be a positive integer")
    if not isinstance(qid, int) or qid <= 0:
        raise ValueError("qid must be a positive integer")
    if not isinstance(is_correct, bool):
        raise ValueError("is_correct must be a boolean")

    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO history (uid, qid, is_correct) VALUES (?, ?, ?)",
            (uid, qid, 1 if is_correct else 0),
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        logger.error("Lỗi SQLite: %s", e)
    finally:
        conn.close()


def get_question_by_diff(level: int) -> List[Dict[str, Any]]:
    """
    Fetch all questions matching the requested difficulty level.

    Args:
        level: Difficulty in [1, 5].

    Returns:
        List[dict] where each dict is one question row.
    """
    if not isinstance(level, int) or level < 1 or level > 5:
        raise ValueError("level must be an integer in range [1, 5]")

    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, content, difficulty, subject, options, answer, explanation
            FROM questions
            WHERE difficulty = ?
            ORDER BY id ASC
            """,
            (level,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_all_subjects() -> List[str]:
    """Fetch all distinct subjects from the questions table."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT subject FROM questions ORDER BY subject ASC"
        ).fetchall()
        return [row["subject"] for row in rows]
    finally:
        conn.close()


def get_questions_filtered(
    level: int,
    subject: str | None = None,
    exclude_uid: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Fetch questions with optional subject filter and answered-question exclusion.

    Args:
        level: Difficulty in [1, 5].
        subject: If provided, only return questions matching this subject.
        exclude_uid: If provided, exclude questions this user already answered correctly.
    """
    if not isinstance(level, int) or level < 1 or level > 5:
        raise ValueError("level must be an integer in range [1, 5]")

    conn = _get_connection()
    try:
        query = """
            SELECT id, content, difficulty, subject, options, answer, explanation
            FROM questions
            WHERE difficulty = ?
        """
        params: list = [level]

        if subject:
            query += " AND subject = ?"
            params.append(subject)

        if exclude_uid and isinstance(exclude_uid, int) and exclude_uid > 0:
            query += """
                AND id NOT IN (
                    SELECT DISTINCT qid FROM history
                    WHERE uid = ? AND is_correct = 1
                )
            """
            params.append(exclude_uid)

        query += " ORDER BY id ASC"

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_user_stats(uid: int) -> Dict[str, Any]:
    """
    Fetch summary statistics for a user's learning session.

    Returns dict with: total_attempted, total_correct, accuracy, current_streak
    """
    if not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid must be a positive integer")

    conn = _get_connection()
    try:
        # Overall stats
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct
            FROM history WHERE uid = ?
            """,
            (uid,),
        ).fetchone()

        total = int(row["total"] or 0)
        correct = int(row["correct"] or 0)
        accuracy = (correct / total * 100) if total > 0 else 0.0

        # Current streak (consecutive correct from most recent)
        streak_rows = conn.execute(
            """
            SELECT is_correct FROM history
            WHERE uid = ?
            ORDER BY timestamp DESC, rowid DESC
            LIMIT 20
            """,
            (uid,),
        ).fetchall()

        streak = 0
        for sr in streak_rows:
            if int(sr["is_correct"]) == 1:
                streak += 1
            else:
                break

        return {
            "total_attempted": total,
            "total_correct": correct,
            "accuracy": accuracy,
            "current_streak": streak,
        }
    finally:
        conn.close()


def get_weak_topics(uid: int) -> Dict[str, Dict[str, float]]:
    """
    Compute per-subject correctness statistics for a user.

    Returns a dictionary keyed by subject. Each value contains:
    - correct: number of correct answers
    - incorrect: number of incorrect answers
    - total: total attempts
    - accuracy: correct / total

    This structure is UI-friendly and can directly feed charts.
    """
    if not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid must be a positive integer")

    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT
                q.subject AS subject,
                SUM(CASE WHEN h.is_correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN h.is_correct = 0 THEN 1 ELSE 0 END) AS incorrect_count,
                COUNT(*) AS total_count
            FROM history h
            INNER JOIN questions q ON q.id = h.qid
            WHERE h.uid = ?
            GROUP BY q.subject
            ORDER BY q.subject ASC
            """,
            (uid,),
        )

        results: Dict[str, Dict[str, float]] = {}
        for row in cursor.fetchall():
            subject = row["subject"]
            correct = float(row["correct_count"] or 0)
            incorrect = float(row["incorrect_count"] or 0)
            total = float(row["total_count"] or 0)
            accuracy = (correct / total) if total > 0 else 0.0

            results[subject] = {
                "correct": correct,
                "incorrect": incorrect,
                "total": total,
                "accuracy": accuracy,
            }

        return results
    finally:
        conn.close()

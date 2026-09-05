"""
Repository layer for AI Tutor: all business SQL lives here.

Required interfaces from the spec:
- save_history(uid, qid, is_correct)
- get_question_by_diff(level) -> List[dict]
- get_all_subjects() -> List[str]
- get_questions_filtered(level, subject, exclude_uid) -> List[dict]
- get_weak_topics(uid) -> dict

The module keeps its historic name (it is referenced by the docs and the UI), but
it is backend-agnostic: connections, dialect handling and row conversion come from
:mod:`db_manager`, so the same queries run on SQLite and PostgreSQL.

The module docstring's original promise holds: SQL is concentrated here so the UI
(app.py / dashboard.py) and the adaptive logic never hand-write queries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import db_manager
from schemas import options_to_json
from validation import require_bool, require_level, require_non_empty_str, require_positive_int

logger = logging.getLogger(__name__)

# Columns the UI expects on a question dict; single definition avoids drift
# between the two question queries below.
QUESTION_COLUMNS = "id, content, difficulty, subject, options, answer, explanation"

MAX_STREAK_HISTORY = 20


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def get_user_level(uid: int) -> Optional[int]:
    """Return a user's stored difficulty level, or None when the user is unknown."""
    require_positive_int(uid, "uid")
    row = db_manager.fetch_one("SELECT level FROM users WHERE id = ?", (uid,))
    return None if row is None or row.get("level") is None else int(row["level"])


def set_user_level(uid: int, level: int) -> None:
    """Persist a user's adaptive level."""
    require_positive_int(uid, "uid")
    require_level(level, "level")
    db_manager.execute("UPDATE users SET level = ? WHERE id = ?", (level, uid))


def get_or_create_user(name: str) -> tuple[int, int]:
    """
    Resolve a user by exact (trimmed) name, creating them with level 1 if needed.

    Low-friction onboarding is intentional for demo/prototype usage.

    Returns:
        (user_id, current_level)
    """
    cleaned = require_non_empty_str(name, "name", max_length=100)

    row = db_manager.fetch_one(
        "SELECT id, level FROM users WHERE name = ? ORDER BY id ASC LIMIT 1",
        (cleaned,),
    )
    if row is not None:
        return int(row["id"]), int(row["level"])

    user_id = db_manager.insert_returning_id("users", {"name": cleaned, "level": 1})
    return user_id, 1


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def get_question_by_diff(level: int) -> List[Dict[str, Any]]:
    """
    Fetch all questions matching the requested difficulty level.

    Args:
        level: Difficulty in [1, 5].

    Returns:
        List[dict] where each dict is one question row.
    """
    require_level(level, "level")
    return db_manager.fetch_all(
        f"""
        SELECT {QUESTION_COLUMNS}
        FROM questions
        WHERE difficulty = ?
        ORDER BY id ASC
        """,
        (level,),
    )


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
    require_level(level, "level")

    query = f"SELECT {QUESTION_COLUMNS} FROM questions WHERE difficulty = ?"
    params: List[Any] = [level]

    if subject:
        query += " AND subject = ?"
        params.append(require_non_empty_str(subject, "subject"))

    if exclude_uid is not None:
        require_positive_int(exclude_uid, "exclude_uid")
        query += """
            AND id NOT IN (
                SELECT DISTINCT qid FROM history
                WHERE uid = ? AND is_correct = 1
            )
        """
        params.append(exclude_uid)

    query += " ORDER BY id ASC"
    return db_manager.fetch_all(query, params)


def get_all_subjects() -> List[str]:
    """Fetch all distinct subjects from the questions table."""
    rows = db_manager.fetch_all("SELECT DISTINCT subject FROM questions ORDER BY subject ASC")
    return [row["subject"] for row in rows if row.get("subject")]


def insert_question(question: Dict[str, Any]) -> int:
    """
    Insert one generated question dict (schema.json shape) and return its row id.

    Options are normalized into the canonical ``{"A": ..., "D": ...}`` JSON object
    the UI expects, so every writer produces the same shape.
    """
    if not isinstance(question, dict):
        raise ValueError("question must be a dict")

    content = require_non_empty_str(question.get("content"), "content")
    subject = require_non_empty_str(question.get("subject"), "subject")
    difficulty = require_level(question.get("difficulty"), "difficulty")
    answer = require_non_empty_str(question.get("answer"), "answer").upper()
    options_json = options_to_json(question.get("options"))

    return db_manager.insert_returning_id(
        "questions",
        {
            "content": content,
            "difficulty": difficulty,
            "subject": subject,
            "options": options_json,
            "answer": answer,
            "explanation": str(question.get("explanation") or "").strip(),
        },
    )


def insert_questions(questions: List[Dict[str, Any]]) -> int:
    """
    Insert many generated questions, skipping ones that fail validation.

    Returns:
        Number of questions actually saved.
    """
    saved = 0
    for question in questions:
        try:
            insert_question(question)
            saved += 1
        except (ValueError, db_manager.DatabaseError) as exc:
            # A single malformed generation must not abort the whole batch.
            logger.warning("Skipped invalid generated question: %s", exc)
    return saved


# ---------------------------------------------------------------------------
# Answer history
# ---------------------------------------------------------------------------
def save_history(uid: int, qid: int, is_correct: bool) -> None:
    """
    Insert one answer event into history.

    Args:
        uid: User ID.
        qid: Question ID.
        is_correct: True if user's answer was correct.

    Raises:
        ValueError: on invalid argument values.
        DatabaseError: on database constraint or connectivity errors.
    """
    require_positive_int(uid, "uid")
    require_positive_int(qid, "qid")
    require_bool(is_correct, "is_correct")

    db_manager.execute(
        "INSERT INTO history (uid, qid, is_correct) VALUES (?, ?, ?)",
        (uid, qid, 1 if is_correct else 0),
    )


def fetch_recent_outcomes(uid: int, limit: int = MAX_STREAK_HISTORY) -> List[int]:
    """
    Fetch recent correctness flags as ints (1 correct / 0 incorrect), newest first.

    Ordering falls back to the autoincrement primary key so ties inside the same
    timestamp second stay deterministic on every backend.
    """
    require_positive_int(uid, "uid")
    require_positive_int(limit, "limit")

    rows = db_manager.fetch_all(
        """
        SELECT is_correct FROM history
        WHERE uid = ?
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (uid, limit),
    )
    return [int(row["is_correct"]) for row in rows]


def count_consecutive(values: List[int], expected: int) -> int:
    """Count how many values at the start of the list equal ``expected``."""
    count = 0
    for value in values:
        if value != expected:
            break
        count += 1
    return count


def count_streak(outcomes: List[int], expected: int = 1) -> int:
    """Current streak of ``expected`` results, newest-first, without a DB round-trip."""
    return count_consecutive(list(outcomes), expected)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def get_user_stats(uid: int) -> Dict[str, Any]:
    """
    Summary statistics for a user's learning history.

    Returns dict with: total_attempted, total_correct, accuracy, current_streak
    """
    require_positive_int(uid, "uid")

    row = db_manager.fetch_one(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct
        FROM history WHERE uid = ?
        """,
        (uid,),
    ) or {}

    total = int(row.get("total") or 0)
    correct = int(row.get("correct") or 0)

    return {
        "total_attempted": total,
        "total_correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0.0,
        "current_streak": count_streak(fetch_recent_outcomes(uid)),
    }


def get_weak_topics(uid: int) -> Dict[str, Dict[str, float]]:
    """
    Compute per-subject correctness statistics for a user.

    Returns a dict keyed by subject; each value holds:
    correct, incorrect, total (floats) and accuracy in [0.0, 1.0].

    This structure is UI-friendly and feeds both the pie and radar charts, so the
    dashboard renders from a single query instead of repeating it.
    """
    require_positive_int(uid, "uid")

    rows = db_manager.fetch_all(
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
    for row in rows:
        correct = float(row.get("correct_count") or 0)
        incorrect = float(row.get("incorrect_count") or 0)
        total = float(row.get("total_count") or 0)
        results[row["subject"]] = {
            "correct": correct,
            "incorrect": incorrect,
            "total": total,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }
    return results


def get_progress_timeline(uid: int) -> List[Dict[str, Any]]:
    """Answer events ordered oldest-first, for the progress-over-time chart."""
    require_positive_int(uid, "uid")
    return db_manager.fetch_all(
        """
        SELECT timestamp AS ts, is_correct AS is_correct
        FROM history
        WHERE uid = ?
        ORDER BY timestamp ASC, id ASC
        """,
        (uid,),
    )


def get_difficulty_scores(uid: int) -> List[Dict[str, Any]]:
    """Average correctness per difficulty level, for the score-by-difficulty chart."""
    require_positive_int(uid, "uid")
    return db_manager.fetch_all(
        """
        SELECT
            q.difficulty AS difficulty,
            AVG(CASE WHEN h.is_correct = 1 THEN 1.0 ELSE 0.0 END) AS avg_score,
            COUNT(*) AS total_attempts
        FROM history h
        INNER JOIN questions q ON q.id = h.qid
        WHERE h.uid = ?
        GROUP BY q.difficulty
        ORDER BY q.difficulty ASC
        """,
        (uid,),
    )

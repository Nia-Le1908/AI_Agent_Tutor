"""
Adaptive difficulty logic for AI Tutor.

Rules:
- 3 correct answers in a row  => level + 1
- 2 incorrect in a row         => level - 1
- level always clamped to [1, 5]

Design: the decision is a pure function (:func:`compute_next_level`) so the rule
set is unit-testable without a database, while :func:`get_next_difficulty` keeps
the spec-mandated signature and handles loading/persisting state through the
repository layer.
"""

from __future__ import annotations

from typing import List

from sqlite_manager import count_consecutive, fetch_recent_outcomes, get_user_level, set_user_level
from validation import MAX_LEVEL, MIN_LEVEL, clamp, require_positive_int

__all__ = [
    "MIN_LEVEL",
    "MAX_LEVEL",
    "CORRECT_STREAK_TO_PROMOTE",
    "WRONG_STREAK_TO_DEMOTE",
    "compute_next_level",
    "get_next_difficulty",
]

# Streak thresholds from the specification.
CORRECT_STREAK_TO_PROMOTE = 3
WRONG_STREAK_TO_DEMOTE = 2

# How many recent attempts are inspected when measuring streaks.
RECENT_ATTEMPTS_WINDOW = 10


def _clamp_level(level: int) -> int:
    """Clamp an integer level to the allowed [1, 5] range."""
    return clamp(level, MIN_LEVEL, MAX_LEVEL)


def compute_next_level(
    current_level: int,
    outcomes: List[int],
    *,
    promote_after: int = CORRECT_STREAK_TO_PROMOTE,
    demote_after: int = WRONG_STREAK_TO_DEMOTE,
) -> int:
    """
    Pure difficulty decision.

    Args:
        current_level: Level the user is on now (caller is responsible for clamping
            stored values; this function clamps its result anyway).
        outcomes: Recent results, newest first, as 1 (correct) / 0 (incorrect).
        promote_after: Consecutive correct answers needed to level up.
        demote_after: Consecutive incorrect answers needed to level down.

    Returns:
        Next level, clamped to [MIN_LEVEL, MAX_LEVEL].
    """
    if not outcomes:
        return _clamp_level(current_level)

    if count_consecutive(outcomes, 1) >= promote_after:
        return _clamp_level(current_level + 1)
    if count_consecutive(outcomes, 0) >= demote_after:
        return _clamp_level(current_level - 1)
    return _clamp_level(current_level)


def get_next_difficulty(uid: int) -> int:
    """
    Compute and persist the next difficulty level for a user.

    Args:
        uid: User ID.

    Returns:
        int: next difficulty level in [1, 5]. Unknown users fall back to MIN_LEVEL.

    Raises:
        ValueError: if uid is invalid.
        db_manager.DatabaseError: for database errors.
    """
    require_positive_int(uid, "uid")

    # A missing user keeps the historical "treat as level 1" behaviour.
    current_level = _clamp_level(get_user_level(uid) or MIN_LEVEL)
    outcomes = fetch_recent_outcomes(uid, limit=RECENT_ATTEMPTS_WINDOW)
    next_level = compute_next_level(current_level, outcomes)

    if next_level != current_level:
        set_user_level(uid, next_level)

    return next_level

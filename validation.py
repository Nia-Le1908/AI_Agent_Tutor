"""
Shared argument validation helpers for AI Tutor.

Every layer of this project (UI, orchestration, data access, generation) used to
carry its own copy of checks like "uid must be a positive integer" or
"level must be in [1, 5]". Keeping them here gives one error message style, one
place to change rules, and makes the ranges themselves part of the contract.
"""

from __future__ import annotations

MIN_LEVEL = 1
MAX_LEVEL = 5

OPTION_KEYS: tuple[str, str, str, str] = ("A", "B", "C", "D")


def clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer into the inclusive [low, high] range."""
    return max(low, min(high, value))


def require_int_in_range(value: object, name: str, low: int, high: int) -> int:
    """
    Return ``value`` as an int inside [low, high].

    Raises:
        ValueError: if value is a bool, not an int, or outside the range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in range [{low}, {high}], got {value!r}")
    if value < low or value > high:
        raise ValueError(f"{name} must be an integer in range [{low}, {high}], got {value}")
    return int(value)


def require_positive_int(value: object, name: str) -> int:
    """
    Return ``value`` as a positive int.

    Raises:
        ValueError: if value is a bool, not an int, or <= 0.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return int(value)


def require_level(value: object, name: str = "level") -> int:
    """Validate a difficulty level against the project range [1, 5]."""
    return require_int_in_range(value, name, MIN_LEVEL, MAX_LEVEL)


def require_non_empty_str(value: object, name: str, *, max_length: int | None = None) -> str:
    """
    Return ``value`` stripped and guaranteed non-empty.

    Raises:
        ValueError: if value is not a string, blank, or longer than max_length.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must be a non-empty string")
    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters, got {len(cleaned)}")
    return cleaned


def require_bool(value: object, name: str) -> bool:
    """Return ``value`` as a bool, rejecting ints (so 1/0 typos are visible)."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value

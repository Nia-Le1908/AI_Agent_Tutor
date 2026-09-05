"""Tests for shared validation rules (validation.py)."""

from __future__ import annotations

import pytest

from validation import (
    MAX_LEVEL,
    MIN_LEVEL,
    clamp,
    require_bool,
    require_int_in_range,
    require_level,
    require_non_empty_str,
    require_positive_int,
)


class TestRequirePositiveInt:
    @pytest.mark.parametrize("value", [1, 5, 999])
    def test_accepts_positive_values(self, value):
        assert require_positive_int(value, "uid") == value

    @pytest.mark.parametrize("value", [0, -1, None, "5", 1.5, True, False])
    def test_rejects_non_positive_integers(self, value):
        with pytest.raises(ValueError, match="uid must be a positive integer"):
            require_positive_int(value, "uid")


class TestRequireLevel:
    @pytest.mark.parametrize("value", [MIN_LEVEL, 3, MAX_LEVEL])
    def test_accepts_in_range_levels(self, value):
        assert require_level(value) == value

    @pytest.mark.parametrize("value", [0, -3, 6, 100, "2", None, 1.0])
    def test_rejects_out_of_range_levels(self, value):
        with pytest.raises(ValueError, match=r"level must be an integer in range \[1, 5\]"):
            require_level(value)

    def test_custom_name_appears_in_message(self):
        with pytest.raises(ValueError, match="difficulty"):
            require_level(9, "difficulty")

    def test_bool_is_rejected_even_though_bool_is_int(self):
        # ``True`` would otherwise silently pass as level 1.
        with pytest.raises(ValueError):
            require_level(True)


class TestRequireIntInRange:
    def test_bounds_are_inclusive(self):
        assert require_int_in_range(10, "count", 1, 10) == 10

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="count must be an integer in range \\[1, 10\\]"):
            require_int_in_range(11, "count", 1, 10)


class TestRequireNonEmptyStr:
    def test_strips_whitespace(self):
        assert require_non_empty_str("  BFS  ", "topic") == "BFS"

    @pytest.mark.parametrize("value", ["", "   ", "\n\t", None, 42])
    def test_rejects_blank_and_non_string(self, value):
        with pytest.raises(ValueError, match="topic"):
            require_non_empty_str(value, "topic")

    def test_max_length_enforced(self):
        with pytest.raises(ValueError, match="at most 3 characters"):
            require_non_empty_str("abcd", "name", max_length=3)


class TestMisc:
    def test_clamp(self):
        assert clamp(0, MIN_LEVEL, MAX_LEVEL) == 1
        assert clamp(9, MIN_LEVEL, MAX_LEVEL) == 5
        assert clamp(3, MIN_LEVEL, MAX_LEVEL) == 3

    def test_require_bool_rejects_ints(self):
        assert require_bool(True, "is_correct") is True
        with pytest.raises(ValueError, match="is_correct must be a boolean"):
            require_bool(1, "is_correct")

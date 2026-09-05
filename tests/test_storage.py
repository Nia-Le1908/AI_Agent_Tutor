"""
Tests for the storage layer: db_manager, repository functions, and the adaptive rules.

These double as the behavioural specification that existed only implicitly before:
the streak thresholds, level clamping, exclusion of correctly-answered questions and
the exact statistics the dashboard renders.
"""

from __future__ import annotations

import sqlite3

import pytest

import db_manager
import sqlite_manager
from adaptive_logic import MAX_LEVEL, MIN_LEVEL, compute_next_level, get_next_difficulty
from helpers import db_rows, insert_history, question_payload


# ---------------------------------------------------------------------------
# db_manager
# ---------------------------------------------------------------------------
class TestDatabaseManager:
    def test_health_check_against_sqlite(self, db):
        assert db_manager.get_manager().health_check() is True

    def test_unreachable_database_is_reported_not_raised(self, tmp_path, monkeypatch):
        # A path that cannot hold a database must degrade to health_check() -> False
        # instead of escaping as a driver error (the UI treats this as "not ready").
        import config

        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(config, "DB_PATH", str(blocker / "nope.db"))
        assert db_manager.get_manager().health_check() is False

    def test_connect_commits_implicitly_on_success(self, db):
        with db_manager.get_manager().connect() as conn:
            conn.execute("INSERT INTO users (name, level) VALUES (?, ?)", ("carol", 2))

        assert db_rows(db, "SELECT name FROM users WHERE name = 'carol'") == [("carol",)]

    def test_connect_rolls_back_on_error(self, db):
        with pytest.raises(RuntimeError):
            with db_manager.get_manager().connect() as conn:
                conn.execute("INSERT INTO users (name, level) VALUES (?, ?)", ("dave", 3))
                raise RuntimeError("boom")

        assert db_rows(db, "SELECT name FROM users WHERE name = 'dave'") == []

    def test_query_errors_are_wrapped_as_database_error(self, db):
        with pytest.raises(db_manager.DatabaseError, match="no_such_table"):
            db_manager.fetch_all("SELECT * FROM no_such_table")

    def test_rows_are_plain_dicts(self, db):
        db_manager.execute("INSERT INTO users (id, name, level) VALUES (10, 'Erin', 4)")
        row = db_manager.fetch_one("SELECT * FROM users WHERE id = 10")
        assert isinstance(row, dict)
        assert row["name"] == "Erin" and row["level"] == 4

    def test_insert_returning_id(self, db):
        user_id = db_manager.insert_returning_id("users", {"name": "Frank", "level": 1})
        assert user_id > 0
        assert db_rows(db, "SELECT level FROM users WHERE id = ?", (user_id,)) == [(1,)]

    def test_execute_many_returns_count(self, db):
        affected = db_manager.execute_many(
            "INSERT INTO users (name, level) VALUES (?, ?)",
            [("g1", 1), ("g2", 2)],
        )
        assert affected == 2

    def test_execute_many_with_empty_list_is_a_noop(self, db):
        assert db_manager.execute_many("INSERT INTO users (name) VALUES (?)", []) == 0

    def test_postgres_translation_only_outside_literals(self):
        sql = "SELECT * FROM q WHERE a = ? AND b <> '?' AND c = ?"
        assert (
            db_manager.to_backend_sql(sql, backend="postgresql")
            == "SELECT * FROM q WHERE a = %s AND b <> '?' AND c = %s"
        )

    def test_sqlite_sql_is_untouched(self, db):
        assert db_manager.to_backend_sql("SELECT ? AS x", backend="sqlite") == "SELECT ? AS x"

    def test_unsupported_backend_rejected(self):
        with pytest.raises(db_manager.DatabaseError, match="Unsupported DB_TYPE"):
            db_manager.DatabaseManager(backend="mysql")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
class TestUsers:
    def test_get_or_create_user_creates_and_returns_level(self, db):
        uid, level = sqlite_manager.get_or_create_user("Alice")
        assert level == 1
        assert db_rows(db, "SELECT name, level FROM users WHERE id = ?", (uid,)) == [("Alice", 1)]

    def test_get_or_create_user_reuses_existing_name(self, db):
        first, _ = sqlite_manager.get_or_create_user("Alice")
        db_manager.execute("UPDATE users SET level = 4 WHERE id = ?", (first,))

        second, level = sqlite_manager.get_or_create_user("  Alice  ")
        assert (second, level) == (first, 4)
        assert db_rows(db, "SELECT COUNT(*) FROM users")[0][0] == 1

    def test_get_or_create_user_rejects_blank_name(self, db):
        with pytest.raises(ValueError, match="name"):
            sqlite_manager.get_or_create_user("   ")

    def test_get_user_level_roundtrip(self, seeded_db):
        assert sqlite_manager.get_user_level(1) == 1
        sqlite_manager.set_user_level(1, 3)
        assert sqlite_manager.get_user_level(1) == 3

    def test_get_user_level_unknown_user_is_none(self, seeded_db):
        assert sqlite_manager.get_user_level(999) is None

    def test_set_user_level_validates_range(self, seeded_db):
        with pytest.raises(ValueError, match=r"level must be an integer in range \[1, 5\]"):
            sqlite_manager.set_user_level(1, 6)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
class TestQuestions:
    def test_get_question_by_diff_filters_and_orders(self, seeded_db):
        questions = sqlite_manager.get_question_by_diff(1)
        assert [q["id"] for q in questions] == [1, 2, 4]
        assert all(q["difficulty"] == 1 for q in questions)

        # The malformed-options question lives at level 5 and is still returned:
        # filtering by shape is a presentation concern, not a query concern.
        assert [q["id"] for q in sqlite_manager.get_question_by_diff(5)] == [6]
        assert set(questions[0]) == {
            "id",
            "content",
            "difficulty",
            "subject",
            "options",
            "answer",
            "explanation",
        }

    @pytest.mark.parametrize("level", [0, 6, "1", None, True])
    def test_get_question_by_diff_validates_level(self, seeded_db, level):
        with pytest.raises(ValueError, match=r"\[1, 5\]"):
            sqlite_manager.get_question_by_diff(level)

    def test_subject_filter(self, seeded_db):
        assert [q["id"] for q in sqlite_manager.get_questions_filtered(1, subject="Science")] == [4]

    def test_exclude_uid_removes_correctly_answered_only(self, seeded_db):
        all_level1 = [q["id"] for q in sqlite_manager.get_questions_filtered(1)]
        skipping = [q["id"] for q in sqlite_manager.get_questions_filtered(1, exclude_uid=1)]

        # Alice answered 1 and 2 correctly but got 4 wrong, so only 4 must survive.
        assert all_level1 == [1, 2, 4]
        assert skipping == [4]

    def test_get_all_subjects_sorted_distinct(self, seeded_db):
        assert sqlite_manager.get_all_subjects() == ["Math", "Science"]

    def test_insert_question_normalizes_options(self, db):
        row_id = sqlite_manager.insert_question(question_payload(options=["a", "b", "c", "d"]))
        stored = sqlite_manager.get_question_by_diff(1)

        assert stored[0]["id"] == row_id
        assert sqlite3.connect(db).execute(
            "SELECT options FROM questions WHERE id = ?", (row_id,)
        ).fetchone()[0] == '{"A": "a", "B": "b", "C": "c", "D": "d"}'

    def test_insert_question_rejects_garbage(self, db):
        with pytest.raises(ValueError):
            sqlite_manager.insert_question({"content": "x", "difficulty": 9, "subject": "y"})

    def test_insert_questions_skips_invalid_rows(self, db, caplog):
        good = question_payload(content="valid")
        bad = question_payload(content="bad", options=["only", "three"])

        assert sqlite_manager.insert_questions([good, bad, good]) == 2
        assert "invalid" in caplog.text.lower()

    def test_insert_questions_empty_list(self, db):
        assert sqlite_manager.insert_questions([]) == 0


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
class TestHistory:
    def test_save_history_stores_boolean_as_int(self, seeded_db):
        sqlite_manager.save_history(uid=2, qid=1, is_correct=True)
        assert db_rows(seeded_db, "SELECT is_correct FROM history WHERE uid = 2") == [(1,)]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"uid": 0, "qid": 1, "is_correct": True},
            {"uid": None, "qid": 1, "is_correct": True},
            {"uid": 1, "qid": 0, "is_correct": True},
            {"uid": 1, "qid": 1, "is_correct": "yes"},
            {"uid": 1, "qid": 1, "is_correct": 1},
        ],
    )
    def test_save_history_validates_arguments(self, seeded_db, kwargs):
        with pytest.raises(ValueError):
            sqlite_manager.save_history(**kwargs)

    def test_save_history_enforces_foreign_keys(self, seeded_db):
        with pytest.raises(db_manager.DatabaseError):
            sqlite_manager.save_history(uid=1, qid=9999, is_correct=True)

    def test_fetch_recent_outcomes_newest_first(self, seeded_db):
        assert sqlite_manager.fetch_recent_outcomes(1) == [0, 0, 1, 1]

    def test_fetch_recent_outcomes_respects_limit(self, seeded_db):
        assert sqlite_manager.fetch_recent_outcomes(1, limit=2) == [0, 0]

    def test_fetch_recent_outcomes_empty_for_new_user(self, seeded_db):
        assert sqlite_manager.fetch_recent_outcomes(2) == []

    def test_consecutive_helpers(self):
        assert sqlite_manager.count_consecutive([1, 1, 0, 1], 1) == 2
        assert sqlite_manager.count_consecutive([0, 1], 1) == 0
        assert sqlite_manager.count_streak([]) == 0


# ---------------------------------------------------------------------------
# Adaptive difficulty
# ---------------------------------------------------------------------------
class TestAdaptiveLogic:
    @pytest.mark.parametrize(
        "current, outcomes, expected",
        [
            (1, [], 1),
            (1, [1, 1], 1),                       # two correct: not enough
            (1, [1, 1, 1], 2),                    # three correct: promote
            (2, [1, 1, 1, 0], 3),                 # streak measured from most recent
            (3, [0, 0], 2),                       # two wrong: demote
            (3, [0, 1, 0], 3),                   # interrupted streak: unchanged
            (5, [1, 1, 1], 5),                    # clamp at the top
            (1, [0, 0], 1),                       # clamp at the bottom
            (1, [0, 0, 0, 0, 0], 1),              # clamp at the bottom
        ],
    )
    def test_compute_next_level_rules(self, current, outcomes, expected):
        assert compute_next_level(current, outcomes) == expected

    def test_thresholds_are_configurable(self):
        assert compute_next_level(2, [1, 1], promote_after=2) == 3
        assert compute_next_level(2, [0], demote_after=1) == 1

    def test_get_next_difficulty_promotes_and_persists(self, seeded_db):
        # Alice's stored history ends on a wrong answer; three newer correct answers
        # start a fresh streak and promote level 1 -> 2.
        insert_history(seeded_db, [(1, 1, 1), (1, 2, 1), (1, 1, 1)])
        assert get_next_difficulty(1) == 2
        assert db_rows(seeded_db, "SELECT level FROM users WHERE id = 1") == [(2,)]

    def test_get_next_difficulty_demotes(self, seeded_db):
        insert_history(seeded_db, [(2, 1, 0), (2, 2, 0)])
        assert get_next_difficulty(2) == 2
        assert db_rows(seeded_db, "SELECT level FROM users WHERE id = 2") == [(2,)]

    def test_get_next_difficulty_keeps_level_without_history(self, seeded_db):
        assert get_next_difficulty(2) == 3
        assert db_rows(seeded_db, "SELECT level FROM users WHERE id = 2") == [(3,)]

    def test_get_next_difficulty_for_unknown_user_defaults_to_min(self, seeded_db):
        assert get_next_difficulty(4242) == MIN_LEVEL

    def test_clamp_protects_against_levels_outside_the_schema_range(self):
        # The CHECK constraint normally prevents this, but legacy databases are
        # still clamped on read rather than trusted.
        assert compute_next_level(99, [1, 1, 1]) == MAX_LEVEL
        assert compute_next_level(-4, [0, 0]) == MIN_LEVEL

    def test_get_next_difficulty_validates_uid(self, seeded_db):
        with pytest.raises(ValueError, match="uid must be a positive integer"):
            get_next_difficulty(0)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
class TestAnalytics:
    def test_user_stats(self, seeded_db):
        stats = sqlite_manager.get_user_stats(1)
        assert stats["total_attempted"] == 4
        assert stats["total_correct"] == 2
        assert stats["accuracy"] == pytest.approx(50.0)
        assert stats["current_streak"] == 0  # the two most recent answers were wrong

    def test_user_stats_for_user_without_history(self, seeded_db):
        assert sqlite_manager.get_user_stats(2) == {
            "total_attempted": 0,
            "total_correct": 0,
            "accuracy": 0.0,
            "current_streak": 0,
        }

    def test_weak_topics_shape(self, seeded_db):
        weak = sqlite_manager.get_weak_topics(1)
        assert set(weak) == {"Math", "Science"}
        assert weak["Math"] == {
            "correct": 2.0,
            "incorrect": 1.0,
            "total": 3.0,
            "accuracy": pytest.approx(2 / 3),
        }
        assert weak["Science"] == {
            "correct": 0.0,
            "incorrect": 1.0,
            "total": 1.0,
            "accuracy": 0.0,
        }

    def test_progress_timeline_is_oldest_first(self, seeded_db):
        timeline = sqlite_manager.get_progress_timeline(1)
        assert [row["is_correct"] for row in timeline] == [1, 1, 0, 0]
        assert timeline[0]["ts"].startswith("2026-01-01")

    def test_difficulty_scores(self, seeded_db):
        scores = {row["difficulty"]: row for row in sqlite_manager.get_difficulty_scores(1)}
        assert sorted(scores) == [1, 2]
        assert scores[1]["total_attempts"] == 3
        assert scores[1]["avg_score"] == pytest.approx(2 / 3)
        assert scores[2]["total_attempts"] == 1
        assert scores[2]["avg_score"] == pytest.approx(0.0)

    def test_analytics_reject_bad_uid(self, seeded_db):
        for func in (
            sqlite_manager.get_user_stats,
            sqlite_manager.get_weak_topics,
            sqlite_manager.get_progress_timeline,
            sqlite_manager.get_difficulty_scores,
        ):
            with pytest.raises(ValueError):
                func(-1)

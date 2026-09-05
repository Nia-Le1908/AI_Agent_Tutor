"""
End-to-end checks for the command line entry points the README documents.

These are deliberately subprocess-based: they prove `python init_db.py`,
`python generate_mock_data.py` and the import-time behaviour of `streamlit run
app.py` still work after the refactor, including their exit codes and stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, *args: str, cwd: Path | None = None, env_extra: dict | None = None):
    """
    Execute a project script from ``cwd``.

    The copy inside ``cwd`` is what runs (not the repo file), so config.py resolves
    PROJECT_ROOT - and therefore DB_PATH - to the scratch directory. That keeps
    these tests from ever touching the real data/ai_tutor_v5.db.
    """
    workspace = Path(cwd or REPO_ROOT)
    env = {"PYTHONUNBUFFERED": "1"}
    if env_extra:
        env.update(env_extra)

    return subprocess.run(
        [sys.executable, str(workspace / name), *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, **env},
    )


# Files the CLI scripts import, so a scratch workspace is a self-contained project.
PROJECT_FILES = (
    "schema.sql",
    "schema.json",
    "init_db.py",
    "generate_mock_data.py",
    "config.py",
    "config_db.py",
    "db_manager.py",
    "validation.py",
    "schemas.py",
    "json_parser.py",
    "sqlite_manager.py",
    "logging_setup.py",
    "faiss_store.py",
)


@pytest.fixture
def workspace(tmp_path) -> Path:
    """A minimal copy of the project that CLI scripts can run inside."""
    for name in PROJECT_FILES:
        shutil.copy(REPO_ROOT / name, tmp_path / name)
    return tmp_path


class TestInitDbCli:
    def test_creates_a_database_and_reports_the_path(self, workspace):
        result = run_script("init_db.py", cwd=workspace)

        assert result.returncode == 0, result.stderr
        assert "[OK] Database initialized at:" in result.stdout
        assert (workspace / "data" / "ai_tutor_v5.db").exists()

    def test_workspace_run_does_not_touch_the_repo_database(self, workspace):
        before = (REPO_ROOT / "data" / "ai_tutor_v5.db").read_bytes()
        run_script("init_db.py", cwd=workspace)
        assert (REPO_ROOT / "data" / "ai_tutor_v5.db").read_bytes() == before

    def test_second_run_is_a_no_op(self, workspace):
        assert run_script("init_db.py", cwd=workspace).returncode == 0
        second = run_script("init_db.py", cwd=workspace)

        assert second.returncode == 0, second.stderr
        assert "[OK]" in second.stdout

    def test_failure_exits_non_zero_with_a_readable_message(self, workspace):
        # A path that cannot hold a database must produce a clean error, not a traceback.
        (workspace / "data").write_text("blocked", encoding="utf-8")
        result = run_script("init_db.py", cwd=workspace, env_extra={"DB_PATH": str(workspace / "data")})

        assert result.returncode == 1
        assert "[ERROR] Failed to initialize database" in result.stderr
        assert "Traceback" not in result.stderr


class TestMockDataCli:
    def test_generates_valid_questions_and_a_mock_database(self, workspace):
        result = run_script("generate_mock_data.py", cwd=workspace)
        assert result.returncode == 0, result.stderr

        questions = json.loads((workspace / "mock_data" / "mock_questions.json").read_text(encoding="utf-8"))
        assert len(questions) == 10
        assert {q["answer"] for q in questions} <= {"A", "B", "C", "D"}

        connection = sqlite3.connect(workspace / "mock_data" / "mock_db.sqlite")
        try:
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("users", "questions", "history", "weak_topics")
            }
        finally:
            connection.close()

        assert counts["questions"] == 10
        assert counts["history"] > 0
        # weak_topics is materialized from history, so it must not be empty.
        assert counts["weak_topics"] > 0

    def test_options_are_stored_in_the_canonical_object_shape(self, workspace):
        run_script("generate_mock_data.py", cwd=workspace)
        connection = sqlite3.connect(workspace / "mock_data" / "mock_db.sqlite")
        try:
            raw = connection.execute("SELECT options FROM questions LIMIT 1").fetchone()[0]
        finally:
            connection.close()

        assert sorted(json.loads(raw)) == ["A", "B", "C", "D"]


class TestAppImport:
    def test_app_module_imports_cleanly(self):
        """`streamlit run app.py` starts by importing; that import must not explode."""
        result = subprocess.run(
            [sys.executable, "-c", "import app; print('imported', bool(app.main))"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "imported True" in result.stdout

    def test_no_module_imports_the_dead_gemini_or_ollama_paths(self):
        sources = {path.name: path.read_text(encoding="utf-8") for path in REPO_ROOT.glob("*.py")}
        offenders = {
            name: text
            for name, text in sources.items()
            if "google.generativeai" in text or "import ollama" in text
        }
        assert offenders == {}


class TestShippedDatabase:
    """
    The repository ships data/ai_tutor_v5.db, so the queries must run against it.

    This catches drift between schema.sql and the checked-in database, which unit
    tests built from a scratch DB cannot see.
    """

    @pytest.fixture
    def shipped_db(self, tmp_path) -> Path:
        """
        A throwaway copy of the checked-in demo database.

        Reading the real file would still bump SQLite's header bookkeeping, which
        shows up as an unrelated diff on a tracked binary - so tests never open it
        directly.
        """
        source = REPO_ROOT / "data" / "ai_tutor_v5.db"
        if not source.exists():
            pytest.skip("shipped demo database is not present")

        copy = tmp_path / "shipped_copy.db"
        shutil.copy(source, copy)
        return copy

    def test_repository_queries_run_against_it(self, shipped_db, monkeypatch):
        import config
        import sqlite_manager

        monkeypatch.setattr(config, "DB_PATH", str(shipped_db))

        sqlite_manager.get_all_subjects()
        sqlite_manager.get_question_by_diff(1)
        sqlite_manager.get_questions_filtered(1, subject=None, exclude_uid=1)
        sqlite_manager.get_user_stats(1)
        sqlite_manager.get_weak_topics(1)
        sqlite_manager.get_progress_timeline(1)
        sqlite_manager.get_difficulty_scores(1)
        sqlite_manager.get_user_level(1)

    def test_dashboard_chart_builders_accept_its_rows(self, shipped_db, monkeypatch):
        import config
        import dashboard
        import sqlite_manager

        monkeypatch.setattr(config, "DB_PATH", str(shipped_db))
        uid = 1

        dashboard.build_subject_pie(sqlite_manager.get_weak_topics(uid))
        dashboard.build_weak_topic_radar(sqlite_manager.get_weak_topics(uid))
        dashboard.build_progress_line(sqlite_manager.get_progress_timeline(uid))
        dashboard.build_difficulty_bar(sqlite_manager.get_difficulty_scores(uid))

    def test_read_paths_issue_no_writes(self, shipped_db, monkeypatch):
        """
        Reading questions and statistics must leave the file byte-identical.

        Guards against a read helper silently gaining a write (e.g. persisting a
        computed level) and corrupting a shipped/read-only database in production.
        """
        import config
        import sqlite_manager

        before = shipped_db.read_bytes()
        monkeypatch.setattr(config, "DB_PATH", str(shipped_db))

        sqlite_manager.get_all_subjects()
        sqlite_manager.get_user_stats(1)
        sqlite_manager.get_weak_topics(1)

        assert shipped_db.read_bytes() == before

    def test_schema_matches_the_checked_in_database(self, shipped_db, tmp_path):
        """Every column the app reads must exist in the shipped file."""
        fresh = tmp_path / "fresh.db"
        with sqlite3.connect(fresh) as connection:
            connection.executescript((REPO_ROOT / "schema.sql").read_text(encoding="utf-8"))

        def columns(path):
            with sqlite3.connect(path) as conn:
                return {
                    table: [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
                    for table in ("users", "questions", "history", "sessions")
                }

        assert columns(shipped_db) == columns(fresh)

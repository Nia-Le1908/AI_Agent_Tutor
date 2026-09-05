"""Tests for the dashboard, configuration, schema bootstrap and authentication."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import auth
import config
import config_db
import db_manager
import init_db
from helpers import db_rows


# ---------------------------------------------------------------------------
# Dashboard chart builders (pure functions, no Streamlit runtime)
# ---------------------------------------------------------------------------
class TestDashboardCharts:
    @pytest.fixture(autouse=True)
    def _import_dashboard(self):
        import dashboard

        self.dashboard = dashboard

    def test_subject_pie_splits_correct_and_incorrect(self):
        figure = self.dashboard.build_subject_pie(
            {"Math": {"correct": 3, "incorrect": 1}, "Science": {"correct": 0, "incorrect": 2}}
        )
        pie = figure.data[0]

        assert list(pie.labels) == ["Math - Đúng", "Math - Sai", "Science - Sai"]
        assert list(pie.values) == [3, 1, 2]

    def test_subject_pie_without_data_is_labelled_not_blank(self):
        figure = self.dashboard.build_subject_pie({})
        assert list(figure.data[0].labels) == [self.dashboard.NO_DATA_LABEL]

    def test_subject_pie_ignores_zero_counts(self):
        figure = self.dashboard.build_subject_pie({"Math": {"correct": 0, "incorrect": 0}})
        assert list(figure.data[0].labels) == [self.dashboard.NO_DATA_LABEL]

    def test_radar_closes_the_polygon(self):
        figure = self.dashboard.build_weak_topic_radar(
            {"Math": {"accuracy": 1.0}, "Science": {"accuracy": 0.5}}
        )
        trace = figure.data[0]

        assert list(trace.theta) == ["Math", "Science", "Math"]
        assert list(trace.r) == pytest.approx([100.0, 50.0, 100.0])
        assert list(figure.layout.polar.radialaxis.range) == [0, 100]

    def test_radar_without_data_shows_placeholder(self):
        figure = self.dashboard.build_weak_topic_radar({})
        assert not figure.data
        assert figure.layout.annotations[0]["text"] == self.dashboard.NO_DATA_LABEL

    def test_progress_line_is_cumulative(self):
        figure = self.dashboard.build_progress_line(
            [
                {"ts": "2026-01-01 09:00:00", "is_correct": 1},
                {"ts": "2026-01-01 09:01:00", "is_correct": 0},
                {"ts": "2026-01-01 09:02:00", "is_correct": 1},
            ]
        )
        trace = figure.data[0]

        assert trace.y[0] == pytest.approx(100.0)
        assert trace.y[1] == pytest.approx(50.0)
        assert trace.y[2] == pytest.approx(2 / 3 * 100)
        assert list(trace.x) == ["2026-01-01 09:00", "2026-01-01 09:01", "2026-01-01 09:02"]

    def test_progress_line_tolerates_malformed_timestamps(self):
        figure = self.dashboard.build_progress_line([{"ts": "not-a-date", "is_correct": 1}])
        assert list(figure.data[0].x) == ["not-a-date"]

    def test_progress_line_without_data(self):
        assert not self.dashboard.build_progress_line([]).data

    def test_difficulty_bar_reports_attempt_counts(self):
        figure = self.dashboard.build_difficulty_bar(
            [{"difficulty": 1, "avg_score": 0.75, "total_attempts": 4}]
        )
        bar = figure.data[0]

        assert list(bar.x) == ["Độ khó 1"]
        assert bar.y[0] == pytest.approx(75.0)
        assert list(bar.text) == ["75.0% (4 câu)"]

    def test_format_timestamp(self):
        assert self.dashboard.format_timestamp("2026-05-04T10:11:12") == "2026-05-04 10:11"
        assert self.dashboard.format_timestamp(None) == "None"

    def test_render_dashboard_rejects_bad_uid(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(self.dashboard.st, "warning", lambda message: warnings.append(message))

        for bad in (None, 0, -1, "7"):
            self.dashboard.render_dashboard(bad)

        assert len(warnings) == 4

    def test_render_dashboard_queries_once_per_section(self, seeded_db, monkeypatch):
        """
        The pie and radar charts share one query.

        Before this refactor the dashboard aggregated subject stats twice (once via
        its own SQL, once via get_weak_topics), which is exactly the kind of drift
        this test guards against.
        """
        import dashboard

        calls = {"weak_topics": 0}
        original = dashboard.get_weak_topics

        def counting_weak_topics(uid):
            calls["weak_topics"] += 1
            return original(uid)

        monkeypatch.setattr(dashboard, "get_weak_topics", counting_weak_topics)
        monkeypatch.setattr(dashboard.st, "subheader", lambda *a, **k: None)
        monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **k: None)
        monkeypatch.setattr(dashboard.st, "columns", lambda n=2: [FakeContainer() for _ in range(n)])
        monkeypatch.setattr(dashboard.st, "plotly_chart", lambda *a, **k: None)
        monkeypatch.setattr(dashboard.st, "metric", lambda *a, **k: None)

        dashboard.render_dashboard(1)
        assert calls["weak_topics"] == 1


class FakeContainer:
    """Stand-in for an st.columns() container used as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def metric(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def run_config_snippet(env: dict, snippet: str) -> subprocess.CompletedProcess:
    """
    Evaluate a snippet in a fresh interpreter with the given environment.

    config.py validates once, at import, so checking its failure modes requires a
    subprocess rather than monkeypatching.
    """
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO_ROOT),
        env={**os.environ, **env, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )


class TestConfig:
    def test_defaults_match_the_project_spec(self):
        # Chunking/top-k defaults are frozen by the project spec.
        assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.TOP_K) == (256, 50, 3)
        assert config.DB_TYPE == "sqlite"
        assert config.MAX_BATCH_SIZE == 10

    def test_paths_are_absolute(self):
        from pathlib import Path

        for value in (config.DB_PATH, config.FAISS_INDEX_PATH, config.LOG_PATH):
            assert Path(value).is_absolute()

    def test_resolve_path_anchors_relative_values(self):
        assert config.resolve_path("data/x.db").is_absolute()
        assert config.resolve_path("/abs/x.db").as_posix() == "/abs/x.db"

    def test_env_flag_parsing(self, monkeypatch):
        for raw, expected in [("true", True), ("1", True), ("on", True), ("off", False), ("", False)]:
            monkeypatch.setenv("SOME_FLAG", raw)
            assert config.env_flag("SOME_FLAG") is expected
        monkeypatch.delenv("SOME_FLAG", raising=False)
        assert config.env_flag("SOME_FLAG", default=True) is True

    @pytest.mark.parametrize("value, expected", [("400", "400"), ("256", "256")])
    def test_chunk_size_accepts_the_documented_range(self, value, expected):
        result = run_config_snippet(
            {"CHUNK_SIZE": value}, "import config; print(config.CHUNK_SIZE)"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

    @pytest.mark.parametrize("value", ["128", "1024", "abc"])
    def test_invalid_chunk_size_fails_loudly(self, value):
        result = run_config_snippet({"CHUNK_SIZE": value}, "import config")
        assert result.returncode != 0
        assert "CHUNK_SIZE" in result.stderr

    def test_overlap_must_be_smaller_than_chunk_size(self):
        result = run_config_snippet(
            {"CHUNK_SIZE": "256", "CHUNK_OVERLAP": "256"}, "import config"
        )
        assert result.returncode != 0
        assert "strictly smaller" in result.stderr

    def test_unknown_db_type_is_rejected(self):
        result = run_config_snippet({"DB_TYPE": "mysql"}, "import config")
        assert result.returncode != 0
        assert "DB_TYPE must be one of" in result.stderr

    def test_postgres_requires_a_password(self):
        result = run_config_snippet(
            {"DB_TYPE": "postgresql", "POSTGRES_PASSWORD": ""}, "import config"
        )
        assert result.returncode != 0
        assert "POSTGRES_PASSWORD is required" in result.stderr

    def test_postgres_settings_are_accepted(self):
        result = run_config_snippet(
            {"DB_TYPE": "postgresql", "POSTGRES_PASSWORD": "secret"},
            "import config; print(config.DB_TYPE, config.POSTGRES_POOL_SIZE)",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "postgresql 10"

    def test_sqlite_db_path_alias_is_honoured(self):
        result = run_config_snippet(
            {"SQLITE_DB_PATH": "data/legacy.db"},
            "import config; print(config.DB_PATH.endswith('data/legacy.db'))",
        )
        assert result.stdout.strip() == "True"

    def test_db_path_wins_over_the_legacy_alias(self):
        result = run_config_snippet(
            {"DB_PATH": "data/canonical.db", "SQLITE_DB_PATH": "data/legacy.db"},
            "import config; print(config.DB_PATH)",
        )
        assert result.stdout.strip().endswith("data/canonical.db")

    def test_importing_config_creates_nothing(self, tmp_path):
        """Import is side-effect free: directories are created on demand only."""
        result = run_config_snippet(
            {"DB_PATH": str(tmp_path / "sub" / "x.db")},
            "import pathlib, config; print(pathlib.Path(config.DB_PATH).parent.exists())",
        )
        assert result.stdout.strip() == "False"

    def test_deepseek_key_is_only_required_when_used(self, tmp_path):
        result = run_config_snippet(
            {"DEEPSEEK_API_KEY": ""},
            "import config\ntry:\n    config.require_deepseek_api_key()\nexcept config.ConfigError as e:\n    print('guarded')",
        )
        assert result.stdout.strip() == "guarded"

    def test_no_gemini_settings_remain(self):
        """The provider moved to DeepSeek; a stale GEMINI key would mislead setup."""
        assert not hasattr(config, "GEMINI_API_KEY")
        assert not hasattr(config, "require_gemini_api_key")


# ---------------------------------------------------------------------------
# config_db facade
# ---------------------------------------------------------------------------
class TestConfigDbFacade:
    """The facade must stay a *derived* view: one source of values, no re-parsing."""

    def test_facade_is_self_consistent(self):
        assert config_db.DB_TYPE == config_db.db_config.DB_TYPE
        assert config_db.DB_PATH == config_db.db_config.SQLITE_DB_PATH
        assert config_db.get_db_type() == "sqlite"
        assert config_db.is_sqlite() and not config_db.is_postgresql()
        assert config_db.db_config.get_driver_name() == "sqlite"

    def test_connection_string_matches_the_configured_sqlite_path(self):
        assert config_db.CONNECTION_STRING == f"sqlite:///{Path(config_db.DB_PATH).resolve()}"

    def test_facade_does_not_re_read_the_environment(self):
        """
        config_db used to parse .env a second time with its own defaults, which let
        DB_PATH and SQLITE_DB_PATH disagree. It now borrows config's values.
        """
        source = (
            Path(__file__).resolve().parents[1] / "config_db.py"
        ).read_text(encoding="utf-8")
        assert "load_dotenv" not in source
        assert "os.getenv" not in source

    def test_explicit_backend_selection(self):
        backend = config_db.DatabaseConfig(db_type="postgresql", sqlite_path="ignored")
        assert backend.get_driver_name() == "postgresql"
        assert backend.is_postgresql()
        assert backend.get_connection_string().startswith("postgresql://")

    def test_password_is_not_exposed_as_an_attribute(self):
        # It is reachable for building the DSN, but not through a public attribute.
        assert not hasattr(config_db.db_config, "POSTGRES_PASSWORD")


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------
class TestInitDb:
    def test_initializes_all_required_tables(self, isolated_config):
        path = init_db.initialize_database()
        assert path.exists()

        tables = {
            row[0]
            for row in db_rows(
                path, "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert init_db.REQUIRED_TABLES <= tables

    def test_is_idempotent(self, isolated_config):
        init_db.initialize_database()
        init_db.initialize_database()  # must not raise on the second run

    def test_cli_reports_success(self, isolated_config):
        assert init_db.main() == 0

    def test_missing_schema_file_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(init_db, "SCHEMA_SQL_PATH", tmp_path / "absent.sql")
        with pytest.raises(FileNotFoundError, match="schema.sql not found"):
            init_db.initialize_database()

    def test_verify_tables_names_what_is_missing(self, tmp_path):
        import sqlite3

        connection = sqlite3.connect(tmp_path / "partial.db")
        connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        connection.commit()

        with pytest.raises(init_db.SchemaBootstrapError, match="history, questions, sessions"):
            init_db.verify_tables(connection)
        connection.close()

    def test_empty_schema_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.sql"
        empty.write_text("   ", encoding="utf-8")
        with pytest.raises(ValueError, match="schema.sql is empty"):
            init_db.read_schema_sql(empty)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
class TestPasswords:
    def test_hash_is_bcrypt_and_verifiable(self):
        digest = auth.hash_password("correct horse battery staple")
        assert digest.startswith(auth.BCRYPT_HASH_PREFIX)
        assert auth.verify_password("correct horse battery staple", digest)
        assert not auth.verify_password("wrong", digest)

    def test_same_password_hashes_differ(self):
        assert auth.hash_password("pw") != auth.hash_password("pw")

    def test_legacy_sha256_hashes_still_verify(self):
        import hashlib

        legacy = hashlib.sha256(b"old-password").hexdigest()
        assert auth.verify_password("old-password", legacy)
        assert not auth.verify_password("nope", legacy)

    def test_legacy_verification_can_be_switched_off(self, monkeypatch):
        monkeypatch.setattr(auth, "LEGACY_PASSWORD_FALLBACK_ENABLED", False)
        assert auth.verify_password("old-password", "0" * 64) is False

    def test_hashing_requires_bcrypt(self, monkeypatch):
        monkeypatch.setattr(auth, "BCRYPT_AVAILABLE", False)
        with pytest.raises(auth.SecurityUnavailableError, match="bcrypt is required"):
            auth.hash_password("x")

    def test_empty_password_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            auth.hash_password("")

    def test_malformed_hash_does_not_raise(self):
        assert auth.verify_password("x", "$2b$12$not-a-real-hash") is False
        assert auth.verify_password("", "anything") is False
        assert auth.verify_password(None, None) is False

    def test_rehash_is_requested_for_legacy_hashes_only(self):
        assert auth.needs_password_rehash("a" * 64)
        assert not auth.needs_password_rehash(auth.hash_password("x"))


class TestTokens:
    def test_round_trip(self):
        token = auth.generate_token(user_id=7, role="admin", expires_hours=1)
        payload = auth.verify_token(token)

        assert payload["user_id"] == 7
        assert payload["role"] == "admin"
        assert payload["type"] == "access"

    def test_token_expires(self, monkeypatch):
        import jwt as pyjwt

        expired_token = pyjwt.encode(
            {"user_id": 1, "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
            auth._get_secret_key(),
            algorithm="HS256",
        )
        with pytest.raises(auth.AuthenticationError, match="expired"):
            auth.verify_token(expired_token)

    def test_tampered_token_is_rejected(self):
        token = auth.generate_token(user_id=1)
        with pytest.raises(auth.AuthenticationError, match="Invalid token"):
            auth.verify_token(token + "x")

    @pytest.mark.parametrize("bad", ["", None])
    def test_missing_token_is_rejected(self, bad):
        with pytest.raises(auth.AuthenticationError, match="Authentication token required"):
            auth.verify_token(bad)

    def test_default_lifetime_comes_from_config(self):
        token = auth.generate_token(user_id=1)
        payload = auth.verify_token(token)
        lifetime = datetime.fromtimestamp(payload["exp"], tz=timezone.utc) - datetime.fromtimestamp(
            payload["iat"], tz=timezone.utc
        )
        assert lifetime == timedelta(hours=config.SESSION_TIMEOUT_HOURS)


class TestDecorators:
    def test_require_auth_injects_identity(self):
        token = auth.generate_token(user_id=42, role="user")

        @auth.require_auth
        def handler(**kwargs):
            return kwargs["user_id"], kwargs["user_role"]

        assert handler(token=token) == (42, "user")

    def test_require_auth_without_token_fails(self):
        @auth.require_auth
        def handler(**kwargs):
            return "unreachable"

        with pytest.raises(auth.AuthenticationError):
            handler()

    def test_role_checks_use_precedence(self):
        user_token = auth.generate_token(user_id=1, role="user")
        admin_token = auth.generate_token(user_id=2, role="admin")

        @auth.require_role("admin")
        def admin_only(**kwargs):
            return "ok"

        assert admin_only(token=admin_token) == "ok"
        with pytest.raises(auth.AuthorizationError, match="Required role: admin"):
            admin_only(token=user_token)

    def test_user_role_is_open_to_everyone(self):
        @auth.require_role("user")
        def any_signed_in(**kwargs):
            return "ok"

        assert any_signed_in(token=auth.generate_token(user_id=1, role="admin")) == "ok"

    def test_authenticated_caller_can_pass_a_resolved_user(self):
        @auth.require_auth
        def handler(**kwargs):
            return kwargs["user_id"]

        assert handler(current_user={"user_id": 9, "role": "user"}) == 9

    def test_unknown_role_is_a_definition_error(self):
        with pytest.raises(ValueError, match="Unknown role"):
            auth.require_role("superuser")

    def test_pyjwt_required_for_tokens(self, monkeypatch):
        monkeypatch.setattr(auth, "JWT_AVAILABLE", False)
        with pytest.raises(auth.SecurityUnavailableError, match="PyJWT"):
            auth.generate_token(user_id=1)
        with pytest.raises(auth.SecurityUnavailableError, match="PyJWT"):
            auth.verify_token("x")


class TestUserManager:
    @pytest.fixture
    def users(self, db) -> auth.UserManager:
        return auth.UserManager()

    def test_create_user_stores_a_hash_not_the_password(self, users, db):
        uid = users.create_user("Alice", email="Alice@Example.com ", password="s3cret")
        record = db_manager.fetch_one("SELECT * FROM users WHERE id = ?", (uid,))

        assert record["name"] == "Alice"
        assert record["email"] == "alice@example.com"
        assert "s3cret" not in record["password_hash"]
        assert record["password_hash"].startswith("$2")

    def test_create_user_validates_input(self, users, db):
        with pytest.raises(ValueError, match="name"):
            users.create_user("   ")
        with pytest.raises(ValueError, match="level"):
            users.create_user("Bob", level=9)

    def test_duplicate_email_is_rejected_as_database_error(self, users, db):
        users.create_user("First", email="dup@example.com")
        with pytest.raises(db_manager.DatabaseError):
            users.create_user("Second", email="dup@example.com")

    def test_get_user_by_id_roundtrip(self, users, db):
        uid = users.create_user("Carol", level=3)
        record = users.get_user_by_id(uid)

        assert record["name"] == "Carol"
        assert record["level"] == 3
        assert "password_hash" not in record  # never exposed by this lookup
        assert users.get_user_by_id(999) is None

    def test_authentication_by_name_or_email(self, users, db):
        users.create_user("Dave", email="dave@example.com", password="hunter2")

        for identifier in ("Dave", "dave@example.com", "DAVE@EXAMPLE.COM"):
            token = users.authenticate_user(identifier, "hunter2")
            assert token, identifier
            assert auth.verify_token(token)["role"] == "user"

    def test_wrong_password_and_unknown_user_are_indistinguishable(self, users, db):
        users.create_user("Eve", password="right")

        assert users.authenticate_user("Eve", "wrong") is None
        assert users.authenticate_user("ghost", "right") is None

    def test_passwordless_accounts_cannot_sign_in(self, users, db):
        users.create_user("NoPassword")
        assert users.authenticate_user("NoPassword", "") is None

    def test_deactivated_users_are_locked_out(self, users, db):
        uid = users.create_user("Frank", password="pw")
        users.deactivate_user(uid)
        assert users.authenticate_user("Frank", "pw") is None

    def test_last_login_is_recorded(self, users, db):
        uid = users.create_user("Grace", password="pw")
        assert users.authenticate_user("Grace", "pw")
        assert db_rows(db, "SELECT last_login FROM users WHERE id = ?", (uid,))[0][0]

    def test_roles_can_be_assigned_once(self, users, db):
        uid = users.create_user("Heidi")
        assert users.assign_role(uid, "admin") is True
        assert users.assign_role(uid, "admin") is True  # idempotent, not a duplicate row
        assert users.get_user_roles(uid) == ["admin", "user"]

        assert len(db_rows(db, "SELECT * FROM user_roles WHERE user_id = ?", (uid,))) == 2

    def test_assign_role_reports_an_unknown_role(self, users, db):
        uid = users.create_user("Ivan")
        assert users.assign_role(uid, "wizard") is False

    def test_legacy_password_is_upgraded_on_login(self, users, db):
        import hashlib

        uid = users.create_user("Judy")
        legacy = hashlib.sha256(b"legacy-pw").hexdigest()
        db_manager.execute("UPDATE users SET password_hash = ? WHERE id = ?", (legacy, uid))

        assert users.authenticate_user("Judy", "legacy-pw")
        upgraded = db_rows(db, "SELECT password_hash FROM users WHERE id = ?", (uid,))[0][0]
        assert upgraded.startswith("$2")

    def test_admin_role_beats_plain_user_in_the_token(self, users, db):
        uid = users.create_user("Karl", password="pw")
        db_manager.execute("INSERT INTO user_roles (user_id, role_id) VALUES (?, 1)", (uid,))

        token = users.authenticate_user("Karl", "pw")
        assert auth.verify_token(token)["role"] == "admin"

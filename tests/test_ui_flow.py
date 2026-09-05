"""
End-to-end UI tests driven by Streamlit's official AppTest harness.

These are the only tests that execute app.py the way `streamlit run` does, so they
cover what unit tests cannot: session-state wiring, widget keys, rerun behaviour,
and the fact that a click on "Tải câu hỏi mới" really does grade an answer, write
history, and move the adaptive level.

The database is still the scratch copy from conftest and the LLM is still faked, so
the suite stays hermetic.
"""

from __future__ import annotations

import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1", reason="Streamlit >= 1.28 required")
AppTest = st_testing.AppTest

# AppTest resolves relative script paths against the calling file, so pin it down.
APP_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app_test(seeded_db, fake_llm, monkeypatch):
    """A fresh AppTest for app.py, with the LLM faked and the scratch DB in place."""
    import config

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")
    return AppTest.from_file(APP_PATH, default_timeout=60)


def run(app):
    app.run()
    assert not app.exception, f"app raised: {app.exception}"
    return app


class TestBoot:
    def test_app_renders_without_a_user(self, app_test):
        app = run(app_test)

        assert app.title[0].value == "AI Tutor"
        assert [tab.label for tab in app.tabs] == ["Trò chuyện", "Luyện tập", "Thống kê"]
        # Every tab asks for a user first instead of crashing on a missing id.
        infos = [element.value for element in app.info]
        assert any("thanh bên" in text for text in infos)

    def test_session_state_is_seeded(self, app_test):
        app = run(app_test)

        assert app.session_state["current_level"] == 1
        assert app.session_state["subject_filter"] == "Tất cả"
        assert app.session_state["chat_history"] == []


class TestOnboarding:
    def test_blank_name_warns_and_creates_nothing(self, app_test, db):
        app = run(app_test)
        app.sidebar.text_input[0].set_value("   ")
        app.sidebar.button[0].set_value(True).run()

        assert not app.exception
        assert app.session_state["user_id"] is None

    def test_create_then_reload_keeps_the_same_user(self, app_test, db):
        import sqlite_manager

        app = run(app_test)
        app.sidebar.text_input[0].set_value("Streamlit User")
        app.sidebar.button[0].set_value(True).run()

        first = app.session_state["user_id"]
        assert first, "a user id must be stored after clicking the button"
        assert sqlite_manager.get_user_level(first) == 1

        # A second click with the same name must reuse, not duplicate.
        app.sidebar.button[0].set_value(True).run()
        assert app.session_state["user_id"] == first
        assert len(sqlite_manager.get_all_subjects()) >= 1

    def test_current_level_is_shown_in_the_sidebar(self, app_test, seeded_db):
        app = run(app_test)
        app.sidebar.text_input[0].set_value("Alice")
        app.sidebar.button[0].set_value(True).run()

        assert app.session_state["current_level"] == 1
        assert app.session_state["user_name"] == "Alice"


class TestExerciseFlow:
    def _select_user(self, app):
        app.sidebar.text_input[0].set_value("Alice")
        app.sidebar.button[0].set_value(True).run()
        return app

    @staticmethod
    def _label_for(app, letter: str) -> str:
        """Find the rendered radio label that starts with the given option letter."""
        return next(label for label in app.radio[0].options if label.startswith(f"{letter}. "))

    def test_load_question_then_answer_correctly(self, app_test, seeded_db):
        import db_manager

        app = self._select_user(run(app_test))
        app.button[0].set_value(True).run()  # "Tải câu hỏi mới"

        question = app.session_state["current_question"]
        assert question is not None
        assert question["difficulty"] == 1

        history_before = len(db_manager.fetch_all("SELECT id FROM history WHERE uid = 1"))

        label = self._label_for(app, question["answer"])
        app.radio[0].set_value(label).run()

        # The submit button is the next button after "Tải câu hỏi mới".
        submit = next(button for button in app.button if button.label == "Nộp đáp án")
        submit.set_value(True).run()

        assert not app.exception
        assert any("Chính xác" in element.value for element in app.success)

        history_after = db_manager.fetch_all("SELECT * FROM history WHERE uid = 1")
        assert len(history_after) == history_before + 1
        newest = history_after[-1]
        assert newest["qid"] == question["id"] and newest["is_correct"] == 1

    def test_wrong_answer_shows_the_correct_option_and_explanation(self, app_test, seeded_db):
        app = self._select_user(run(app_test))
        app.button[0].set_value(True).run()

        question = app.session_state["current_question"]
        wrong_letter = next(
            letter for letter in ("A", "B", "C", "D") if letter != question["answer"]
        )
        app.radio[0].set_value(self._label_for(app, wrong_letter)).run()

        next(button for button in app.button if button.label == "Nộp đáp án").set_value(True).run()

        errors = [element.value for element in app.error]
        assert any(f"Đáp án đúng là {question['answer']}" in text for text in errors)
        assert any("Giải thích" in text for text in [info.value for info in app.info])

    def test_submitting_without_a_choice_is_blocked(self, app_test, seeded_db):
        app = self._select_user(run(app_test))
        app.button[0].set_value(True).run()

        next(button for button in app.button if button.label == "Nộp đáp án").set_value(True).run()
        assert any("chọn một đáp án" in element.value for element in app.warning)

    def test_level_moves_up_after_three_correct_answers(self, app_test, seeded_db):
        """Adaptive difficulty must survive the full UI round trip."""
        import db_manager

        db_manager.execute("DELETE FROM history WHERE uid = 1")
        db_manager.execute("UPDATE users SET level = 1 WHERE id = 1")

        app = self._select_user(run(app_test))
        for _ in range(3):
            app.button[0].set_value(True).run()  # load question
            question = app.session_state["current_question"]
            app.radio[0].set_value(self._label_for(app, question["answer"])).run()
            next(b for b in app.button if b.label == "Nộp đáp án").set_value(True).run()

        assert app.session_state["current_level"] == 2
        assert db_manager.fetch_one("SELECT level FROM users WHERE id = 1")["level"] == 2

    def test_malformed_question_shows_an_error_instead_of_crashing(self, app_test, seeded_db):
        """Question 6 in the fixture has unparseable options; the UI must say so."""
        app = self._select_user(run(app_test))
        app.selectbox[0].set_value("Math").run()
        app.session_state["current_level"] = 5
        app.button[0].set_value(True).run()

        assert app.session_state["current_question"]["id"] == 6
        assert any("định dạng đáp án không hợp lệ" in text for text in [error.value for error in app.error])
        assert not app.exception

    def test_subject_filter_narrows_the_pool(self, app_test, seeded_db):
        app = self._select_user(run(app_test))
        app.selectbox[0].set_value("Science").run()
        app.button[0].set_value(True).run()

        assert app.session_state["subject_filter"] == "Science"
        assert app.session_state["current_question"]["subject"] == "Science"

    def test_empty_result_set_explains_itself(self, app_test, seeded_db):
        app = self._select_user(run(app_test))
        app.selectbox[0].set_value("Math").run()
        # Level 4 only exists for Science, so Math + high level must be empty.
        app.session_state["current_level"] = 4
        app.button[0].set_value(True).run()

        assert app.session_state["current_question"] is None
        warnings = [element.value for element in app.warning]
        assert any("Không tìm thấy câu hỏi ở độ khó 4 cho môn 'Math'" in text for text in warnings)


class TestChatFlow:
    def _select_user(self, app):
        app.sidebar.text_input[0].set_value("Alice")
        app.sidebar.button[0].set_value(True).run()
        return app

    def test_question_is_answered_with_citations(self, app_test, seeded_db, fake_llm, vector_store, monkeypatch):
        import config

        monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(vector_store["index_path"]))
        monkeypatch.setattr(config, "VECTOR_DIR", str(vector_store["dir"]))

        fake_llm.queue("Newton's second law relates force, mass and acceleration.")

        app = self._select_user(run(app_test))
        app.chat_input[0].set_value("What does Newton's second law say?").run()

        assert not app.exception
        assert fake_llm.calls == 1
        # Retrieved chunks reached the prompt...
        assert "F = ma" in fake_llm.last_prompt or "Newton" in fake_llm.last_prompt
        # ...the answer and the citation are both on screen...
        markdowns = [element.value for element in app.markdown]
        assert any("relates force, mass and acceleration" in text for text in markdowns)
        assert any("Nguồn tham khảo" in text for text in markdowns)
        # ...and the exchange is persisted in session state for the next rerun.
        assert [message["role"] for message in app.session_state["chat_history"]] == [
            "user",
            "assistant",
        ]

    def test_provider_failure_degrades_to_a_message_not_a_crash(self, app_test, seeded_db, monkeypatch):
        import llm_client

        def explode(*args, **kwargs):
            raise llm_client.LLMError("provider unreachable")

        monkeypatch.setattr(llm_client, "chat", explode)

        app = self._select_user(run(app_test))
        app.chat_input[0].set_value("hello?").run()

        assert not app.exception
        assert any("Lỗi khi tạo câu trả lời" in text for text in [markdown.value for markdown in app.markdown])


class TestDashboard:
    def test_dashboard_renders_stats_for_the_selected_user(self, app_test, seeded_db):
        app = run(app_test)
        app.sidebar.text_input[0].set_value("Alice")
        app.sidebar.button[0].set_value(True).run()

        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["📝 Tổng câu đã làm"] == "4"
        assert metrics["✅ Số câu đúng"] == "2"
        assert metrics["🎯 Độ chính xác"] == "50.0%"
        assert not app.exception

    def test_dashboard_metrics_are_absent_before_a_user_exists(self, app_test, db):
        app = run(app_test)
        assert all(metric.label != "📝 Tổng câu đã làm" for metric in app.metric)

"""Tests for orchestration (controller) and the Streamlit UI layer."""

from __future__ import annotations

import pytest

import controller
import db_manager
import llm_client
import sqlite_manager
from helpers import question_payload


# ---------------------------------------------------------------------------
# Prompt + citation helpers (pure)
# ---------------------------------------------------------------------------
class TestPromptBuilding:
    def test_context_blocks_are_labelled_in_order(self):
        prompt = controller.build_chat_prompt("What is BFS?", ["first chunk", "second chunk"])

        assert "[Context 1] first chunk" in prompt
        assert "[Context 2] second chunk" in prompt
        assert "Student question:\nWhat is BFS?" in prompt

    def test_the_tutor_persona_is_kept_in_the_prompt(self):
        assert "student-friendly answers" in controller.build_chat_prompt("q", ["c"])

    def test_missing_context_is_stated_instead_of_fabricated(self):
        assert "No external context found" in controller.build_chat_prompt("q", [])

    @pytest.mark.parametrize(
        "chunks, expected",
        [
            ([], "cannot generate a full model answer"),
            (["alpha", "beta", "gamma"], "alpha\n\nbeta"),
        ],
    )
    def test_quota_fallback(self, chunks, expected):
        assert expected in controller.build_quota_fallback_response(chunks)

    def test_quota_fallback_only_uses_the_top_chunks(self):
        chunks = [f"chunk{index}" for index in range(10)]
        fallback = controller.build_quota_fallback_response(chunks)

        assert "chunk0" in fallback and "chunk1" in fallback and "chunk2" not in fallback

    def test_citations_are_appended_once_per_source(self):
        answer = controller.append_citations("body text", ["a.pdf", "b.pdf"])

        assert answer.startswith("body text")
        assert "a.pdf" in answer and "b.pdf" in answer
        assert answer.count("Nguồn tham khảo") == 1

    def test_no_sources_means_no_citation_block(self):
        assert controller.append_citations("answer", []) == "answer"


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_retrieval(monkeypatch):
    """Replace vector search with fixed chunks so chat() can be tested in isolation."""
    calls: dict = {"fallback": False}

    def fake_search(query, top_k):
        calls.update(query=query, top_k=top_k)
        return [
            {"text": "Newton's second law: F = ma", "source_file": "physics.pdf"},
            {"text": "Momentum is mass times velocity", "source_file": "mechanics.pdf"},
        ]

    def fake_plain_search(query, top_k):
        calls["fallback"] = True
        return ["Newton's second law: F = ma"]

    monkeypatch.setattr(controller.retriever, "retrieve_with_sources", fake_search)
    monkeypatch.setattr(controller.retriever, "retrieve", fake_plain_search)
    return calls


class TestChat:
    def test_answer_is_grounded_and_cited(self, fake_llm, stub_retrieval):
        fake_llm.queue("Newton's second law links force and mass.")

        answer = controller.chat("What is F = ma?")

        assert answer.startswith("Newton's second law links force and mass.")
        assert "physics.pdf" in answer and "mechanics.pdf" in answer
        assert "F = ma" in fake_llm.last_prompt  # retrieved text reached the model

    def test_sources_are_deduplicated_but_ordered(self, fake_llm, stub_retrieval, monkeypatch):
        monkeypatch.setattr(
            controller.retriever,
            "retrieve_with_sources",
            lambda query, top_k: [
                {"text": "a", "source_file": "one.pdf"},
                {"text": "b", "source_file": "one.pdf"},
                {"text": "c", "source_file": "two.pdf"},
            ],
        )
        fake_llm.queue("answer")

        assert controller.chat("q").count("one.pdf") == 1

    def test_top_k_is_forwarded(self, fake_llm, stub_retrieval):
        fake_llm.queue("ok")
        controller.chat("q", top_k=7)
        assert stub_retrieval["top_k"] == 7

    def test_rate_limits_degrade_to_retrieval_only(self, monkeypatch, stub_retrieval):
        def raise_rate_limit(*args, **kwargs):
            raise llm_client.LLMRateLimitError("429 quota")

        monkeypatch.setattr(llm_client, "chat", raise_rate_limit)
        answer = controller.chat("q")

        assert "quota/rate-limit issues" in answer
        assert "Newton" in answer  # local retrieval still delivers something

    def test_other_provider_errors_propagate(self, monkeypatch, stub_retrieval):
        def raise_server_error(*args, **kwargs):
            raise llm_client.LLMError("500 internal")

        monkeypatch.setattr(llm_client, "chat", raise_server_error)
        with pytest.raises(llm_client.LLMError):
            controller.chat("q")

    def test_source_aware_retrieval_failure_falls_back_to_plain_retrieval(
        self, fake_llm, monkeypatch
    ):
        def broken(query, top_k):
            raise RuntimeError("index has no source_file keys")

        monkeypatch.setattr(controller.retriever, "retrieve_with_sources", broken)
        monkeypatch.setattr(controller.retriever, "retrieve", lambda query, top_k: ["chunk"])
        fake_llm.queue("answer")

        assert controller.chat("q") == "answer"  # chat still succeeded
        assert "chunk" in fake_llm.last_prompt  # via the plain-retrieval path

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_invalid_input_is_rejected_before_any_io(self, fake_llm, stub_retrieval, bad):
        with pytest.raises(ValueError, match="user_input"):
            controller.chat(bad)
        assert fake_llm.calls == 0

    def test_non_positive_top_k_is_rejected(self, fake_llm, stub_retrieval):
        with pytest.raises(ValueError, match="top_k"):
            controller.chat("q", top_k=0)


# ---------------------------------------------------------------------------
# generate_exercise_for_user / record_answer
# ---------------------------------------------------------------------------
class TestExerciseFlow:
    def test_exercise_uses_the_adaptive_level(self, seeded_db, monkeypatch):
        db_manager.execute("UPDATE users SET level = 4 WHERE id = 2")

        seen: dict = {}

        def fake_generate(topic, difficulty, model_name):
            seen.update(topic=topic, difficulty=difficulty, model_name=model_name)
            return question_payload(difficulty=difficulty)

        monkeypatch.setattr(controller, "generate", fake_generate)
        result = controller.generate_exercise_for_user(uid=2, topic="BFS", model_name="m1")

        assert seen == {"topic": "BFS", "difficulty": 4, "model_name": "m1"}
        assert result["difficulty"] == 4

    def test_promotion_changes_the_generation_difficulty(self, seeded_db, monkeypatch):
        from helpers import insert_history

        insert_history(seeded_db, [(1, 1, 1), (1, 2, 1), (1, 1, 1)])
        captured: dict = {}

        def capture(topic, difficulty, model_name):
            captured["d"] = difficulty
            return {}

        monkeypatch.setattr(controller, "generate", capture)
        controller.generate_exercise_for_user(uid=1, topic="Math")
        assert captured["d"] == 2  # promoted from level 1 by the three-correct streak

    def test_uid_and_topic_are_validated(self, seeded_db):
        with pytest.raises(ValueError, match="uid"):
            controller.generate_exercise_for_user(uid=0, topic="BFS")
        with pytest.raises(ValueError, match="topic"):
            controller.generate_exercise_for_user(uid=1, topic="  ")


class TestRecordAnswer:
    def test_correct_answer_persists_history_and_reports_fields(self, seeded_db):
        question = sqlite_manager.get_question_by_diff(1)[0]

        result = controller.record_answer(
            uid=2, question=question, selected_answer=question["answer"]
        )

        assert result["is_correct"] is True
        assert result["correct_answer"] == question["answer"]
        assert result["explanation"] == question["explanation"]
        assert db_manager.fetch_all("SELECT * FROM history WHERE uid = 2") != []

    def test_wrong_answer_reports_the_correct_letter(self, seeded_db):
        question = next(q for q in sqlite_manager.get_question_by_diff(1) if q["id"] == 1)
        result = controller.record_answer(uid=2, question=question, selected_answer="C")

        assert result["is_correct"] is False
        assert result["correct_answer"] == "B"

    def test_selection_is_compared_case_and_space_insensitively(self, seeded_db):
        question = sqlite_manager.get_question_by_diff(1)[0]
        padded = f"  {question['answer'].lower()}  "
        assert controller.record_answer(uid=2, question=question, selected_answer=padded)["is_correct"]

    def test_level_recomputes_after_two_consecutive_wrong_answers(self, seeded_db):
        db_manager.execute("UPDATE users SET level = 3 WHERE id = 2")
        question = next(q for q in sqlite_manager.get_question_by_diff(4) if q["id"] == 5)

        assert controller.record_answer(uid=2, question=question, selected_answer="B")["new_level"] == 3
        assert controller.record_answer(uid=2, question=question, selected_answer="B")["new_level"] == 2
        assert db_manager.fetch_one("SELECT level FROM users WHERE id = 2")["level"] == 2

    def test_missing_question_id_is_reported(self, seeded_db):
        with pytest.raises(ValueError, match="question id"):
            controller.record_answer(uid=1, question={"answer": "A"}, selected_answer="A")

    def test_foreign_key_violation_surfaces_as_database_error(self, seeded_db):
        with pytest.raises(db_manager.DatabaseError):
            controller.record_answer(uid=1, question={"id": 9999, "answer": "A"}, selected_answer="A")

    def test_grade_answer_helper(self):
        assert controller.grade_answer({"answer": "a"}, " A ")
        assert not controller.grade_answer({"answer": "A"}, "B")
        assert not controller.grade_answer({}, "A")


# ---------------------------------------------------------------------------
# Streamlit UI helpers (pure functions extracted from app.py)
# ---------------------------------------------------------------------------
@pytest.fixture
def app_module():
    """
    Import app.py for its pure helpers.

    Safe because app.py only calls Streamlit APIs inside main(); importing the
    module does not execute the script body.
    """
    import app

    return app


class TestAppHelpers:
    def test_parse_selected_option(self, app_module):
        assert app_module.parse_selected_option("C. Photosynthesis") == "C"
        assert app_module.parse_selected_option("a. x") == "A"
        assert app_module.parse_selected_option(None) == ""
        assert app_module.parse_selected_option("") == ""

    def test_no_question_message_mentions_level_and_subject(self, app_module):
        assert "độ khó 2" in app_module.build_no_question_message(2, None)
        assert "Toán" in app_module.build_no_question_message(2, "Toán")

    def test_select_question_handles_the_empty_pool(self, app_module):
        assert app_module.select_question([]) is None
        assert app_module.select_question([{"id": 1}]) == {"id": 1}

    def test_subject_options_always_start_with_all(self, app_module):
        assert app_module.subject_options(["Math", "Science"]) == ["Tất cả", "Math", "Science"]
        assert app_module.subject_options([]) == ["Tất cả"]

    def test_subject_selection_index_falls_back_to_all(self, app_module):
        options = app_module.subject_options(["Math"])
        assert app_module.subject_selection_index(options, "Math") == 1
        assert app_module.subject_selection_index(options, "Deleted subject") == 0
        assert app_module.subject_selection_index(options, "Tất cả") == 0

    def test_session_defaults_cover_the_keys_the_app_reads(self, app_module, repo_root):
        """A session key must be declared before use, or Streamlit raises on rerun."""
        source = (repo_root / "app.py").read_text(encoding="utf-8")
        for key in app_module.SESSION_DEFAULTS:
            assert f'"{key}"' in source, f"session key {key} declared but never used"

    def test_ui_stays_out_of_sql(self, repo_root):
        """
        Layering guard: the UI goes through the repository.

        A hand-written connection in app.py is how two divergent connection
        factories and duplicated INSERT statements appeared in the first place.
        """
        source = (repo_root / "app.py").read_text(encoding="utf-8")
        assert "sqlite3" not in source
        assert "INSERT INTO" not in source
        assert "json.loads" not in source  # option parsing belongs to schemas.py


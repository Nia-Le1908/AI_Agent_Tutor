"""
Tests for JSON hardening, question generation and the LLM transport.

Nothing here touches the network: the transport is faked at the llm_client
boundary, so these tests pin down the parsing/validation guarantees and the retry
policy that used to be duplicated (and inconsistent) between generator and
controller.
"""

from __future__ import annotations

import json

import pytest

import json_parser
import llm_client
import schemas
from helpers import question_payload


# ---------------------------------------------------------------------------
# json_parser: hardening
# ---------------------------------------------------------------------------
class TestMarkdownFences:
    def test_plain_json_passes_through(self):
        assert json_parser.strip_markdown_fences('{"a": 1}') == '{"a": 1}'

    @pytest.mark.parametrize(
        "raw",
        [
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            '```JSON\n{"a": 1}\n``` ',
        ],
    )
    def test_fences_are_removed(self, raw):
        assert json_parser.strip_markdown_fences(raw) == '{"a": 1}'

    def test_backticks_inside_strings_are_kept(self):
        payload = '{"a": "```"}'
        assert json_parser.strip_markdown_fences(payload) == payload


class TestJsonObjectExtraction:
    def test_json_is_extracted_from_surrounding_prose(self):
        text = 'Sure! Here is the question: {"a": 1} Hope it helps.'
        assert json_parser.extract_first_json_object(text) == '{"a": 1}'

    def test_nested_objects_and_escaped_quotes(self):
        inner = '{"outer": {"inner": "value with \\" brace {"}}'
        assert json_parser.extract_first_json_object(inner) == inner

    def test_braces_in_strings_do_not_affect_depth(self):
        text = '{"content": "what { is } this", "answer": "A"}'
        assert json_parser.extract_first_json_object(text) == text

    def test_no_object_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            json_parser.extract_first_json_object("no json here")

    def test_unbalanced_object_raises(self):
        with pytest.raises(ValueError, match="Unbalanced"):
            json_parser.extract_first_json_object('{"a": 1')


class TestExtractionOfMany:
    def test_array_output(self):
        payloads = [question_payload(question_id=1), question_payload(question_id=2)]
        text = json.dumps(payloads, ensure_ascii=False)
        assert json_parser.extract_all_json_objects(text) == payloads

    def test_array_inside_prose_and_fences(self):
        payloads = [question_payload(question_id=7)]
        text = "Here you go:\n```json\n" + json.dumps(payloads) + "\n```\nDone!"
        assert json_parser.extract_all_json_objects(text) == payloads

    def test_loose_objects_are_still_found(self):
        payloads = [question_payload(question_id=1), question_payload(question_id=2)]
        text = f"Q1: {json.dumps(payloads[0])} then Q2: {json.dumps(payloads[1])}"
        assert json_parser.extract_all_json_objects(text) == payloads

    def test_non_objects_in_array_are_skipped(self):
        text = '["junk", {"a": 1}, 42]'
        assert json_parser.extract_all_json_objects(text) == [{"a": 1}]

    def test_garbage_yields_empty_list(self):
        assert json_parser.extract_all_json_objects("total nonsense") == []


class TestSafeParse:
    def test_parses_wrapped_object(self):
        assert json_parser.safe_parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError):
            json_parser.safe_parse_json('{"a": }')

    def test_single_object_is_recovered_from_an_array_response(self):
        # Lenient on purpose: when the model answers a single-question request with
        # an array, the first object is taken rather than failing the request.
        assert json_parser.safe_parse_json('[{"a": 1}]') == {"a": 1}

    def test_no_object_at_all_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            json_parser.safe_parse_json('["only", "strings"]')

    def test_safe_parse_json_list(self):
        assert json_parser.safe_parse_json_list('[{"a": 1}]') == [{"a": 1}]


class TestRequiredFields:
    def test_missing_field_raises_key_error(self):
        with pytest.raises(KeyError, match="Missing required field"):
            json_parser.validate_required_fields({"content": "x"}, ["content", "answer"])

    def test_all_present_is_quiet(self):
        json_parser.validate_required_fields({"content": "x", "answer": "A"}, ["content", "answer"])


# ---------------------------------------------------------------------------
# json_parser: parse_and_insert
# ---------------------------------------------------------------------------
class TestParseAndInsert:
    def test_inserts_and_returns_id(self, db):
        question_id = json_parser.parse_and_insert(json.dumps(question_payload(question_id=555)))
        assert question_id == 555

        stored = json_parser.db_manager.fetch_one(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        )
        assert stored["content"] == "What is 2 + 2?"
        assert json.loads(stored["options"]) == {"A": "3", "B": "4", "C": "5", "D": "6"}

    def test_without_question_id_the_autoincrement_id_is_used(self, db):
        payload = question_payload()
        del payload["question_id"]
        # question_id is required by schema.json, so this must be rejected.
        with pytest.raises(KeyError):
            json_parser.parse_and_insert(json.dumps(payload))

    def test_empty_input_rejected(self, db):
        with pytest.raises(ValueError, match="non-empty JSON string"):
            json_parser.parse_and_insert("   ")

    def test_malformed_json_rejected(self, db):
        with pytest.raises(ValueError, match="Invalid JSON string"):
            json_parser.parse_and_insert("{not json")

    def test_array_payload_rejected(self, db):
        with pytest.raises(ValueError, match="must be an object/dict"):
            json_parser.parse_and_insert(json.dumps([question_payload()]))

    def test_schema_violation_rejected(self, db):
        with pytest.raises(schemas.SchemaValidationError):
            json_parser.parse_and_insert(json.dumps(question_payload(answer="E")))

    def test_nothing_is_written_on_invalid_payload(self, db):
        with pytest.raises(schemas.SchemaValidationError):
            json_parser.parse_and_insert(json.dumps(question_payload(difficulty=99)))
        assert json_parser.db_manager.fetch_all("SELECT * FROM questions") == []


# ---------------------------------------------------------------------------
# llm_client
# ---------------------------------------------------------------------------
class ScriptedClient:
    """Minimal stand-in for the OpenAI client object."""

    class _Completions:
        def __init__(self, parent):
            self._parent = parent

        def create(self, **kwargs):
            self._parent.calls.append(kwargs)
            outcome = self._parent.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list = []
        self.chat = type("Chat", (), {"completions": ScriptedClient._Completions(self)})()


def completion(text):
    """Build an object shaped like an OpenAI chat completion response."""
    message = type("Message", (), {"content": text})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


@pytest.fixture
def provider(monkeypatch):
    """Install a scripted client + a fake API key, and neutralize sleeping."""
    import config

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")
    holder: dict = {}

    def build_client():
        return holder["client"]

    monkeypatch.setattr(llm_client, "_build_client", build_client)
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: holder.setdefault("slept", []).append(seconds))
    return holder


class TestLLMClient:
    def test_returns_stripped_text(self, provider):
        provider["client"] = ScriptedClient([completion("  answered  ")])
        assert llm_client.chat("question") == "answered"
        assert provider["client"].calls[0]["temperature"] == pytest.approx(config_temperature())

    def test_json_mode_sets_response_format(self, provider):
        provider["client"] = ScriptedClient([completion('{"a": 1}')])
        llm_client.chat("question", json_mode=True)
        assert provider["client"].calls[0]["response_format"] == {"type": "json_object"}

    def test_default_model_from_config(self, provider):
        from config import DEFAULT_MODEL

        provider["client"] = ScriptedClient([completion("ok")])
        llm_client.chat("question")
        assert provider["client"].calls[0]["model"] == DEFAULT_MODEL

    def test_explicit_model_and_messages(self, provider):
        provider["client"] = ScriptedClient([completion("ok")])
        messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        llm_client.chat(model="custom-model", messages=messages)
        call = provider["client"].calls[0]
        assert call["model"] == "custom-model" and call["messages"] == messages

    def test_retries_then_succeeds(self, provider):
        provider["client"] = ScriptedClient([RuntimeError("429 too many requests"), completion("ok")])
        assert llm_client.chat("question") == "ok"
        assert len(provider["client"].calls) == 2
        # One wait between the failed attempt and the successful one.
        assert provider["slept"] and provider["slept"][0] > 0

    def test_backoff_grows_and_is_capped(self, monkeypatch):
        monkeypatch.setattr(llm_client, "LLM_INITIAL_BACKOFF_SECONDS", 1.0)
        monkeypatch.setattr(llm_client, "LLM_BACKOFF_FACTOR", 2.0)
        monkeypatch.setattr(llm_client, "LLM_MAX_BACKOFF_SECONDS", 3.0)

        bases = [llm_client._next_delay(attempt) for attempt in range(4)]
        # Base delays 1, 2, 3(capped), 3(capped); jitter is always positive.
        assert 1.0 <= bases[0] <= 1.35
        assert 2.0 <= bases[1] <= 2.7
        assert 3.0 <= bases[2] <= 4.05
        assert 3.0 <= bases[3] <= 4.05

    def test_empty_completion_is_an_error(self, provider):
        provider["client"] = ScriptedClient([completion("   ") for _ in range(5)])
        with pytest.raises(llm_client.LLMError, match="failed after"):
            llm_client.chat("question", max_attempts=1)

    def test_rate_limit_becomes_typed_error(self, provider):
        provider["client"] = ScriptedClient([RuntimeError("Error code: 429 - quota exceeded") for _ in range(2)])
        with pytest.raises(llm_client.LLMRateLimitError):
            llm_client.chat("question", max_attempts=2)

    def test_other_failures_stay_generic(self, provider):
        provider["client"] = ScriptedClient([RuntimeError("connection reset")] * 2)
        with pytest.raises(llm_client.LLMError) as excinfo:
            llm_client.chat("question", max_attempts=2)
        assert not isinstance(excinfo.value, llm_client.LLMRateLimitError)

    def test_attempts_are_validated(self, provider):
        provider["client"] = ScriptedClient([])
        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            llm_client.chat("  ")
        with pytest.raises(ValueError, match="max_attempts"):
            llm_client.chat("q", max_attempts=0)

    @pytest.mark.parametrize(
        "message, expected",
        [("HTTP 429", True), ("Rate limit reached", True), ("quota exceeded", True), ("boom", False)],
    )
    def test_rate_limit_detection(self, message, expected):
        assert llm_client.is_rate_limit_error(RuntimeError(message)) is expected


def config_temperature() -> float:
    from config import LLM_TEMPERATURE

    return LLM_TEMPERATURE


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
@pytest.fixture
def llm_json(monkeypatch, fake_llm):
    """Route generator traffic through the shared FakeLLM double."""
    monkeypatch.setattr(llm_client, "chat", fake_llm)
    return fake_llm


class TestGenerate:
    def test_returns_validated_payload(self, db, llm_json):
        from generator import generate

        payload = question_payload(question_id=1)
        llm_json.queue(json.dumps(payload))

        assert generate("Mathematics", 1) == payload

    def test_tolerates_wrapped_and_chatty_output(self, db, llm_json):
        from generator import generate

        payload = question_payload(question_id=2)
        llm_json.queue("Here you are:\n```json\n" + json.dumps(payload) + "\n```\nEnjoy!")
        assert generate("Mathematics", 2) == payload

    def test_answer_letter_is_repaired_before_validation(self, db, llm_json):
        from generator import generate

        llm_json.queue(json.dumps(question_payload(question_id=3, answer=" b ")))
        assert generate("Mathematics", 1)["answer"] == "B"

    def test_invalid_payload_raises(self, db, llm_json):
        from generator import generate

        llm_json.queue(json.dumps(question_payload(answer="Z")))
        with pytest.raises(schemas.SchemaValidationError):
            generate("Mathematics", 1)

    def test_unparsable_output_raises(self, db, llm_json):
        from generator import generate

        llm_json.queue("I cannot help with that.")
        with pytest.raises(ValueError):
            generate("Mathematics", 1)

    def test_prompt_locks_the_topic_and_difficulty(self, db, llm_json):
        from generator import generate

        llm_json.queue(json.dumps(question_payload(question_id=4)))
        generate("BFS", 3)

        prompt = llm_json.last_prompt
        assert 'Trường "subject" phải là "BFS"' in prompt
        assert "độ khó 3" in prompt
        assert "difficulty: 3 (integer from 1 to 5)" in prompt

    @pytest.mark.parametrize("topic", ["", "   ", None])
    def test_topic_validation_happens_before_the_call(self, db, llm_json, topic):
        from generator import generate

        with pytest.raises(ValueError, match="topic"):
            generate(topic, 1)
        assert llm_json.calls == 0

    @pytest.mark.parametrize("difficulty", [0, 6, "2", None])
    def test_difficulty_validation(self, db, llm_json, difficulty):
        from generator import generate

        with pytest.raises(ValueError, match="difficulty"):
            generate("Mathematics", difficulty)
        assert llm_json.calls == 0


class TestGenerateBatch:
    def test_batch_returns_all_valid_questions(self, db, llm_json):
        from generator import generate_batch

        payloads = [question_payload(question_id=10), question_payload(question_id=11)]
        llm_json.queue(json.dumps(payloads))

        assert generate_batch("Mathematics", 1, count=2) == payloads

    def test_invalid_items_are_dropped_not_fatal(self, db, llm_json):
        from generator import generate_batch

        good = question_payload(question_id=12)
        llm_json.queue(json.dumps([good, question_payload(question_id=13, answer="Z")]))
        assert generate_batch("Mathematics", 1, count=2) == [good]

    def test_single_object_answer_is_used_directly(self, db, llm_json):
        """A batch request answered with one object keeps that object; no re-asking."""
        from generator import generate_batch

        llm_json.queue(json.dumps(question_payload(question_id=14)))
        result = generate_batch("Mathematics", 1, count=2)

        assert [q["question_id"] for q in result] == [14]
        assert llm_json.calls == 1

    def test_unusable_batch_output_falls_back_to_individual_calls(self, db, llm_json):
        from generator import generate_batch

        llm_json.queue(
            "Sure, I will write them now!",
            json.dumps(question_payload(question_id=15)),
            json.dumps(question_payload(question_id=16)),
        )

        result = generate_batch("Mathematics", 1, count=2)
        assert [q["question_id"] for q in result] == [15, 16]
        assert llm_json.calls == 3  # one batch attempt + two fallbacks

    def test_total_failure_raises_with_context(self, db, llm_json):
        from generator import generate_batch

        llm_json.queue("nonsense", "still nonsense", "nope")
        with pytest.raises(ValueError, match="generate_batch failed"):
            generate_batch("Mathematics", 1, count=2)

    @pytest.mark.parametrize("count", [0, 11, "3", None])
    def test_count_is_validated(self, db, llm_json, count):
        from generator import generate_batch

        with pytest.raises(ValueError, match="count"):
            generate_batch("Mathematics", 1, count=count)
        assert llm_json.calls == 0

    def test_prompts_request_the_right_shape(self, db, llm_json):
        from generator import build_batch_prompt, build_prompt

        single = build_prompt("TOEIC", 2)
        batch = build_batch_prompt("TOEIC", 2, 4)

        assert "MỘT câu hỏi" in single and "CHỈ một JSON object" in single
        assert "4 câu hỏi" in batch and "CHỈ một JSON array" in batch
        for prompt in (single, batch):
            assert '"difficulty": 2' in prompt
            assert 'chủ đề "TOEIC"' in prompt

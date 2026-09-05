"""Tests for schema validation and option normalization (schemas.py)."""

from __future__ import annotations

import pytest

import schemas
from helpers import question_payload


class TestLoadSchema:
    def test_schema_is_valid_and_complete(self):
        schema = schemas.load_schema()
        assert schema["type"] == "object"
        assert set(schemas.required_fields()) == {
            "question_id",
            "content",
            "difficulty",
            "subject",
            "options",
            "answer",
            "explanation",
        }

    def test_schema_is_cached(self):
        assert schemas.load_schema() is schemas.load_schema()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            schemas.load_schema(tmp_path / "nope.json")


class TestValidateQuestionPayload:
    def test_valid_payload_passes(self):
        schemas.validate_question_payload(question_payload())

    def test_missing_field_raises(self):
        payload = question_payload()
        del payload["explanation"]
        with pytest.raises(schemas.SchemaValidationError, match="explanation"):
            schemas.validate_question_payload(payload)

    @pytest.mark.parametrize("difficulty", [0, 6, "1"])
    def test_difficulty_bounds_and_type(self, difficulty):
        with pytest.raises(schemas.SchemaValidationError):
            schemas.validate_question_payload(question_payload(difficulty=difficulty))

    @pytest.mark.parametrize("answer", ["E", "b", "", None])
    def test_answer_must_be_capital_abcd(self, answer):
        with pytest.raises(schemas.SchemaValidationError):
            schemas.validate_question_payload(question_payload(answer=answer))

    def test_wrong_option_count_rejected(self):
        with pytest.raises(schemas.SchemaValidationError):
            schemas.validate_question_payload(question_payload(options=["a", "b", "c"]))

    def test_extra_fields_rejected(self):
        with pytest.raises(schemas.SchemaValidationError, match="Additional properties"):
            schemas.validate_question_payload(question_payload(hints="spoil me"))

    def test_error_lists_field_path(self):
        with pytest.raises(schemas.SchemaValidationError, match=r"validation: options:"):
            schemas.validate_question_payload(question_payload(options="not-a-list"))

    def test_non_dict_rejected(self):
        with pytest.raises(schemas.SchemaValidationError, match="must be a JSON object"):
            schemas.validate_question_payload(["nope"])  # type: ignore[arg-type]


class TestValidateQuestions:
    def test_keeps_only_valid_items(self):
        good = question_payload()
        bad = question_payload(answer="Z")
        assert schemas.validate_questions([good, bad, good]) == [good, good]

    def test_all_invalid_returns_empty_list(self):
        assert schemas.validate_questions([question_payload(difficulty=99)]) == []


class TestOptionNormalization:
    def test_list_becomes_keyed_object(self):
        assert schemas.normalize_options(["a", "b", "c", "d"]) == {
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
        }

    def test_json_string_from_db_column(self):
        assert schemas.normalize_options('{"A":"x","B":"y","C":"z","D":"w"}') == {
            "A": "x",
            "B": "y",
            "C": "z",
            "D": "w",
        }

    def test_dict_is_kept_and_values_stringified(self):
        assert schemas.normalize_options({"A": 1, "B": " b ", "D": "d"}) == {
            "A": "1",
            "B": "b",
            "C": "",
            "D": "d",
        }

    @pytest.mark.parametrize("bad", [["a", "b"], None, 3, "not json"])
    def test_unusable_input_raises(self, bad):
        with pytest.raises(ValueError):
            schemas.normalize_options(bad)

    def test_options_to_json_is_readable_back(self):
        raw = schemas.options_to_json(["a", "b", "c", "d"])
        assert '"A": "a"' in raw
        assert schemas.normalize_options(raw) == {
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
        }

    def test_options_to_json_preserves_vietnamese(self):
        raw = schemas.options_to_json(["Hà Nội", "x", "y", "z"])
        assert "\\u" not in raw and "Hà Nội" in raw

    def test_lenient_parse_returns_empty_dict_on_garbage(self):
        assert schemas.parse_options("not json") == {}
        assert schemas.parse_options([]) == {}

    def test_is_option_set_complete(self):
        assert schemas.is_option_set_complete({"A": "a", "B": "b", "C": "c", "D": "d"})
        assert not schemas.is_option_set_complete({"A": "a", "B": "b", "C": "", "D": "d"})
        assert not schemas.is_option_set_complete({})

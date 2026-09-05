"""
Question schema loading, validation, and answer-option normalization.

This is the single source of truth for what "a valid question" means:

- ``schema.json`` defines the *generated* payload shape (options as a 4-item list).
- The ``questions.options`` DB column stores options as a JSON *object* keyed by
  A/B/C/D (see ``options_to_json``).

Historically generator.py, json_parser.py, generate_mock_data.py and app.py each
carried their own copy of one of these responsibilities, which let the two option
shapes drift. They all call into this module now.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from jsonschema import Draft202012Validator

from validation import OPTION_KEYS

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"

# Difficulty bounds, mirrored from schema.json for callers that do not need the
# full JSON-Schema machinery (e.g. SQL query parameters).
DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 5


class SchemaValidationError(ValueError):
    """Raised when a question payload does not satisfy schema.json."""


@lru_cache(maxsize=1)
def load_schema(schema_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load schema.json once per process and verify it is a valid JSON Schema.

    Args:
        schema_path: Optional override, mainly for tests and tooling.

    Raises:
        FileNotFoundError: if the schema file is missing.
        SchemaValidationError: if schema.json itself is malformed.
    """
    path = Path(schema_path) if schema_path is not None else SCHEMA_PATH
    if not path.exists():
        raise FileNotFoundError(f"schema.json not found: {path}")

    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"schema.json is not valid JSON: {exc}") from exc

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema raises several error types here
        raise SchemaValidationError(f"schema.json is not a valid JSON Schema: {exc}") from exc

    return schema


@lru_cache(maxsize=1)
def _validator_for(_) -> Draft202012Validator:
    """Compile (and cache) the validator for the default schema."""
    return Draft202012Validator(load_schema())


def validate_question_payload(payload: Dict[str, Any]) -> None:
    """
    Validate one question payload against schema.json.

    Args:
        payload: Parsed model output or hand-authored mock question.

    Raises:
        SchemaValidationError: listing every violation found, ordered by field path.
    """
    if not isinstance(payload, dict):
        raise SchemaValidationError("Question payload must be a JSON object")

    errors = sorted(_validator_for(True).iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        details = "; ".join(f"{_error_path(err)}: {err.message}" for err in errors)
        raise SchemaValidationError(f"Question payload failed schema validation: {details}")


def validate_questions(payloads: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter a batch of payloads down to the valid ones.

    Partial success is intentional: one malformed question from the LLM should not
    discard its well-formed siblings.
    """
    valid: List[Dict[str, Any]] = []
    for payload in payloads:
        try:
            validate_question_payload(payload)
        except SchemaValidationError:
            continue
        valid.append(payload)
    return valid


def required_fields() -> List[str]:
    """Return the required top-level fields declared by schema.json."""
    return list(load_schema().get("required", []))


def _error_path(error) -> str:
    """Render a jsonschema error path as a dotted field name ('' for root)."""
    parts = ".".join(str(token) for token in error.path)
    return parts or "<root>"


def options_to_json(options: Any) -> str:
    """
    Normalize question options into the canonical JSON object stored in the DB.

    Accepts either the schema-shaped list ``["a", "b", "c", "d"]`` or an already
    keyed dict ``{"A": "a", ...}``; anything else is rejected so bad generations
    never reach the database.
    """
    return json.dumps(normalize_options(options), ensure_ascii=False)


def normalize_options(options: Any) -> Dict[str, str]:
    """
    Convert stored/supplied options into ``{"A": ..., "B": ..., "C": ..., "D": ...}``.

    Returns a dict with empty strings for missing/blank entries so callers can
    detect an unusable question by checking ``all(values)``.

    Raises:
        ValueError: if options is neither a 4-item sequence nor a mapping.
    """
    if isinstance(options, str):
        # A JSON string straight from the DB column.
        try:
            options = json.loads(options)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Options JSON is malformed: {exc}") from exc

    if isinstance(options, dict):
        return {key: str(options.get(key, "")).strip() for key in OPTION_KEYS}

    if isinstance(options, (list, tuple)):
        if len(options) != len(OPTION_KEYS):
            raise ValueError(
                f"Options must contain exactly {len(OPTION_KEYS)} items, got {len(options)}"
            )
        return {key: str(value).strip() for key, value in zip(OPTION_KEYS, options)}

    raise ValueError(f"Unsupported options type: {type(options).__name__}")


def parse_options(options: Any) -> Dict[str, str]:
    """
    Lenient variant of :func:`normalize_options` for UI rendering.

    Never raises: returns an empty dict when options are unusable, which lets the
    Streamlit layer show a graceful "malformed question" message instead of
    crashing a rerun.
    """
    try:
        return normalize_options(options)
    except ValueError:
        return {}


def is_option_set_complete(option_map: Dict[str, str]) -> bool:
    """True when every A/B/C/D slot has non-empty text."""
    return bool(option_map) and all(option_map.get(key) for key in OPTION_KEYS)

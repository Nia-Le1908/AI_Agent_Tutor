"""
JSON hardening, parsing and validation for LLM-generated question payloads.

Required interface from spec:
- parse_and_insert(json_str)

Parsing behaviour:
1. Strip markdown code fences the model may add despite instructions.
2. Extract balanced JSON object(s) from surrounding prose.
3. Raise KeyError when a required field is missing (explicit, per spec).
4. Validate strictly against schema.json (extra fields rejected).
5. Insert the validated question through the repository layer.

The extraction helpers are public because ``generator.py`` needs the same
defensive parsing for both single objects and batches — keeping one copy means a
model that wraps output in ```json fences behaves identically in both paths.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import db_manager
import schemas
from schemas import options_to_json  # noqa: F401  (re-exported for existing callers)

__all__ = [
    "strip_markdown_fences",
    "extract_first_json_object",
    "extract_all_json_objects",
    "safe_parse_json",
    "safe_parse_json_list",
    "validate_required_fields",
    "validate_payload",
    "parse_and_insert",
    "options_to_json",
]


def strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences if the model ignored instructions.

    Defensive by design: even with strict prompting, LLMs occasionally return
    output wrapped in ```json ... ``` blocks.
    """
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _scan_balanced(text: str, start: int, open_char: str, close_char: str) -> str:
    """
    Return the balanced ``open_char...close_char`` span starting at ``start``.

    String literals and escapes inside them are respected, so braces/brackets in
    quoted text never confuse the depth counter.
    """
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError(f"Unbalanced JSON {open_char}{close_char} block in model output")


def extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from text.

    Allows recovery when extra prose appears before/after the JSON payload.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    return _scan_balanced(text, start, "{", "}")


def _extract_json_array(text: str) -> List[Dict[str, Any]] | None:
    """
    Parse a leading balanced JSON array of objects from ``text``.

    Returns None unless the array actually holds objects: a nested ``options``
    array can otherwise be mistaken for the batch payload, which would silently
    turn a valid response into an empty one.
    """
    start = text.find("[")
    if start == -1:
        return None

    try:
        parsed = json.loads(_scan_balanced(text, start, "[", "]"))
    except (ValueError, json.JSONDecodeError):
        return None

    if not isinstance(parsed, list):
        return None

    items = [item for item in parsed if isinstance(item, dict)]
    return items or None


def extract_all_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract every JSON object from model output.

    Prefers a well-formed JSON array (the format the batch prompt requests) and
    falls back to positional brace scanning for models that emit a sequence of
    loose objects, or an array wrapped in commentary.
    """
    cleaned = strip_markdown_fences(text)

    array = _extract_json_array(cleaned)
    if array is not None:
        return array

    objects: List[Dict[str, Any]] = []
    position = 0
    while position < len(cleaned):
        next_brace = cleaned.find("{", position)
        if next_brace == -1:
            break
        try:
            fragment = extract_first_json_object(cleaned[next_brace:])
            parsed = json.loads(fragment)
        except (ValueError, json.JSONDecodeError):
            position = next_brace + 1
            continue

        if isinstance(parsed, dict):
            objects.append(parsed)
        position = next_brace + len(fragment)

    return objects


def safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """Parse one model output into a dict after fence-stripping and extraction."""
    cleaned = strip_markdown_fences(raw_text)
    fragment = extract_first_json_object(cleaned)

    try:
        payload = json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Generated payload must be a JSON object")
    return payload


def safe_parse_json_list(raw_text: str) -> List[Dict[str, Any]]:
    """Parse model output that should contain several questions."""
    return extract_all_json_objects(raw_text)


def validate_required_fields(payload: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Raise KeyError for missing required fields, as requested by the spec.

    jsonschema can report this too, but callers rely on the explicit KeyError, so
    the check stays separate from schema validation.
    """
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise KeyError(f"Missing required field(s): {', '.join(missing)}")


def validate_payload(payload: Dict[str, Any]) -> None:
    """
    Strictly validate one payload against schema.json.

    Raises:
        SchemaValidationError: subclass of ValueError, listing all violations.
    """
    schemas.validate_question_payload(payload)


def parse_and_insert(json_str: str) -> int:
    """
    Parse an LLM JSON string, validate it, and insert it into ``questions``.

    Args:
        json_str: Raw JSON string produced by the LLM.

    Returns:
        int: Inserted question row id.

    Raises:
        KeyError: if a required field is missing.
        ValueError: if JSON is malformed or fails schema validation.
        db_manager.DatabaseError: if insertion fails.
    """
    if not isinstance(json_str, str) or not json_str.strip():
        raise ValueError("json_str must be a non-empty JSON string")

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON string: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object/dict")

    validate_required_fields(payload, schemas.required_fields())
    validate_payload(payload)

    return _insert_with_stable_id(payload)


def _insert_with_stable_id(payload: Dict[str, Any]) -> int:
    """
    Persist a validated payload, preserving the LLM-provided ``question_id``.

    Keeping the supplied id preserves stable identifiers between the generation
    and storage layers, which the mock-data tooling depends on.
    """
    row = {
        "content": str(payload["content"]).strip(),
        "difficulty": int(payload["difficulty"]),
        "subject": str(payload["subject"]).strip(),
        "options": options_to_json(payload["options"]),
        "answer": str(payload["answer"]).strip().upper(),
        "explanation": str(payload["explanation"]).strip(),
    }

    question_id = payload.get("question_id")
    if question_id is None:
        return db_manager.insert_returning_id("questions", row)

    row_with_id = {"id": int(question_id), **row}
    columns = ", ".join(row_with_id)
    placeholders = ", ".join("?" for _ in row_with_id)
    db_manager.execute(
        f"INSERT INTO questions ({columns}) VALUES ({placeholders})",
        tuple(row_with_id.values()),
    )
    return int(question_id)

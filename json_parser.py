"""
JSON parsing and validation for Gemini-generated question objects.

Required interface from spec:
- parse_and_insert(json_str)

Behavior:
1. Parse JSON string into a dictionary.
2. Raise KeyError if required fields are missing.
3. Strictly validate dict against schema.json (no extra fields allowed).
4. Insert validated question into table `questions`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator, ValidationError

from config import DB_PATH


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"


def _load_schema() -> Dict[str, Any]:
    """Load JSON schema from schema.json and validate schema structure."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.json not found: {SCHEMA_PATH}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def _validate_required_fields(payload: Dict[str, Any], required_fields: list[str]) -> None:
    """
    Raise KeyError for missing required fields, as requested in the spec.

    Even though jsonschema can report missing fields, this explicit check ensures
    the function surfaces the exact KeyError behavior expected by callers.
    """
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise KeyError(f"Missing required field(s): {', '.join(missing)}")


def _options_to_json(options: list[str]) -> str:
    """Convert options list [A,B,C,D] into canonical JSON object string."""
    option_map = {
        "A": options[0],
        "B": options[1],
        "C": options[2],
        "D": options[3],
    }
    return json.dumps(option_map, ensure_ascii=False)


def parse_and_insert(json_str: str) -> int:
    """
    Parse Gemini JSON output, strictly validate it, and insert into Questions.

    Args:
        json_str: Raw JSON string produced by LLM.

    Returns:
        int: Inserted question row id.

    Raises:
        KeyError: If required field is missing.
        ValueError: If JSON is malformed or fails schema validation.
        sqlite3.Error: If insertion fails.
    """
    if not isinstance(json_str, str) or not json_str.strip():
        raise ValueError("json_str must be a non-empty JSON string")

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON string: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object/dict")

    schema = _load_schema()
    required_fields = schema.get("required", [])
    _validate_required_fields(payload, required_fields)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        details = "; ".join(err.message for err in errors)
        raise ValueError(f"Schema validation failed: {details}")

    options_json = _options_to_json(payload["options"])

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        # We store LLM-provided question_id as row id when possible to preserve
        # stable IDs between generation and storage layers.
        cursor = conn.execute(
            """
            INSERT INTO questions (id, content, difficulty, subject, options, answer, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(payload["question_id"]),
                payload["content"].strip(),
                int(payload["difficulty"]),
                payload["subject"].strip(),
                options_json,
                payload["answer"].strip(),
                payload["explanation"].strip(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()

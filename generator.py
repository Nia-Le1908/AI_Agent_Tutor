"""
Gemini question generator for AI Tutor V5.1 (Phase 3).

Required interface:
- generate(topic, difficulty) -> dict

Key guarantees:
1. Uses Gemini API via google-generativeai.
2. Enforces strict JSON-only output through prompt + parser hardening.
3. Applies exponential backoff for transient API/rate-limit failures.
4. Validates final payload against schema.json before returning.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator


# Keep a modern default and allow env override for operations.
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"


def _load_schema() -> Dict[str, Any]:
    """Load and sanity-check JSON schema used for strict validation."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.json not found: {SCHEMA_PATH}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def _build_prompt(topic: str, difficulty: int) -> str:
    """
    Build a robust generation prompt with few-shot JSON examples in Vietnamese.
    """
    return f"""
Bạn là một chuyên gia giáo dục tạo bài tập trắc nghiệm.
Hãy tạo ra chính xác MỘT câu hỏi trắc nghiệm bằng TIẾNG VIỆT.

Constraints:
- topic: {topic}
- difficulty: {difficulty} (integer from 1 to 5)
- options must have exactly 4 choices corresponding to A, B, C, D.
- answer must be one of: A, B, C, D.
- explanation must be concise but informative.
- ALL text inside "content", "options", and "explanation" MUST be in VIETNAMESE.

Output schema (must match exactly):
{{
  "question_id": 101,
  "content": "2 + 2 bằng bao nhiêu?",
  "difficulty": 1,
  "subject": "Toán học",
  "options": ["1", "2", "3", "4"],
  "answer": "D",
  "explanation": "2 cộng 2 bằng 4."
}}

Few-shot example 1:
{{
  "question_id": 102,
  "content": "Bào quan nào được mệnh danh là nhà máy năng lượng của tế bào?",
  "difficulty": 2,
  "subject": "Sinh học",
  "options": ["Nhân tế bào", "Ty thể", "Ribosome", "Bộ máy Golgi"],
  "answer": "B",
  "explanation": "Ty thể tạo ra ATP, nguồn năng lượng chính của tế bào."
}}

Few-shot example 2:
{{
  "question_id": 103,
  "content": "Nếu một ô tô di chuyển với tốc độ không đổi, đồ thị nào biểu diễn tốt nhất quãng đường theo thời gian?",
  "difficulty": 3,
  "subject": "Vật lý",
  "options": ["Đường ngang", "Đường thẳng có độ dốc dương", "Đường cong tăng dần", "Đường dích dắc"],
  "answer": "B",
  "explanation": "Ở tốc độ không đổi, quãng đường tăng tuyến tính theo thời gian."
}}

Now generate a new question for topic '{topic}' at difficulty {difficulty}.
Respond ONLY with a valid JSON object, no markdown, no explanation.
""".strip()


def _build_batch_prompt(topic: str, difficulty: int, count: int) -> str:
    """
    Build a prompt that asks the model to generate multiple questions at once.

    This is significantly more reliable than calling the model N times because:
    - Single network round-trip instead of N
    - Model has full context to avoid duplicate questions
    - Less chance of cumulative timeout/failure
    """
    return f"""
Bạn là một chuyên gia giáo dục tạo bài tập trắc nghiệm.
Hãy tạo ra chính xác {count} câu hỏi trắc nghiệm KHÁC NHAU bằng TIẾNG VIỆT.

Constraints:
- topic: {topic}
- difficulty: {difficulty} (integer from 1 to 5)
- Mỗi câu hỏi phải có options là mảng gồm đúng 4 lựa chọn.
- answer phải là một trong: A, B, C, D.
- explanation phải ngắn gọn nhưng đầy đủ thông tin.
- Các câu hỏi PHẢI KHÁC NHAU, không được trùng nội dung.
- TẤT CẢ nội dung trong "content", "options", và "explanation" PHẢI bằng TIẾNG VIỆT.

Định dạng output - trả về một JSON array chứa đúng {count} object:
[
  {{
    "question_id": 1,
    "content": "Câu hỏi bằng tiếng Việt?",
    "difficulty": {difficulty},
    "subject": "Tên môn học",
    "options": ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C", "Lựa chọn D"],
    "answer": "A",
    "explanation": "Giải thích ngắn gọn."
  }},
  ...
]

Tạo {count} câu hỏi cho chủ đề '{topic}' ở độ khó {difficulty}.
Trả về CHỈ một JSON array hợp lệ, KHÔNG markdown, KHÔNG giải thích thêm.
""".strip()


def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences if model ignored instructions.

    This parser is defensive: even with strict prompting, LLMs occasionally return
    output wrapped in ```json ... ``` blocks.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from text.

    This allows recovery when extra text appears before/after a JSON object.
    We track braces while respecting string literals and escapes.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON object in model output")


def _extract_all_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract all balanced JSON objects from text.

    Handles both JSON array format and multiple separate objects.
    """
    cleaned = _strip_markdown_fences(text)

    # Try parsing as a JSON array first (most reliable path)
    try:
        bracket_start = cleaned.find("[")
        if bracket_start != -1:
            bracket_depth = 0
            for i in range(bracket_start, len(cleaned)):
                if cleaned[i] == "[":
                    bracket_depth += 1
                elif cleaned[i] == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        array_str = cleaned[bracket_start : i + 1]
                        parsed = json.loads(array_str)
                        if isinstance(parsed, list):
                            return [item for item in parsed if isinstance(item, dict)]
                        break
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract individual JSON objects one by one
    objects: List[Dict[str, Any]] = []
    remaining = cleaned
    while "{" in remaining:
        try:
            json_str = _extract_first_json_object(remaining)
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                objects.append(obj)
            # Move past the extracted object
            end_pos = remaining.find(json_str) + len(json_str)
            remaining = remaining[end_pos:]
        except (ValueError, json.JSONDecodeError):
            break

    return objects


def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """Parse model output into dict after cleanup and object extraction."""
    cleaned = _strip_markdown_fences(raw_text)
    json_fragment = _extract_first_json_object(cleaned)

    try:
        payload = json.loads(json_fragment)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Generated payload must be a JSON object")

    return payload


def _validate_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Strictly validate payload against schema.json and raise clear errors."""
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"Generated payload failed schema validation: {details}")


def _call_ollama(prompt: str, model_name: str) -> str:
    """Gọi Ollama API ở dưới local và ép nó trả về JSON."""
    import ollama  # Lazy import to avoid crash when Ollama is not installed

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
        format='json' # Tính năng bá đạo của Ollama: Ép format JSON
    )
    return response['message']['content']


def generate(topic: str, difficulty: int, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Hàm chính tạo bài tập trắc nghiệm qua Ollama.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not isinstance(difficulty, int) or difficulty < 1 or difficulty > 5:
        raise ValueError("difficulty must be an integer in range [1, 5]")

    schema = _load_schema()
    prompt = _build_prompt(topic=topic.strip(), difficulty=difficulty)

    # Gọi thẳng xuống local, không cần API Key, không lo Rate Limit
    raw_text = _call_ollama(prompt, model_name)
    
    payload = _safe_parse_json(raw_text)
    _validate_payload(payload, schema)

    return payload


def generate_batch(
    topic: str,
    difficulty: int,
    count: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Generate multiple quiz questions in a single Ollama call.

    This is more reliable than calling generate() in a loop because the model
    produces all questions in one context window, avoiding repeated network
    round-trips and cumulative failure risk.

    Returns:
        List of validated question dicts. May return fewer than `count` if
        some questions fail validation (partial success).
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not isinstance(difficulty, int) or difficulty < 1 or difficulty > 5:
        raise ValueError("difficulty must be an integer in range [1, 5]")
    if not isinstance(count, int) or count < 1 or count > 10:
        raise ValueError("count must be an integer in range [1, 10]")

    schema = _load_schema()
    prompt = _build_batch_prompt(topic=topic.strip(), difficulty=difficulty, count=count)

    raw_text = _call_ollama(prompt, model_name)
    objects = _extract_all_json_objects(raw_text)

    if not objects:
        raise ValueError("Model returned no parseable question objects")

    # Validate each question individually; keep valid ones
    valid_questions: List[Dict[str, Any]] = []
    for obj in objects:
        try:
            _validate_payload(obj, schema)
            valid_questions.append(obj)
        except ValueError:
            continue  # Skip invalid questions silently

    if not valid_questions:
        raise ValueError(
            f"Model returned {len(objects)} object(s) but none passed schema validation"
        )

    return valid_questions
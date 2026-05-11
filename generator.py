"""
Question generator for AI Tutor V5.1.

Provider: DeepSeek API (cố định), tương thích OpenAI SDK.
Yêu cầu: DEEPSEEK_API_KEY trong file .env.

Required interface:
- generate(topic, difficulty) -> dict
- generate_batch(topic, difficulty, count) -> list[dict]

Key guarantees:
1. Enforces strict JSON-only output through prompt + parser hardening.
2. Validates final payload against schema.json before returning.
3. Falls back to individual calls if batch mode fails.
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


# Default model read from env; falls back to DeepSeek's flagship chat model.
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
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

!!! QUAN TRỌNG NHẤT: Câu hỏi BẮT BUỘC phải thuộc chủ đề "{topic}".
KHÔNG được tạo câu hỏi về chủ đề khác. Trường "subject" phải là "{topic}".

Constraints:
- topic (BẮT BUỘC): {topic}
- difficulty: {difficulty} (integer from 1 to 5)
- "content" phải là câu hỏi kiến thức trực tiếp về chủ đề "{topic}".
- "subject" phải bằng đúng "{topic}".
- options phải có đúng 4 lựa chọn tương ứng A, B, C, D.
- answer phải là một trong: A, B, C, D.
- explanation phải ngắn gọn nhưng đầy đủ thông tin.
- TẤT CẢ nội dung trong "content", "options", "explanation" phải bằng TIẾNG VIỆT.

Ví dụ cấu trúc JSON (đây chỉ là VÍ DỤ cấu trúc, KHÔNG copy nội dung):
{{
  "question_id": 101,
  "content": "<câu hỏi về {topic}>",
  "difficulty": {difficulty},
  "subject": "{topic}",
  "options": ["<lựa chọn A>", "<lựa chọn B>", "<lựa chọn C>", "<lựa chọn D>"],
  "answer": "A",
  "explanation": "<giải thích tại sao đáp án đúng>"
}}

Bây giờ hãy tạo MỘT câu hỏi về chủ đề "{topic}" ở độ khó {difficulty}.
Trả về CHỈ một JSON object hợp lệ, KHÔNG markdown, KHÔNG giải thích thêm.
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

!!! QUAN TRỌNG NHẤT: TẤT CẢ {count} câu hỏi BẮT BUỘC phải thuộc chủ đề "{topic}".
KHÔNG được tạo câu hỏi về chủ đề khác. Trường "subject" của mỗi câu phải là "{topic}".

Constraints:
- topic (BẮT BUỘC cho tất cả câu hỏi): {topic}
- difficulty: {difficulty} (integer from 1 to 5)
- "content" của mỗi câu phải là kiến thức trực tiếp về "{topic}".
- "subject" của mỗi câu phải bằng đúng "{topic}".
- Mỗi câu hỏi phải có options là mảng gồm đúng 4 lựa chọn.
- answer phải là một trong: A, B, C, D.
- explanation phải ngắn gọn nhưng đầy đủ thông tin.
- Các câu hỏi PHẢI KHÁC NHAU về nội dung, không được trùng lặp.
- TẤT CẢ nội dung trong "content", "options", "explanation" PHẢI bằng TIẾNG VIỆT.

Định dạng output - trả về một JSON array chứa đúng {count} object:
[
  {{
    "question_id": 1,
    "content": "<câu hỏi về {topic}>",
    "difficulty": {difficulty},
    "subject": "{topic}",
    "options": ["<lựa chọn A>", "<lựa chọn B>", "<lựa chọn C>", "<lựa chọn D>"],
    "answer": "A",
    "explanation": "<giải thích>"
  }}
]

Tạo đúng {count} câu hỏi về chủ đề "{topic}" ở độ khó {difficulty}.
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
            in_str = False
            esc = False
            for i in range(bracket_start, len(cleaned)):
                ch = cleaned[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        array_str = cleaned[bracket_start : i + 1]
                        parsed = json.loads(array_str)
                        if isinstance(parsed, list):
                            return [item for item in parsed if isinstance(item, dict)]
                        break
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract individual JSON objects one by one using positional scanning
    objects: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(cleaned):
        next_brace = cleaned.find("{", pos)
        if next_brace == -1:
            break
        try:
            json_str = _extract_first_json_object(cleaned[next_brace:])
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                objects.append(obj)
            pos = next_brace + len(json_str)
        except (ValueError, json.JSONDecodeError):
            pos = next_brace + 1

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


def _call_llm(prompt: str, model_name: str, *, json_mode: bool = False) -> str:
    """
    Gọi DeepSeek API và trả về nội dung text.

    Args:
        prompt   : Prompt gửi đến model.
        model_name: Tên model (mặc định: deepseek-chat).
        json_mode : Nếu True, ép model trả về JSON object.
    """
    from openai import OpenAI
    from config import DEEPSEEK_BASE_URL, require_deepseek_api_key

    client = OpenAI(
        api_key=require_deepseek_api_key(),
        base_url=DEEPSEEK_BASE_URL,
    )
    kwargs: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def generate(topic: str, difficulty: int, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Hàm chính tạo một câu hỏi trắc nghiệm qua LLM provider đang cấu hình.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not isinstance(difficulty, int) or difficulty < 1 or difficulty > 5:
        raise ValueError("difficulty must be an integer in range [1, 5]")

    schema = _load_schema()
    prompt = _build_prompt(topic=topic.strip(), difficulty=difficulty)

    raw_text = _call_llm(prompt, model_name, json_mode=True)

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
    Generate multiple quiz questions via the configured LLM provider.

    Falls back to calling generate() individually if the model returns
    a single object instead of a JSON array.

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

    raw_text = _call_llm(prompt, model_name, json_mode=True)
    objects = _extract_all_json_objects(raw_text)

    # Validate each question individually; keep valid ones
    valid_questions: List[Dict[str, Any]] = []
    for obj in objects:
        try:
            _validate_payload(obj, schema)
            valid_questions.append(obj)
        except ValueError:
            continue  # Skip invalid questions silently

    # If batch call yielded results, return them
    if valid_questions:
        return valid_questions

    # Fallback: model returned a single object or failed array format.
    # Call generate() individually for each question requested.
    errors: List[str] = []
    for i in range(count):
        try:
            q = generate(topic=topic.strip(), difficulty=difficulty, model_name=model_name)
            valid_questions.append(q)
        except Exception as exc:
            errors.append(str(exc))

    if not valid_questions:
        err_summary = "; ".join(errors[:3]) if errors else "No questions generated"
        raise ValueError(
            f"generate_batch failed: batch call returned no valid objects and "
            f"individual fallback also failed. Last errors: {err_summary}"
        )

    return valid_questions
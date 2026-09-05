"""
Question generator for AI Tutor V5.1.

Provider: the OpenAI-compatible LLM configured in :mod:`config` (DeepSeek by
default); the request itself (and its retry policy) lives in :mod:`llm_client`.

Required interface:
- generate(topic, difficulty) -> dict
- generate_batch(topic, difficulty, count) -> list[dict]

Guarantees:
1. Strict JSON-only output enforced by the prompt plus defensive parsing.
2. Every returned payload is validated against schema.json.
3. Batch mode falls back to single questions when the model ignores the array format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import json_parser
import llm_client
import schemas
from config import DEFAULT_MODEL, MAX_BATCH_SIZE
from validation import require_int_in_range, require_level, require_non_empty_str

logger = logging.getLogger(__name__)

# Re-exported so `from generator import DEFAULT_MODEL` keeps working.
__all__ = ["DEFAULT_MODEL", "generate", "generate_batch", "build_prompt", "build_batch_prompt"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def build_prompt(topic: str, difficulty: int) -> str:
    """Build the single-question generation prompt (Vietnamese, JSON-only)."""
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


def build_batch_prompt(topic: str, difficulty: int, count: int) -> str:
    """
    Build a prompt asking for several questions in one response.

    Preferable to N separate calls: one network round trip, and the model can see
    its own earlier questions so it avoids duplicates.
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _request_json(prompt: str, model_name: str) -> str:
    """Call the LLM asking for a JSON response, returning the raw text."""
    return llm_client.chat(prompt, model=model_name, json_mode=True)


def _normalize_answer_key(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uppercase the answer letter in-place when the model returns "a"/" d".

    schema.json only accepts A-D, so repairing obvious casing noise before
    validation saves a wasted round trip.
    """
    answer = payload.get("answer")
    if isinstance(answer, str):
        payload["answer"] = answer.strip().upper()
    return payload


def generate(topic: str, difficulty: int, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Generate one multiple-choice question as a schema-validated dict.

    Args:
        topic: Required subject of the question.
        difficulty: Integer in [1, 5].
        model_name: Provider model identifier.

    Returns:
        The validated question payload (schema.json shape).

    Raises:
        ValueError: invalid topic/difficulty, unparsable or invalid model output.
        llm_client.LLMError: provider unavailable after retries.
    """
    topic = require_non_empty_str(topic, "topic")
    difficulty = require_level(difficulty, "difficulty")

    raw_text = _request_json(build_prompt(topic, difficulty), model_name)
    payload = _normalize_answer_key(json_parser.safe_parse_json(raw_text))
    schemas.validate_question_payload(payload)
    return payload


def generate_batch(
    topic: str,
    difficulty: int,
    count: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Generate several questions in one call, falling back to single generations.

    Returns:
        Validated question dicts. May contain fewer than ``count`` items: one
        malformed question must not discard its well-formed siblings.

    Raises:
        ValueError: invalid arguments, or nothing valid was produced at all.
        llm_client.LLMError: provider unavailable after retries.
    """
    topic = require_non_empty_str(topic, "topic")
    difficulty = require_level(difficulty, "difficulty")
    count = _require_batch_count(count)

    raw_text = _request_json(build_batch_prompt(topic, difficulty, count), model_name)
    candidates = json_parser.safe_parse_json_list(raw_text)
    valid_questions = schemas.validate_questions(_normalize_answer_key(q) for q in candidates)

    if valid_questions:
        return valid_questions

    logger.warning(
        "Batch generation returned no valid questions (%d candidates); falling back to %d single calls",
        len(candidates),
        count,
    )
    return _generate_individually(topic, difficulty, count, model_name)


def _require_batch_count(count: int) -> int:
    """Clamp-and-check the batch size against the configured provider ceiling."""
    return require_int_in_range(count, "count", 1, MAX_BATCH_SIZE)


def _generate_individually(
    topic: str,
    difficulty: int,
    count: int,
    model_name: str,
) -> List[Dict[str, Any]]:
    """Fallback path: ask for questions one at a time and keep the valid ones."""
    valid_questions: List[Dict[str, Any]] = []
    errors: List[str] = []

    for _ in range(count):
        try:
            valid_questions.append(generate(topic=topic, difficulty=difficulty, model_name=model_name))
        except Exception as exc:  # noqa: BLE001 - partial success is intentional
            errors.append(str(exc))

    if not valid_questions:
        summary = "; ".join(errors[:3]) if errors else "No questions generated"
        raise ValueError(
            "generate_batch failed: batch call returned no valid objects and "
            f"individual fallback also failed. Last errors: {summary}"
        )

    return valid_questions

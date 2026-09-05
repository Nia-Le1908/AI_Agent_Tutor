"""
Orchestration layer for AI Tutor.

Responsibilities:
1. Chat flow: user input -> retrieve context -> build prompt -> LLM -> answer (+ citations).
2. Exercise flow: compute adaptive difficulty -> generate strict JSON question.
3. Answer flow: grade an answer, persist it, and return what the UI needs to show.

Provider retries live in :mod:`llm_client`; SQL lives in :mod:`sqlite_manager`.
This module stays framework-agnostic so Streamlit (or a REST layer later) can call
it directly without adapters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import llm_client
import retriever
import sqlite_manager
from adaptive_logic import get_next_difficulty
from config import DEBUG_RAG_CONTEXT, DEFAULT_MODEL, TOP_K
from generator import generate
from validation import require_non_empty_str, require_positive_int

logger = logging.getLogger(__name__)

MAX_LOGGED_CHUNKS = 3
CHUNK_PREVIEW_CHARS = 500

# Number of retrieved chunks injected into the rate-limit fallback answer.
FALLBACK_CONTEXT_CHUNKS = 2

_ANSWER_INSTRUCTIONS = (
    "You are AI Tutor V5.1. Provide accurate, concise, student-friendly answers. "
    "Use the provided context when relevant. If context is insufficient, clearly say so."
)

_NO_CONTEXT_NOTE = (
    "No external context found. Answer with best effort and state uncertainty when needed."
)

CITATION_HEADING = "\n\n---\n📚 **Nguồn tham khảo:** "


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_chat_prompt(user_input: str, context_chunks: List[str]) -> str:
    """
    Compose the RAG prompt from the question plus retrieved chunks.

    Context blocks are labelled so grounding is easy to audit in logs and so a
    human can tell which chunk the model was leaning on.
    """
    context_text = "\n\n".join(
        f"[Context {index + 1}] {chunk}" for index, chunk in enumerate(context_chunks)
    ) or _NO_CONTEXT_NOTE

    return (
        f"{_ANSWER_INSTRUCTIONS}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Student question:\n{user_input}\n\n"
        "Answer in clear plain text."
    )


def build_quota_fallback_response(context_chunks: List[str]) -> str:
    """
    Graceful answer when the provider is rate limited or out of quota.

    Retrieval results are local, so the app keeps delivering learning value
    instead of surfacing a hard error to students.
    """
    header = "LLM API is currently unavailable due to quota/rate-limit issues. "

    if not context_chunks:
        return (
            header
            + "I cannot generate a full model answer right now, and no local context "
            "was retrieved for this question. Please try again later or update API billing/quota."
        )

    top_context = "\n\n".join(context_chunks[:FALLBACK_CONTEXT_CHUNKS])
    return (
        header
        + "Below is the most relevant context retrieved from your documents so you can continue learning:\n\n"
        f"{top_context}\n\n"
        "Note: This is a retrieval-only fallback (not a generated explanation)."
    )


def _log_retrieved_chunks(chunks: List[str]) -> None:
    """Log chunk previews when DEBUG_RAG_CONTEXT is on; off by default to stay quiet."""
    if not DEBUG_RAG_CONTEXT:
        return

    logger.warning("RAG debug enabled. Retrieved %d chunk(s).", len(chunks))
    for index, chunk in enumerate(chunks[:MAX_LOGGED_CHUNKS], start=1):
        logger.warning("[RAG chunk %d] %s", index, " ".join(chunk.split())[:CHUNK_PREVIEW_CHARS])


# ---------------------------------------------------------------------------
# Retrieval + citations
# ---------------------------------------------------------------------------
def retrieve_context(query: str, top_k: int) -> tuple[List[str], List[str]]:
    """
    Retrieve grounding chunks plus the distinct source files they came from.

    Falls back to text-only retrieval when source metadata is unavailable, which
    keeps chat working for indexes built before citations existed.
    """
    try:
        hits = retriever.retrieve_with_sources(query=query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - retrieval trouble must not block chat
        logger.warning("Source-aware retrieval unavailable (%s); using plain retrieval", exc)
        return retriever.retrieve(query=query, top_k=top_k), []

    chunks = [hit["text"] for hit in hits]
    # dict.fromkeys: dedupe while preserving retrieval order.
    sources = list(dict.fromkeys(hit["source_file"] for hit in hits))
    return chunks, sources


def append_citations(answer: str, sources: List[str]) -> str:
    """Append the citation footer when sources are known."""
    if not sources:
        return answer
    formatted = ", ".join(f"*{source}*" for source in sources)
    return f"{answer}{CITATION_HEADING}{formatted}"


def chat(user_input: str, top_k: int = TOP_K, model_name: str = DEFAULT_MODEL) -> str:
    """
    Answer a student question with RAG grounding and citations.

    Flow: retrieve -> build prompt -> generate via LLM (with retries) -> cite.

    Args:
        user_input: Non-empty question text.
        top_k: Number of retrieved chunks injected into the prompt.
        model_name: LLM model identifier.

    Returns:
        Assistant answer text (a retrieval-only fallback under rate limits).

    Raises:
        ValueError: for invalid input.
        llm_client.LLMError: provider failures that are not rate limits.
    """
    question = require_non_empty_str(user_input, "user_input")
    require_positive_int(top_k, "top_k")

    context_chunks, sources = retrieve_context(question, top_k)
    _log_retrieved_chunks(context_chunks)

    prompt = build_chat_prompt(question, context_chunks)
    try:
        answer = llm_client.chat(prompt, model=model_name)
    except llm_client.LLMRateLimitError as exc:
        logger.warning("LLM quota/rate-limit encountered, using retrieval fallback: %s", exc)
        return build_quota_fallback_response(context_chunks)

    return append_citations(answer, sources)


# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
def generate_exercise_for_user(
    uid: int,
    topic: str,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Generate one adaptive exercise for a user at their computed difficulty level.

    Raises:
        ValueError: for invalid uid/topic.
        llm_client.LLMError: when generation fails.
    """
    require_positive_int(uid, "uid")
    topic = require_non_empty_str(topic, "topic")

    difficulty = get_next_difficulty(uid)
    return generate(topic=topic, difficulty=difficulty, model_name=model_name)


def grade_answer(question: Dict[str, Any], selected_answer: str) -> bool:
    """True when the selected option letter matches the stored answer."""
    return str(selected_answer).strip().upper() == str(question.get("answer", "")).strip().upper()


def record_answer(
    uid: int,
    question: Dict[str, Any],
    selected_answer: str,
) -> Dict[str, Any]:
    """
    Grade a submitted answer, persist it, and recompute the adaptive level.

    Living in the orchestrator keeps the UI out of grading rules and SQL.

    Returns:
        {
          "is_correct": bool,
          "correct_answer": str,
          "explanation": str,
          "new_level": int,
        }

    Raises:
        ValueError: for an invalid uid or a question without an id.
        db_manager.DatabaseError: when the answer cannot be stored.
    """
    uid = require_positive_int(uid, "uid")
    question_id = require_positive_int(question.get("id"), "question id")

    is_correct = grade_answer(question, selected_answer)
    sqlite_manager.save_history(uid=uid, qid=question_id, is_correct=is_correct)

    # Recomputed immediately so the next question loaded already reflects the new level.
    new_level = get_next_difficulty(uid)

    return {
        "is_correct": is_correct,
        "correct_answer": str(question.get("answer", "")).strip().upper(),
        "explanation": str(question.get("explanation") or "").strip(),
        "new_level": new_level,
    }

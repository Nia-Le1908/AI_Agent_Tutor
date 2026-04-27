"""
Main orchestration logic for AI Tutor V5.1 (Phase 3).

Responsibilities:
1. Chat flow: user input -> retrieve context -> prompt Gemini -> answer text.
2. Exercise flow: compute adaptive difficulty -> generate strict JSON question.
3. Apply exponential backoff around Gemini calls to handle rate limits/transients.

Notes:
- This module intentionally stays framework-agnostic so Streamlit can call it
  directly without additional adapters.
"""

from __future__ import annotations

import importlib
import logging
import os
import random
import time
from typing import Any, Dict, List

from adaptive_logic import get_next_difficulty
from config import TOP_K, require_gemini_api_key
from generator import DEFAULT_MODEL, generate
from retriever import retrieve, retrieve_with_sources


logger = logging.getLogger(__name__)


def _is_rag_debug_enabled() -> bool:
    """Enable retrieval debug logs via env var DEBUG_RAG_CONTEXT=true."""
    raw = os.getenv("DEBUG_RAG_CONTEXT", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _log_retrieved_chunks(chunks: List[str]) -> None:
    """
    Log the top retrieved chunks to help debug RAG quality issues.

    This is intentionally opt-in to avoid noisy logs in normal runs.
    """
    if not _is_rag_debug_enabled():
        return

    logger.warning("RAG debug enabled. Retrieved %d chunk(s).", len(chunks))

    for idx, chunk in enumerate(chunks[:3], start=1):
        preview = " ".join(chunk.split())[:500]
        logger.warning("[RAG chunk %d] %s", idx, preview)


def _build_chat_prompt(user_input: str, context_chunks: List[str]) -> str:
    """
    Build RAG prompt from user question and retrieved document chunks.

    We separate context blocks clearly so model grounding is easier to follow.
    """
    context_text = "\n\n".join(
        [f"[Context {idx + 1}] {chunk}" for idx, chunk in enumerate(context_chunks)]
    )

    if not context_text:
        context_text = "No external context found. Answer with best effort and state uncertainty when needed."

    return f"""
You are AI Tutor V5.1. Provide accurate, concise, student-friendly answers.
Use the provided context when relevant. If context is insufficient, clearly say so.

Context:
{context_text}

Student question:
{user_input}

Answer in clear plain text.
""".strip()


def _build_quota_fallback_response(user_input: str, context_chunks: List[str]) -> str:
    """
    Build a graceful fallback response when Gemini quota is exceeded.

    The app remains usable by surfacing the most relevant retrieved context
    instead of throwing a hard error to end users.
    """
    if not context_chunks:
        return (
            "LLM API is currently unavailable due to quota/rate-limit issues. "
            "I cannot generate a full model answer right now, and no local context "
            "was retrieved for this question. Please try again later or update API billing/quota."
        )

    top_context = "\n\n".join(context_chunks[:2])
    return (
        "LLM API is currently unavailable due to quota/rate-limit issues. "
        "Below is the most relevant context retrieved from your documents so you can continue learning:\n\n"
        f"{top_context}\n\n"
        "Note: This is a retrieval-only fallback (not a generated explanation)."
    )


def _resolve_provider(model_name: str) -> str:
    """
    Resolve LLM provider from env or model-name heuristic.

    Priority:
    1) LLM_PROVIDER env var if set (ollama/gemini)
    2) model name prefix heuristic
    """
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_provider in {"ollama", "gemini"}:
        return env_provider

    return "gemini" if model_name.strip().lower().startswith("gemini") else "ollama"


def _generate_text_with_backoff(
    prompt: str,
    *,
    model_name: str = DEFAULT_MODEL,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 20.0,
) -> str:
    """
    Send prompt to configured LLM provider with exponential backoff.

    This is used by chatbot flow (separate from generator.py which focuses on
    strict JSON question generation).
    """
    provider = _resolve_provider(model_name)
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            if provider == "ollama":
                import ollama

                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.get("message", {}).get("content", "") or "").strip()
            else:
                genai = importlib.import_module("google.generativeai")

                api_key = require_gemini_api_key()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = (response.text or "").strip()

            if not text:
                raise ValueError(f"{provider} returned empty text")
            return text
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break

            delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
            jitter = random.uniform(0.0, 0.35 * delay)
            time.sleep(delay + jitter)

    raise RuntimeError(
        f"{provider.upper()} chat request failed after {max_attempts} attempts: {last_error}"
    )


def chat(user_input: str, top_k: int = TOP_K, model_name: str = DEFAULT_MODEL) -> str:
    """
    Main chatbot flow with RAG citation support.

    Flow:
    1) Retrieve relevant chunks with source metadata from FAISS.
    2) Build grounded prompt.
    3) Generate answer via LLM with exponential backoff.
    4) Append source citations to the answer.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Retrieve with source metadata for citations
    try:
        retrieved = retrieve_with_sources(query=user_input.strip(), top_k=top_k)
        context_chunks = [r["text"] for r in retrieved]
        source_files = list(dict.fromkeys(r["source_file"] for r in retrieved))  # unique, ordered
    except Exception:
        # Fallback to basic retrieval if sources not available
        context_chunks = retrieve(query=user_input.strip(), top_k=top_k)
        source_files = []

    _log_retrieved_chunks(context_chunks)
    prompt = _build_chat_prompt(user_input.strip(), context_chunks)
    try:
        answer = _generate_text_with_backoff(prompt=prompt, model_name=model_name)
        # Append citation footer if sources are available
        if source_files:
            citation = "\n\n---\n📚 **Nguồn tham khảo:** " + ", ".join(f"*{s}*" for s in source_files)
            answer += citation
        return answer
    except RuntimeError as exc:
        message = str(exc).lower()
        if "429" in message or "quota" in message or "rate limit" in message:
            logger.warning("LLM quota/rate-limit encountered, using retrieval fallback: %s", exc)
            return _build_quota_fallback_response(user_input.strip(), context_chunks)
        raise


def generate_exercise_for_user(
    uid: int,
    topic: str,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Generate one adaptive exercise for a specific user.

    Flow:
    1) Compute next difficulty using answer history.
    2) Generate strict JSON question via generator.generate().
    """
    if not isinstance(uid, int) or uid <= 0:
        raise ValueError("uid must be a positive integer")

    difficulty = get_next_difficulty(uid)
    return generate(topic=topic, difficulty=difficulty, model_name=model_name)

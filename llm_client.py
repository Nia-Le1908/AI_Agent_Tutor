"""
Single LLM access point for AI Tutor (OpenAI-compatible providers, DeepSeek default).

Before this module, ``generator.py`` and ``controller.py`` each built their own
OpenAI client, set temperature, and (in one case) implemented retries. That left
two different reliability profiles for the same API: chat calls retried with
backoff, question generation did not — even though the documented contract promised
backoff for both.

Guarantees:
- one client construction path, one place to change provider settings;
- exponential backoff with jitter on every transient failure;
- typed errors so callers can branch on rate limits instead of string matching.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

from config import (
    DEEPSEEK_BASE_URL,
    DEFAULT_MODEL,
    LLM_BACKOFF_FACTOR,
    LLM_INITIAL_BACKOFF_SECONDS,
    LLM_MAX_ATTEMPTS,
    LLM_MAX_BACKOFF_SECONDS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    require_deepseek_api_key,
)

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("429", "rate limit", "ratelimit", "quota", "too many requests")


class LLMError(RuntimeError):
    """Raised when a chat completion could not be produced."""


class LLMRateLimitError(LLMError):
    """Raised when retries were exhausted on quota/rate-limit responses."""


class LLMConfigurationError(LLMError):
    """Raised when the provider cannot be reached with the current settings."""


def is_rate_limit_error(error: BaseException | str) -> bool:
    """Heuristic provider-agnostic rate limit detection (also accepts messages)."""
    message = str(error).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def _build_client() -> Any:
    """
    Create an OpenAI-compatible client for the configured provider.

    The import stays local so the rest of the app (DB tools, dashboards, mock data)
    runs even on installs without the optional LLM SDK.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise LLMConfigurationError(
            "The 'openai' package is required for LLM features. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    return OpenAI(
        api_key=require_deepseek_api_key(),
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,  # retry policy is owned by this module, not the SDK
    )


def _next_delay(attempt: int) -> float:
    """Exponential backoff with jitter, capped at LLM_MAX_BACKOFF_SECONDS."""
    delay = min(LLM_INITIAL_BACKOFF_SECONDS * (LLM_BACKOFF_FACTOR ** attempt), LLM_MAX_BACKOFF_SECONDS)
    return delay + random.uniform(0.0, 0.35 * delay)


def _extract_text(response: Any) -> str:
    """Pull the assistant message text out of a chat completion response."""
    choices: Optional[List[Any]] = getattr(response, "choices", None)
    if not choices:
        raise LLMError("Provider returned no completion choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return (content or "").strip()


def chat(
    prompt: str = "",
    *,
    model: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    json_mode: bool = False,
    temperature: float = LLM_TEMPERATURE,
    max_attempts: int = LLM_MAX_ATTEMPTS,
) -> str:
    """
    Send one prompt to the configured provider and return the assistant text.

    Args:
        prompt: User-role prompt. Ignored when ``messages`` is provided.
        model: Model identifier; defaults to the configured DeepSeek model.
        messages: Full message list for callers that need a system prompt.
        json_mode: Ask the provider for a JSON object response.
        temperature: Sampling temperature.
        max_attempts: Number of tries before giving up (>= 1).

    Returns:
        Non-empty assistant text.

    Raises:
        ValueError: for invalid arguments.
        LLMRateLimitError: when the failure was a quota/rate-limit problem.
        LLMError: for any other exhausted failure, or an empty completion.
    """
    if messages is None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        messages = [{"role": "user", "content": prompt}]

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    client = _build_client()
    payload: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            text = _extract_text(client.chat.completions.create(**payload))
            if not text:
                raise LLMError("Provider returned empty text")
            return text
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            last_error = exc
            if attempt == max_attempts - 1:
                break
            delay = _next_delay(attempt)
            logger.warning(
                "LLM attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    message = f"LLM request failed after {max_attempts} attempts: {last_error}"
    if is_rate_limit_error(last_error):
        raise LLMRateLimitError(message) from last_error
    raise LLMError(message) from last_error

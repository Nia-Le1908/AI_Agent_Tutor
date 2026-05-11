"""
Centralized configuration for AI Tutor V5.1 (Phase 1).

This module intentionally exposes simple module-level constants so early-phase
scripts can import values directly, while still validating all critical inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when configuration is invalid or unsafe."""


# Load .env if present. Existing process env vars always take precedence.
load_dotenv()


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    """Read environment variable with optional required enforcement."""
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise ConfigError(f"Missing required environment variable: {name}")
    return (value or "").strip()


def _to_int(name: str, raw: str, min_value: int | None = None, max_value: int | None = None) -> int:
    """Convert env string to int and enforce optional bounds."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc

    if min_value is not None and value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"{name} must be <= {max_value}, got {value}")

    return value


# -------------------------
# Provider setting
# -------------------------
# Keep optional at import time so Phase 1/2 modules (DB/RAG tooling) can run
# even before LLM credentials are configured.
GEMINI_API_KEY: str = _env("GEMINI_API_KEY", default="")
DEEPSEEK_API_KEY: str = _env("DEEPSEEK_API_KEY", default="")
DEEPSEEK_MODEL: str = _env("DEEPSEEK_MODEL", default="deepseek-chat")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"


def require_gemini_api_key() -> str:
    """Return GEMINI_API_KEY or raise a clear error when LLM features are used."""
    if not GEMINI_API_KEY:
        raise ConfigError("GEMINI_API_KEY is required for Gemini generation features")
    return GEMINI_API_KEY


def require_deepseek_api_key() -> str:
    """Return DEEPSEEK_API_KEY or raise a clear error when DeepSeek features are used."""
    if not DEEPSEEK_API_KEY:
        raise ConfigError(
            "DEEPSEEK_API_KEY is required. Set it in .env: DEEPSEEK_API_KEY=sk-..."
        )
    return DEEPSEEK_API_KEY


# -------------------------
# Paths
# -------------------------
# Use Path objects for safe path operations, then expose string constants for
# compatibility with modules that expect plain strings.
PROJECT_ROOT = Path(__file__).resolve().parent

DB_PATH = str((PROJECT_ROOT / _env("DB_PATH", "data/ai_tutor_v5.db")).resolve())
FAISS_INDEX_PATH = str((PROJECT_ROOT / _env("FAISS_INDEX_PATH", "vector_store/faiss_index.bin")).resolve())
LOG_PATH = str((PROJECT_ROOT / _env("LOG_PATH", "logs/app.log")).resolve())


# -------------------------
# Retrieval and chunk tuning
# -------------------------
# Spec asks for chunk size 256-512 and overlap=50, top_k=3.
_chunk_size_raw = _env("CHUNK_SIZE", "256")
CHUNK_SIZE = _to_int("CHUNK_SIZE", _chunk_size_raw, min_value=256, max_value=512)

_chunk_overlap_raw = _env("CHUNK_OVERLAP", "50")
CHUNK_OVERLAP = _to_int("CHUNK_OVERLAP", _chunk_overlap_raw, min_value=0, max_value=512)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ConfigError("CHUNK_OVERLAP must be strictly smaller than CHUNK_SIZE")

_top_k_raw = _env("TOP_K", "3")
TOP_K = _to_int("TOP_K", _top_k_raw, min_value=1, max_value=50)


# Embedding model can be overridden for experiments but defaults to the spec.
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


# Ensure runtime directories exist so first-run scripts do not fail unexpectedly.
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

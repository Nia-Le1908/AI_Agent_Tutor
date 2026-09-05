"""
Centralized runtime configuration for AI Tutor.

This module is the ONLY place that reads environment variables / ``.env``.
Modules that previously parsed ``os.getenv`` themselves (config_db.py,
generator.py, controller.py) now import constants from here so a value can never
have two different defaults depending on the import order.

Design notes:
- Plain module-level constants are kept because most callers use
  ``from config import DB_PATH`` style imports, and early-phase scripts (init_db,
  embedder, seed_db) rely on them existing at import time.
- Directory creation is intentionally *not* a silent import side effect; callers
  invoke :func:`ensure_runtime_dirs` from their entry points.

Raises:
    ConfigError: at import time if any provided value is invalid or unsafe.
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


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean feature flag from the environment."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    """
    Resolve a configured path.

    Absolute values are used as-is; relative values are anchored to ``base_dir``
    (the project root by default). This replaces the "is it relative? then join
    vector_store/" logic that embedder, retriever and rag_tester each duplicated.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return ((base_dir or PROJECT_ROOT) / candidate)


# -------------------------
# Provider settings
# -------------------------
# The app talks to LLM providers through the OpenAI-compatible SDK. Keys stay
# optional at import time so non-LLM tooling (init_db, embedder, rag_tester) can
# run before credentials are configured.
DEEPSEEK_API_KEY: str = _env("DEEPSEEK_API_KEY", default="")
DEEPSEEK_MODEL: str = _env("DEEPSEEK_MODEL", default="deepseek-chat")
DEEPSEEK_BASE_URL: str = _env("DEEPSEEK_BASE_URL", default="https://api.deepseek.com")

# Default model name for every LLM entry point, kept as a module constant so
# `from generator import DEFAULT_MODEL` style imports stay stable.
DEFAULT_MODEL: str = DEEPSEEK_MODEL

LLM_TEMPERATURE: float = float(_env("LLM_TEMPERATURE", "0.7") or 0.7)
LLM_TIMEOUT_SECONDS: int = _to_int("LLM_TIMEOUT_SECONDS", _env("LLM_TIMEOUT_SECONDS", "60"), min_value=1, max_value=600)
LLM_MAX_ATTEMPTS: int = _to_int("LLM_MAX_ATTEMPTS", _env("LLM_MAX_ATTEMPTS", "5"), min_value=1, max_value=10)
LLM_INITIAL_BACKOFF_SECONDS: float = float(_env("LLM_INITIAL_BACKOFF_SECONDS", "1.0") or 1.0)
LLM_BACKOFF_FACTOR: float = float(_env("LLM_BACKOFF_FACTOR", "2.0") or 2.0)
LLM_MAX_BACKOFF_SECONDS: float = float(_env("LLM_MAX_BACKOFF_SECONDS", "20.0") or 20.0)

# Upper bound for one batch generation request; keeps a single call inside the
# prompt budget and stops the admin UI from asking for more than we can validate.
MAX_BATCH_SIZE = _to_int("MAX_BATCH_SIZE", _env("MAX_BATCH_SIZE", "10"), min_value=1, max_value=20)


def require_deepseek_api_key() -> str:
    """Return DEEPSEEK_API_KEY or raise a clear error when LLM features are used."""
    if not DEEPSEEK_API_KEY:
        raise ConfigError(
            "DEEPSEEK_API_KEY is required. Set it in .env: DEEPSEEK_API_KEY=sk-..."
        )
    return DEEPSEEK_API_KEY


# -------------------------
# Security / session
# -------------------------
JWT_SECRET_KEY: str = _env("JWT_SECRET_KEY", default="")
SESSION_TIMEOUT_HOURS: int = _to_int(
    "SESSION_TIMEOUT_HOURS", _env("SESSION_TIMEOUT_HOURS", "24"), min_value=1, max_value=720
)
# Older builds stored unsalted SHA-256 hashes. Verification of those hashes is kept
# available so existing accounts can still sign in, and can be switched off once the
# hashes have been migrated to bcrypt.
LEGACY_PASSWORD_FALLBACK_ENABLED = env_flag("ALLOW_LEGACY_PASSWORD_FALLBACK", default=True)


# -------------------------
# Database selection
# -------------------------
DB_TYPES = ("sqlite", "postgresql")

_db_type_raw = _env("DB_TYPE", "sqlite").lower()
if _db_type_raw not in DB_TYPES:
    raise ConfigError(f"DB_TYPE must be one of {DB_TYPES}, got: {_db_type_raw!r}")

DB_TYPE: str = _db_type_raw

POSTGRES_HOST: str = _env("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = _to_int("POSTGRES_PORT", _env("POSTGRES_PORT", "5432"), min_value=1, max_value=65535)
POSTGRES_DB: str = _env("POSTGRES_DB", "ai_tutor")
POSTGRES_USER: str = _env("POSTGRES_USER", "ai_tutor_user")
POSTGRES_PASSWORD: str = _env("POSTGRES_PASSWORD", "")
POSTGRES_POOL_SIZE: int = _to_int("POSTGRES_POOL_SIZE", _env("POSTGRES_POOL_SIZE", "10"), min_value=1, max_value=100)
POSTGRES_MAX_OVERFLOW: int = _to_int("POSTGRES_MAX_OVERFLOW", _env("POSTGRES_MAX_OVERFLOW", "20"), min_value=0, max_value=100)

if DB_TYPE == "postgresql" and not POSTGRES_PASSWORD:
    raise ConfigError(
        "POSTGRES_PASSWORD is required when DB_TYPE=postgresql. Set it in .env file."
    )


# -------------------------
# Paths
# -------------------------
# Path objects are used for safe path operations; string constants are exposed for
# compatibility with modules that expect plain strings.
PROJECT_ROOT = Path(__file__).resolve().parent

# `DB_PATH` is the canonical name; `SQLITE_DB_PATH` was the name used by the
# multi-database config module, so both are accepted with DB_PATH winning.
_sqlite_db_raw = _env("DB_PATH", "") or _env("SQLITE_DB_PATH", "data/ai_tutor_v5.db")
DB_PATH = str(resolve_path(_sqlite_db_raw).resolve())
FAISS_INDEX_PATH = str(resolve_path(_env("FAISS_INDEX_PATH", "vector_store/faiss_index.bin")).resolve())
LOG_PATH = str(resolve_path(_env("LOG_PATH", "logs/app.log")).resolve())
VECTOR_DIR = str(resolve_path(_env("VECTOR_DIR", "vector_store")).resolve())
DATA_DIR = str(resolve_path(_env("DATA_DIR", "data")).resolve())

SQLITE_TIMEOUT_SECONDS: int = _to_int(
    "SQLITE_TIMEOUT_SECONDS", _env("SQLITE_TIMEOUT_SECONDS", "10"), min_value=1, max_value=120
)


# -------------------------
# Retrieval and chunk tuning
# -------------------------
# Spec asks for chunk size 256-512 and overlap=50, top_k=3.
CHUNK_SIZE = _to_int("CHUNK_SIZE", _env("CHUNK_SIZE", "256"), min_value=256, max_value=512)
CHUNK_OVERLAP = _to_int("CHUNK_OVERLAP", _env("CHUNK_OVERLAP", "50"), min_value=0, max_value=512)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ConfigError("CHUNK_OVERLAP must be strictly smaller than CHUNK_SIZE")

TOP_K = _to_int("TOP_K", _env("TOP_K", "3"), min_value=1, max_value=50)

# Embedding model can be overridden for experiments but defaults to the spec.
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")



# -------------------------
# Observability
# -------------------------
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
DEBUG_RAG_CONTEXT = env_flag("DEBUG_RAG_CONTEXT", default=False)


def ensure_runtime_dirs() -> None:
    """
    Create the directories the app writes to (DB, vector store, logs).

    Called from entry points instead of running as an import side effect, so merely
    importing config in a test no longer touches the filesystem.
    """
    for path in (DB_PATH, FAISS_INDEX_PATH, LOG_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

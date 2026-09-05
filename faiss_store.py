"""
Shared FAISS / embedding-model plumbing for AI Tutor.

``embedder.py``, ``retriever.py`` and ``rag_tester.py`` each carried their own copy
of the same three concerns: resolving the index path from config, reading/writing a
FAISS index around the Windows Unicode-path bug, and loading the embedding model.
They all use this module now, so:

- the index and its metadata can never be resolved differently by two callers;
- the Windows fallback stays fixed in one place;
- the SentenceTransformer (and the opened index) are loaded once per process
  instead of once per chat message.

Heavy ML imports live in functions on purpose: importing this module must not drag
in torch/faiss for tools that only need config or database access.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import config

logger = logging.getLogger(__name__)

METADATA_FILENAME = "chunks_metadata.json"

# ASCII-only staging directory used to dodge FAISS' Unicode path failures.
_FAISS_TEMP_DIRNAME = "ai_tutor_faiss_tmp"

_model_lock = threading.Lock()


def default_vector_dir() -> Path:
    """The configured vector store directory, read lazily from config."""
    return Path(config.VECTOR_DIR)


def resolve_index_path(index_path: str | Path | None = None, *, vector_dir: str | Path | None = None) -> Path:
    """
    Resolve the FAISS index location.

    Absolute config values are honoured as-is; a bare filename is placed inside
    ``vector_dir`` (default: the configured vector store directory).

    Config values are read at call time rather than bound at import, so a runtime
    override (or a test fixture) cannot be shadowed by a stale module constant.
    """
    raw = Path(index_path) if index_path is not None else Path(config.FAISS_INDEX_PATH)
    if raw.is_absolute():
        return raw

    base = Path(vector_dir) if vector_dir is not None else default_vector_dir()
    return base / raw.name


def resolve_metadata_path(index_path: Path | None = None) -> Path:
    """
    Locate chunk metadata: next to the index when possible, else the default dir.

    Keeping this next to the index matters because ``build_faiss_index`` can be told
    to write into an arbitrary directory, and retrieval must find the match.
    """
    if index_path is not None:
        candidate = Path(index_path).parent / METADATA_FILENAME
        if candidate.exists():
            return candidate
    return default_vector_dir() / METADATA_FILENAME


def write_index(index: Any, index_path: Path) -> None:
    """
    Persist a FAISS index, with a Windows-safe fallback.

    On some Windows setups the FAISS C++ file APIs fail for Unicode paths (e.g.
    directories with accented characters), so we retry through an ASCII temp dir
    and copy the result with Python.
    """
    import faiss

    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        faiss.write_index(index, str(target))
        return
    except RuntimeError as exc:
        logger.warning("Direct FAISS write failed, using temp fallback: %s", exc)

    temp_dir = _temp_dir()
    temp_index_path = temp_dir / target.name
    faiss.write_index(index, str(temp_index_path))
    shutil.copyfile(temp_index_path, target)


def read_index(index_path: Path) -> Any:
    """
    Load a FAISS index, with the same Unicode-path fallback as :func:`write_index`.

    Raises:
        FileNotFoundError: when the index does not exist yet.
    """
    import faiss

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index file not found: {path}. Build it first via embedder.py"
        )

    try:
        return faiss.read_index(str(path))
    except RuntimeError:
        temp_index_path = _temp_dir() / path.name
        shutil.copyfile(path, temp_index_path)
        return faiss.read_index(str(temp_index_path))


def load_cached_index(index_path: Path) -> Any:
    """
    Read an index through a small cache keyed on (path, modified time).

    Retrieval is called on every chat message, and re-reading a multi-megabyte
    index each time is pure waste. Keying on mtime keeps results correct after a
    rebuild without any manual invalidation.
    """
    path = Path(index_path)
    try:
        fingerprint: Any = path.stat().st_mtime_ns
    except OSError:  # pragma: no cover - race with a rebuild
        fingerprint = None

    return _cached_index_read(str(path), fingerprint)


@lru_cache(maxsize=4)
def _cached_index_read(path_str: str, fingerprint: Any) -> Any:
    """Cache boundary: keyed on path + mtime, so a rebuild busts the entry."""
    return read_index(Path(path_str))


def _temp_dir() -> Path:
    temp_dir = Path(tempfile.gettempdir()) / _FAISS_TEMP_DIRNAME
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load and verify the chunk metadata JSON written by embedder.py.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if the structure is not usable.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Build index first via embedder.py"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Invalid metadata format: missing 'chunks' list")

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Invalid chunk at index {position}: expected object")
        for key in ("chunk_id", "source_file", "text"):
            if key not in chunk:
                raise ValueError(f"Chunk {position} missing required key: {key}")

    return data


def write_metadata(metadata_path: Path, payload: Dict[str, Any]) -> None:
    """Write chunk metadata as UTF-8 JSON, keeping Vietnamese text readable."""
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_metadata_payload(
    chunks: List[Dict[str, Any]],
    *,
    embedding_model: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Dict[str, Any]:
    """Assemble the metadata document shared by the index and its readers."""
    return {
        "embedding_model": embedding_model,
        "chunk_size": config.CHUNK_SIZE if chunk_size is None else chunk_size,
        "chunk_overlap": config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str | None = None) -> Any:
    """
    Return a cached SentenceTransformer for ``model_name``.

    Loading the encoder costs several seconds; caching keeps a chat round-trip from
    paying it on every message.

    Raises:
        ImportError: if sentence-transformers is not installed.
    """
    name = model_name or config.EMBEDDING_MODEL_NAME
    with _model_lock:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading embedding model: %s", name)
        return SentenceTransformer(name)


def load_index_for(metadata_path: Path | None = None, index_path: Path | None = None) -> Any:
    """
    Read the persisted index and its embedding model name together.

    Returns:
        (index, embedding_model_name)
    """
    resolved_index = Path(index_path) if index_path is not None else resolve_index_path()
    resolved_metadata = Path(metadata_path) if metadata_path is not None else resolve_metadata_path(resolved_index)

    metadata = load_metadata(resolved_metadata)
    index = read_index(resolved_index)
    model_name = str(metadata.get("embedding_model") or config.EMBEDDING_MODEL_NAME)
    return index, model_name


def embed_query(model: Any, query: str) -> Any:
    """
    Encode one query into a normalized float32 row for inner-product search.

    Normalizing is what makes the IndexFlatIP scores comparable to cosine
    similarity, so it must stay paired with how chunks were indexed.
    """
    import numpy as np

    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vector, dtype=np.float32)


def search_positions(index: Any, query_vector: Any, top_k: int) -> List[int]:
    """
    Return matched metadata positions for a query vector.

    ``k`` is capped at the number of indexed vectors, and FAISS' ``-1`` (no
    neighbour) markers are dropped.
    """
    return [position for position, _ in search_with_scores(index, query_vector, top_k)]


def search_with_scores(
    index: Any,
    query_vector: Any,
    top_k: int,
) -> List[tuple[int, float]]:
    """
    Return ``(metadata_position, similarity_score)`` pairs, best first.

    Positions and scores stay paired after invalid (-1) matches are dropped, so
    callers can rank or report scores without re-deriving indices.
    """
    if top_k <= 0:
        return []

    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    scores, indices = index.search(query_vector, k)

    results: List[tuple[int, float]] = []
    for position, score in zip(indices[0], scores[0]):
        position = int(position)
        if position >= 0:
            results.append((position, float(score)))

    return results


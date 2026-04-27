"""
Phase 2 - Retrieval over FAISS index for AI Tutor V5.1.

This module provides:
- retrieve(query: str, top_k: int) -> List[str]

It loads the persisted FAISS index and chunk metadata created by embedder.py,
embeds the user query with the same embedding model, and returns the top-k
most similar chunks.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional config import. Fall back to project defaults if constants are not yet
# available in config.py.
try:
    from config import FAISS_INDEX_PATH, TOP_K
except Exception:  # pragma: no cover
    FAISS_INDEX_PATH = "vector_store/faiss_index.bin"
    TOP_K = 3


DEFAULT_VECTOR_DIR = Path("vector_store")
DEFAULT_METADATA_PATH = DEFAULT_VECTOR_DIR / "chunks_metadata.json"


def _load_index(index_path: Path) -> faiss.Index:
    """
    Load FAISS index from disk with a Unicode-path fallback for Windows.

    Some FAISS wheel builds on Windows fail to open paths containing Unicode
    characters. We try direct loading first, then copy the index to an ASCII
    temp path and load from there.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index file not found: {index_path}. Build it first via embedder.py"
        )

    try:
        return faiss.read_index(str(index_path))
    except RuntimeError:
        temp_dir = Path(tempfile.gettempdir()) / "ai_tutor_faiss_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_index_path = temp_dir / "faiss_index.bin"
        shutil.copyfile(index_path, temp_index_path)
        return faiss.read_index(str(temp_index_path))


def _load_metadata(metadata_path: Path) -> dict:
    """Load chunk metadata JSON written by embedder.py."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. Build index first via embedder.py"
        )

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "chunks" not in data or not isinstance(data["chunks"], list):
        raise ValueError("Invalid metadata format: missing 'chunks' list")
    return data


def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    """
    Retrieve top-k relevant chunk texts for a query.

    Args:
        query: User query string.
        top_k: Number of nearest chunks to return.

    Returns:
        List[str]: The content of top-k retrieved chunks.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.is_absolute():
        index_path = DEFAULT_VECTOR_DIR / index_path.name

    metadata = _load_metadata(DEFAULT_METADATA_PATH)
    index = _load_index(index_path)

    embedding_model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
    model = SentenceTransformer(embedding_model_name)

    query_vector = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    # Request at most the number of indexed vectors to avoid invalid accesses.
    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    _, indices = index.search(query_vector, k)

    chunks = metadata["chunks"]
    results: List[str] = []

    # indices has shape (1, k) for a single query.
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue

        chunk_text = chunks[idx].get("text", "").strip()
        if chunk_text:
            results.append(chunk_text)

    return results


def retrieve_with_sources(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    """
    Retrieve top-k relevant chunks with source file metadata for citations.

    Returns:
        List[dict] with keys: 'text', 'source_file'
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.is_absolute():
        index_path = DEFAULT_VECTOR_DIR / index_path.name

    metadata = _load_metadata(DEFAULT_METADATA_PATH)
    index = _load_index(index_path)

    embedding_model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
    model = SentenceTransformer(embedding_model_name)

    query_vector = model.encode(
        [query.strip()], convert_to_numpy=True, normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    _, indices = index.search(query_vector, k)
    chunks = metadata["chunks"]
    results: List[Dict[str, str]] = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk_text = chunks[idx].get("text", "").strip()
        source = chunks[idx].get("source_file", "unknown")
        # Extract just the filename from the full path
        source_name = Path(source).name if source else "unknown"
        if chunk_text:
            results.append({"text": chunk_text, "source_file": source_name})

    return results


if __name__ == "__main__":
    demo_query = "What is Newton's second law?"
    hits = retrieve(demo_query, top_k=3)
    print("Retrieved chunks:")
    for i, chunk in enumerate(hits, start=1):
        print(f"[{i}] {chunk[:200]}...")

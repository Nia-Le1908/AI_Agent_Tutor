"""
Retrieval over the FAISS index for AI Tutor.

Required interfaces:
- retrieve(query: str, top_k: int) -> List[str]
- retrieve_with_sources(query: str, top_k: int) -> List[Dict[str, str]]

Both calls are thin views over one search routine, so the prompt path and the
citation path can never rank the same query differently. Path resolution,
index/model caching and the FAISS query itself live in :mod:`faiss_store`; the
model and index are now reused across calls instead of being reloaded for every
chat message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss_store
from config import EMBEDDING_MODEL_NAME, TOP_K
from validation import require_non_empty_str, require_positive_int

logger = logging.getLogger(__name__)

UNKNOWN_SOURCE = "unknown"

__all__ = ["ChunkHit", "search", "retrieve", "retrieve_with_sources"]


@dataclass(frozen=True)
class ChunkHit:
    """One retrieved chunk with its provenance and similarity score."""

    text: str
    source_file: str
    chunk_id: int
    score: float


def _source_name(source: object) -> str:
    """Reduce a stored path to a display filename for citations."""
    name = Path(str(source)).name if source else ""
    return name or UNKNOWN_SOURCE


def search(query: str, top_k: int = TOP_K) -> List[ChunkHit]:
    """
    Rank indexed chunks for one query.

    Args:
        query: Non-empty search text.
        top_k: Number of neighbours to inspect (clamped to the index size).

    Returns:
        List[ChunkHit] in descending similarity order. Empty when the index holds
        no vectors.

    Raises:
        ValueError: for invalid arguments.
        FileNotFoundError: when the index or metadata has not been built yet.
    """
    cleaned = require_non_empty_str(query, "query")
    require_positive_int(top_k, "top_k")

    index_path = faiss_store.resolve_index_path()
    metadata_path = faiss_store.resolve_metadata_path(index_path)
    metadata = faiss_store.load_metadata(metadata_path)
    chunks: List[Dict[str, Any]] = metadata["chunks"]

    # Older metadata files may not record the model, so fall back to the config default.
    model_name = str(metadata.get("embedding_model") or EMBEDDING_MODEL_NAME)
    index = faiss_store.load_cached_index(index_path)
    matches = faiss_store.search_with_scores(
        index,
        faiss_store.embed_query(faiss_store.get_embedding_model(model_name), cleaned),
        top_k,
    )

    hits: List[ChunkHit] = []
    for position, score in matches:
        if position >= len(chunks):
            # An index rebuilt without regenerating metadata can point past the
            # chunk list; skip the stale match instead of failing the chat.
            logger.warning("Skipping out-of-range chunk position %d", position)
            continue

        chunk = chunks[position]
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        hits.append(
            ChunkHit(
                text=text,
                source_file=_source_name(chunk.get("source_file")),
                chunk_id=int(chunk.get("chunk_id", position)),
                score=score,
            )
        )

    return hits


def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    """
    Retrieve top-k relevant chunk texts for a query.

    Args:
        query: User query string.
        top_k: Number of nearest chunks to return.

    Returns:
        List[str]: the content of the top-k retrieved chunks.
    """
    return [hit.text for hit in search(query, top_k)]


def retrieve_with_sources(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    """
    Retrieve top-k relevant chunks with source metadata for citations.

    Returns:
        List[dict] with the keys 'text' and 'source_file'.
    """
    return [{"text": hit.text, "source_file": hit.source_file} for hit in search(query, top_k)]


if __name__ == "__main__":
    demo_query = "What is Newton's second law?"
    print("Retrieved chunks:")
    for position, chunk in enumerate(retrieve(demo_query, top_k=3), start=1):
        print(f"[{position}] {chunk[:200]}...")

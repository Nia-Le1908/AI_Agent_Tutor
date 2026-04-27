"""
RAG retrieval evaluator for AI Tutor V5.1.

What this script does:
1) Loads the existing FAISS index and chunk metadata built by embedder.py.
2) Automatically constructs exactly 20 deterministic test questions.
3) Runs retrieval for each question with top_k=3.
4) Computes and prints:
   - Precision@3
   - Mean Reciprocal Rank (MRR)

Why this approach is robust:
- The tests are generated from your real indexed chunks, so they always match
  the current corpus and avoid stale/manual test drift.
- Relevance is defined as chunks from the same source document as the target
  chunk, which makes top-3 relevance checks meaningful in multi-chunk docs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse project configuration when available.
try:
    from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH
except Exception:  # pragma: no cover
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "vector_store/faiss_index.bin"


DEFAULT_TOP_K = 3
DEFAULT_TEST_COUNT = 20
DEFAULT_VECTOR_DIR = Path("vector_store")
DEFAULT_METADATA_PATH = DEFAULT_VECTOR_DIR / "chunks_metadata.json"


@dataclass(frozen=True)
class RagTestCase:
    """One retrieval test case."""

    query: str
    target_chunk_id: int
    relevant_chunk_ids: Set[int]
    source_file: str


def _resolve_index_path() -> Path:
    """Resolve FAISS index path from config/defaults."""
    idx = Path(FAISS_INDEX_PATH)
    if idx.is_absolute():
        return idx
    return DEFAULT_VECTOR_DIR / idx.name


def _resolve_metadata_path(index_path: Path) -> Path:
    """Resolve metadata path alongside index whenever possible."""
    candidate = index_path.parent / "chunks_metadata.json"
    if candidate.exists():
        return candidate
    return DEFAULT_METADATA_PATH


def _load_index(index_path: Path) -> faiss.Index:
    """Load FAISS index from disk with clear failures."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Build it first with embedder.py"
        )
    return faiss.read_index(str(index_path))


def _load_metadata(metadata_path: Path) -> Dict:
    """Load chunk metadata JSON and verify required structure."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. Build index first with embedder.py"
        )

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Invalid metadata: missing 'chunks' list")

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Invalid chunk at index {i}: expected object")
        if "chunk_id" not in chunk or "source_file" not in chunk or "text" not in chunk:
            raise ValueError(f"Chunk {i} missing required keys: chunk_id/source_file/text")

    return data


def _normalize_words(text: str) -> List[str]:
    """
    Convert text into normalized words for query synthesis.

    We keep only alphabetic words with length >= 4 to reduce noise.
    """
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    return words


def _build_query_from_chunk_text(text: str) -> str:
    """
    Generate a deterministic natural-language query from chunk text.

    Strategy:
    - Take the first 12 meaningful words from the chunk.
    - Build a question that asks for the concept explained by those terms.
    """
    words = _normalize_words(text)

    # Fallback for extremely short/noisy chunks.
    if not words:
        return "What concept is explained in this study material?"

    key_terms = words[:12]
    joined_terms = " ".join(key_terms)
    return f"What does the document explain about: {joined_terms}?"


def _select_test_chunk_indices(total_chunks: int, test_count: int) -> List[int]:
    """
    Deterministically spread selected chunk indices across the whole corpus.

    This avoids clustering all tests in one region of the index.
    """
    if total_chunks < test_count:
        raise ValueError(
            f"Need at least {test_count} indexed chunks, found {total_chunks}. "
            "Add more documents or reduce test_count."
        )

    # Evenly spaced deterministic sampling.
    step = total_chunks / test_count
    indices: List[int] = []
    for i in range(test_count):
        candidate = int(i * step)
        if candidate >= total_chunks:
            candidate = total_chunks - 1
        indices.append(candidate)

    # Ensure uniqueness (rare edge cases when total_chunks ~ test_count).
    unique = sorted(set(indices))
    if len(unique) == test_count:
        return unique

    # Fill missing slots linearly.
    used = set(unique)
    cur = 0
    while len(unique) < test_count and cur < total_chunks:
        if cur not in used:
            unique.append(cur)
            used.add(cur)
        cur += 1

    unique.sort()
    return unique[:test_count]


def build_test_cases(metadata: Dict, test_count: int = DEFAULT_TEST_COUNT) -> List[RagTestCase]:
    """
    Build exactly test_count test cases from metadata.

    Relevance definition:
    - A retrieved chunk is relevant if it comes from the same source_file as the
      target chunk selected for that test case.
    """
    chunks = metadata["chunks"]
    total_chunks = len(chunks)

    selected_indices = _select_test_chunk_indices(total_chunks=total_chunks, test_count=test_count)

    # Build reverse index: source_file -> all chunk_ids from that source.
    source_to_chunk_ids: Dict[str, Set[int]] = {}
    for chunk in chunks:
        source = str(chunk["source_file"])
        cid = int(chunk["chunk_id"])
        source_to_chunk_ids.setdefault(source, set()).add(cid)

    cases: List[RagTestCase] = []
    for idx in selected_indices:
        chunk = chunks[idx]
        target_chunk_id = int(chunk["chunk_id"])
        source_file = str(chunk["source_file"])
        text = str(chunk["text"])

        query = _build_query_from_chunk_text(text)
        relevant_ids = source_to_chunk_ids.get(source_file, {target_chunk_id})

        cases.append(
            RagTestCase(
                query=query,
                target_chunk_id=target_chunk_id,
                relevant_chunk_ids=set(relevant_ids),
                source_file=source_file,
            )
        )

    return cases


def _embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode query into a normalized float32 vector for FAISS inner-product search."""
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def _search_top_k(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> List[int]:
    """Search FAISS and return retrieved chunk indices (metadata positions)."""
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    _, indices = index.search(query_vec, k)
    return [int(i) for i in indices[0] if int(i) >= 0]


def evaluate_retrieval(
    test_cases: Sequence[RagTestCase],
    metadata: Dict,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[float, float, List[Dict]]:
    """
    Evaluate retrieval using Precision@K and MRR.

    Returns:
    - mean_precision_at_k
    - mean_reciprocal_rank
    - per_case_details (for debugging/reporting)
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    chunks = metadata["chunks"]
    model_name = str(metadata.get("embedding_model", EMBEDDING_MODEL_NAME))
    model = SentenceTransformer(model_name)

    index_path = _resolve_index_path()
    index = _load_index(index_path)

    precision_values: List[float] = []
    reciprocal_ranks: List[float] = []
    details: List[Dict] = []

    for i, case in enumerate(test_cases, start=1):
        query_vec = _embed_query(model, case.query)
        retrieved_positions = _search_top_k(index, query_vec, top_k=top_k)

        # Convert metadata positions -> chunk_ids for relevance checking.
        retrieved_chunk_ids: List[int] = []
        for pos in retrieved_positions:
            if pos < 0 or pos >= len(chunks):
                continue
            retrieved_chunk_ids.append(int(chunks[pos]["chunk_id"]))

        relevant_hits = [cid for cid in retrieved_chunk_ids if cid in case.relevant_chunk_ids]

        precision_at_k = len(relevant_hits) / float(top_k)
        precision_values.append(precision_at_k)

        rr = 0.0
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in case.relevant_chunk_ids:
                rr = 1.0 / float(rank)
                break
        reciprocal_ranks.append(rr)

        details.append(
            {
                "test_id": i,
                "query": case.query,
                "target_chunk_id": case.target_chunk_id,
                "source_file": case.source_file,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "relevant_hits": relevant_hits,
                "precision_at_k": precision_at_k,
                "reciprocal_rank": rr,
            }
        )

    mean_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
    mean_rr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return mean_precision, mean_rr, details


def _print_report(mean_p3: float, mrr: float, details: Sequence[Dict], top_k: int) -> None:
    """Print concise metric summary and per-test inspection lines."""
    print("=" * 72)
    print("RAG Retrieval Evaluation Report")
    print("=" * 72)
    print(f"Total test questions: {len(details)}")
    print(f"Top-K evaluated: {top_k}")
    print(f"Precision@{top_k}: {mean_p3:.4f}")
    print(f"MRR: {mrr:.4f}")
    print("-" * 72)
    print("Per-test details (target -> retrieved):")

    for item in details:
        print(
            f"[Q{item['test_id']:02d}] target={item['target_chunk_id']} "
            f"retrieved={item['retrieved_chunk_ids']} "
            f"P@{top_k}={item['precision_at_k']:.3f} RR={item['reciprocal_rank']:.3f}"
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality with 20 deterministic test questions."
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=DEFAULT_TEST_COUNT,
        help="Number of test questions to run (default: 20).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K retrieval depth for evaluation (default: 3).",
    )
    args = parser.parse_args()

    if args.test_count != 20:
        raise ValueError("This evaluator must run 20 test questions as requested. Use --test-count 20.")

    index_path = _resolve_index_path()
    metadata_path = _resolve_metadata_path(index_path)
    metadata = _load_metadata(metadata_path)

    test_cases = build_test_cases(metadata, test_count=args.test_count)
    mean_p_at_k, mrr, details = evaluate_retrieval(test_cases, metadata, top_k=args.top_k)

    _print_report(mean_p_at_k, mrr, details, args.top_k)


if __name__ == "__main__":
    main()

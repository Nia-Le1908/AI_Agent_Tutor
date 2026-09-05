"""
RAG retrieval evaluator for AI Tutor.

What this script does:
1. Loads the FAISS index and chunk metadata built by embedder.py.
2. Builds exactly 20 deterministic test questions from the indexed chunks.
3. Runs retrieval for each question and scores it.
4. Prints Precision@K and Mean Reciprocal Rank (MRR) plus per-question detail.

Why this design:
- Tests are derived from the real indexed chunks, so they always match the current
  corpus and cannot drift the way a hand-written question list does.
- Relevance means "chunk from the same source document as the target chunk", which
  makes a top-k relevance check meaningful for multi-chunk documents.

Metrics are pure functions (:func:`precision_at_k`, :func:`reciprocal_rank`) so
they can be unit-tested without an index.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import faiss_store
from config import EMBEDDING_MODEL_NAME
from validation import require_positive_int

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_TEST_COUNT = 20


@dataclass(frozen=True)
class RagTestCase:
    """One retrieval test case."""

    query: str
    target_chunk_id: int
    relevant_chunk_ids: Set[int] = field(compare=False, hash=False)
    source_file: str = ""


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------
def precision_at_k(retrieved: Sequence[int], relevant: Set[int], top_k: int) -> float:
    """Fraction of the top-k results that are relevant (0.0 when top_k is 0)."""
    require_positive_int(top_k, "top_k")
    if not retrieved:
        return 0.0
    hits = sum(1 for chunk_id in retrieved[:top_k] if chunk_id in relevant)
    return hits / float(top_k)


def reciprocal_rank(retrieved: Sequence[int], relevant: Set[int]) -> float:
    """1 / rank of the first relevant result, or 0.0 when nothing relevant is found."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / float(rank)
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Query synthesis
# ---------------------------------------------------------------------------
def _normalize_words(text: str) -> List[str]:
    """
    Extract normalized words for query synthesis.

    Only alphabetic tokens of length >= 4 are kept, which filters connectors and
    other noise out of the synthesized query.
    """
    return re.findall(r"[A-Za-z]{4,}", text.lower())



def build_query_from_chunk_text(text: str) -> str:
    """
    Build a deterministic natural-language query from chunk text.

    Uses the chunk's first meaningful terms, so the same corpus always produces the
    same test set.
    """
    words = _normalize_words(text)
    if not words:
        # Fallback for extremely short or noisy chunks.
        return "What concept is explained in this study material?"

    return f"What does the document explain about: {' '.join(words[:12])}?"


def select_test_chunk_indices(total_chunks: int, test_count: int) -> List[int]:
    """
    Spread selected chunk indices evenly and deterministically across the corpus.

    Prevents every test from clustering in one region of the index, which would
    overstate precision on repetitive documents.
    """
    require_positive_int(test_count, "test_count")
    if total_chunks < test_count:
        raise ValueError(
            f"Need at least {test_count} indexed chunks, found {total_chunks}. "
            "Add more documents or reduce test_count."
        )

    step = total_chunks / test_count
    indices = {min(int(position * step), total_chunks - 1) for position in range(test_count)}

    if len(indices) < test_count:
        # Degenerate case (test_count close to total_chunks): fill linearly.
        for candidate in range(total_chunks):
            if len(indices) >= test_count:
                break
            indices.add(candidate)

    return sorted(indices)[:test_count]


def build_test_cases(metadata: Dict, test_count: int = DEFAULT_TEST_COUNT) -> List[RagTestCase]:
    """
    Build ``test_count`` test cases from index metadata.

    Relevance rule: a retrieved chunk counts as relevant when it comes from the
    same source file as the target chunk for that test case.
    """
    chunks = metadata["chunks"]
    selected = select_test_chunk_indices(len(chunks), test_count)

    source_to_chunk_ids: Dict[str, Set[int]] = {}
    for chunk in chunks:
        source_to_chunk_ids.setdefault(str(chunk["source_file"]), set()).add(int(chunk["chunk_id"]))

    cases: List[RagTestCase] = []
    for index in selected:
        chunk = chunks[index]
        source_file = str(chunk["source_file"])
        target_chunk_id = int(chunk["chunk_id"])
        relevant_ids = source_to_chunk_ids.get(source_file, {target_chunk_id})

        cases.append(
            RagTestCase(
                query=build_query_from_chunk_text(str(chunk["text"])),
                target_chunk_id=target_chunk_id,
                relevant_chunk_ids=set(relevant_ids),
                source_file=source_file,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    test_cases: Sequence[RagTestCase],
    metadata: Dict,
    *,
    top_k: int = DEFAULT_TOP_K,
    index_path: Path | None = None,
) -> Tuple[float, float, List[Dict]]:
    """
    Evaluate retrieval quality with Precision@K and MRR.

    Returns:
        (mean_precision_at_k, mean_reciprocal_rank, per_case_details)
    """
    require_positive_int(top_k, "top_k")

    chunks = metadata["chunks"]
    model = faiss_store.get_embedding_model(str(metadata.get("embedding_model") or EMBEDDING_MODEL_NAME))
    index = faiss_store.read_index(index_path or faiss_store.resolve_index_path())

    precision_values: List[float] = []
    reciprocal_ranks: List[float] = []
    details: List[Dict] = []

    for position, case in enumerate(test_cases, start=1):
        matches = faiss_store.search_with_scores(
            index, faiss_store.embed_query(model, case.query), top_k
        )

        retrieved_chunk_ids = [int(chunks[match]["chunk_id"]) for match, _ in matches if match < len(chunks)]

        precision = precision_at_k(retrieved_chunk_ids, case.relevant_chunk_ids, top_k)
        rank = reciprocal_rank(retrieved_chunk_ids, case.relevant_chunk_ids)
        relevant_hits = [cid for cid in retrieved_chunk_ids if cid in case.relevant_chunk_ids]

        precision_values.append(precision)
        reciprocal_ranks.append(rank)
        details.append(
            {
                "test_id": position,
                "query": case.query,
                "target_chunk_id": case.target_chunk_id,
                "source_file": case.source_file,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "relevant_hits": relevant_hits,
                "precision_at_k": precision,
                "reciprocal_rank": rank,
            }
        )

    return mean(precision_values), mean(reciprocal_ranks), details


def print_report(mean_p_at_k: float, mrr: float, details: Sequence[Dict], top_k: int) -> None:
    """Print a concise metric summary plus one line per test case."""
    print("=" * 72)
    print("RAG Retrieval Evaluation Report")
    print("=" * 72)
    print(f"Total test questions: {len(details)}")
    print(f"Top-K evaluated: {top_k}")
    print(f"Precision@{top_k}: {mean_p_at_k:.4f}")
    print(f"MRR: {mrr:.4f}")
    print("-" * 72)
    print("Per-test details (target -> retrieved):")

    for item in details:
        print(
            f"[Q{item['test_id']:02d}] target={item['target_chunk_id']} "
            f"retrieved={item['retrieved_chunk_ids']} "
            f"P@{top_k}={item['precision_at_k']:.3f} RR={item['reciprocal_rank']:.3f}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser, separated from main() so it can be exercised in tests."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality with 20 deterministic test questions."
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=DEFAULT_TEST_COUNT,
        help=f"Number of test questions to run (fixed at {DEFAULT_TEST_COUNT}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-K retrieval depth for evaluation (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    if args.test_count != DEFAULT_TEST_COUNT:
        raise ValueError(
            f"This evaluator must run exactly {DEFAULT_TEST_COUNT} test questions. "
            f"Use --test-count {DEFAULT_TEST_COUNT}."
        )

    index_path = faiss_store.resolve_index_path()
    metadata = faiss_store.load_metadata(faiss_store.resolve_metadata_path(index_path))

    test_cases = build_test_cases(metadata, test_count=args.test_count)
    mean_p_at_k, mrr, details = evaluate_retrieval(test_cases, metadata, top_k=args.top_k)

    print_report(mean_p_at_k, mrr, details, args.top_k)


if __name__ == "__main__":
    main()

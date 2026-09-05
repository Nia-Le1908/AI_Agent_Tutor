"""Reusable data builders and test doubles for the AI Tutor test suite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def question_payload(**overrides: Any) -> Dict[str, Any]:
    """Build a schema-valid question payload with optional field overrides."""
    payload: Dict[str, Any] = {
        "question_id": 101,
        "content": "What is 2 + 2?",
        "difficulty": 1,
        "subject": "Mathematics",
        "options": ["3", "4", "5", "6"],
        "answer": "B",
        "explanation": "Basic arithmetic.",
    }
    payload.update(overrides)
    return payload


def insert_history(db_path: Path, rows: Iterable[tuple]) -> None:
    """
    Append (uid, qid, is_correct) history rows with distinct ordered timestamps.

    Timestamps increase with insertion order so "most recent first" assertions are
    meaningful.
    """
    connection = sqlite3.connect(db_path)
    try:
        for index, (uid, qid, is_correct) in enumerate(rows, start=1):
            connection.execute(
                "INSERT INTO history (uid, qid, is_correct, timestamp) VALUES (?, ?, ?, ?)",
                (uid, qid, int(is_correct), f"2026-01-01 10:{index:02d}:00"),
            )
        connection.commit()
    finally:
        connection.close()


def db_rows(db_path: Path, sql: str, params: tuple = ()) -> List[tuple]:
    """Read raw rows straight from the database, bypassing the app layer."""
    connection = sqlite3.connect(db_path)
    try:
        return list(connection.execute(sql, params))
    finally:
        connection.close()


def db_execute(db_path: Path, sql: str, params: tuple = ()) -> None:
    """Write directly to the database for test setup that has no app API."""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(sql, params)
        connection.commit()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Embedding doubles
# ---------------------------------------------------------------------------
class FakeTokenizer:
    """
    Word-level tokenizer with a stable, reversible vocabulary.

    ``decode(encode(text))`` round-trips the words, so chunking tests can assert on
    real content instead of opaque ids.
    """

    def __init__(self) -> None:
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [self._id_for(word) for word in text.split()]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._id_to_word.get(token, f"unk{token}") for token in token_ids)

    def _id_for(self, word: str) -> int:
        key = word.lower()
        if key not in self._word_to_id:
            self._word_to_id[key] = self._next_id
            self._id_to_word[self._next_id] = key
            self._next_id += 1
        return self._word_to_id[key]


class FakeEmbeddingModel:
    """
    Deterministic vectorizer: one dimension per hashed word.

    Cosine similarity between two texts therefore tracks shared vocabulary, which
    is the property retrieval tests rely on. Counting ``encode_calls`` lets tests
    assert the model is reused rather than reloaded per query.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.tokenizer = FakeTokenizer()
        self.encode_calls = 0

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        import numpy as np

        self.encode_calls += 1
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            for word in str(text).lower().split():
                vectors[row, self._bucket(word)] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        return vectors if convert_to_numpy else vectors.tolist()

    def _bucket(self, word: str) -> int:
        return sum(ord(char) for char in word) % self.dimension


class FakeLLM:
    """
    Scripted stand-in for :func:`llm_client.chat`.

    Push replies with :meth:`queue`; every prompt the app sent is recorded, so tests
    can assert on grounding content without touching the network.
    """

    def __init__(self) -> None:
        self.responses: List[str] = []
        self.prompts: List[Dict[str, Any]] = []

    def queue(self, *payloads: str) -> "FakeLLM":
        self.responses.extend(payloads)
        return self

    @property
    def calls(self) -> int:
        return len(self.prompts)

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]["prompt"]

    def __call__(self, prompt: str = "", **kwargs: Any) -> str:
        self.prompts.append({"prompt": prompt, **kwargs})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)

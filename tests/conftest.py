"""
Shared pytest fixtures for the AI Tutor test suite.

Two rules keep this suite useful:
- it never touches the network, a real LLM, or downloaded sentence-transformer
  weights (fake embeddings + real FAISS keep retrieval tests meaningful);
- every database test runs against a throwaway SQLite file created from the real
  ``schema.sql``, so schema drift shows up as a failing test.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(Path(__file__).parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import config  # noqa: E402
from helpers import FakeEmbeddingModel, FakeLLM  # noqa: E402
from init_db import read_schema_sql  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration isolation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root, used by the architecture-guard tests."""
    return REPO_ROOT


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch) -> Path:
    """
    Point the whole app at a scratch database and vector store for each test.

    ``db_manager`` reads ``config.DB_PATH`` when it opens a connection, so patching
    the module attribute redirects every layer without touching the real data file.
    """
    db_file = tmp_path / "aitutor_test.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "vector_store" / "faiss_index.bin"))
    monkeypatch.setattr(config, "VECTOR_DIR", str(tmp_path / "vector_store"))
    monkeypatch.setattr(config, "DEBUG_RAG_CONTEXT", False)
    return db_file


@pytest.fixture
def db(isolated_config) -> Path:
    """An empty, schema-initialized SQLite database."""
    connection = sqlite3.connect(isolated_config)
    try:
        connection.executescript(read_schema_sql())
        connection.commit()
    finally:
        connection.close()
    return isolated_config


@pytest.fixture
def seeded_db(db) -> Path:
    """
    Database with two users, six questions (two subjects, levels 1-4) and history.

    Question 6 holds malformed options JSON, but sits at difficulty 5 so it only
    surfaces in the test that exercises the "unusable question" path. Alice's four
    answers span two difficulty levels and end on two wrong answers in a row.
    """
    connection = sqlite3.connect(db)
    try:
        connection.executescript(
            """
            INSERT INTO users (id, name, level) VALUES (1, 'Alice', 1), (2, 'Bob', 3);

            INSERT INTO questions (id, content, difficulty, subject, options, answer, explanation) VALUES
                (1, 'Math level 1 first',  1, 'Math',
                 '{"A":"1","B":"2","C":"3","D":"4"}', 'B', 'two'),
                (2, 'Math level 1 second', 1, 'Math',
                 '{"A":"4","B":"5","C":"6","D":"7"}', 'C', 'six'),
                (3, 'Math level 2',        2, 'Math',
                 '{"A":"8","B":"9","C":"10","D":"11"}', 'C', 'ten'),
                (4, 'Science level 1',     1, 'Science',
                 '{"A":"O2","B":"N2","C":"H2","D":"CO2"}', 'B', 'nitrogen'),
                (5, 'Science level 4',     4, 'Science',
                 '{"A":"a","B":"b","C":"c","D":"d"}', 'A', 'first'),
                (6, 'Broken options',      5, 'Math', 'not-json', 'A', 'oops');

            INSERT INTO history (uid, qid, is_correct, timestamp) VALUES
                (1, 1, 1, '2026-01-01 09:00:00'),
                (1, 2, 1, '2026-01-01 09:01:00'),
                (1, 4, 0, '2026-01-01 09:02:00'),
                (1, 3, 0, '2026-01-01 09:03:00');
            """
        )
        connection.commit()
    finally:
        connection.close()
    return db


# ---------------------------------------------------------------------------
# LLM doubles
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Replace :func:`llm_client.chat` with a scripted fake."""
    import llm_client

    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "chat", fake)
    return fake


@pytest.fixture
def no_sleep(monkeypatch) -> List[float]:
    """Record sleeps instead of waiting, so retry tests stay instant."""
    import time

    slept: List[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))
    return slept


# ---------------------------------------------------------------------------
# Vector store doubles
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


CORPUS = [
    "Newton second law force equals mass times acceleration",
    "Photosynthesis converts light into glucose and oxygen",
    "Queue is first in first out while stack is last in first out",
    "The water cycle describes evaporation condensation and precipitation",
]


@pytest.fixture
def vector_store(tmp_path, monkeypatch, fake_model) -> Dict[str, Any]:
    """
    Build a real FAISS index and metadata over fake embeddings, wired into config.

    Returns a dict with the paths and the chunk list so tests can assert on
    provenance without re-reading files.
    """
    import json

    import faiss
    import numpy as np

    import faiss_store

    store_dir = tmp_path / "vector_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"chunk_id": position, "source_file": f"/docs/doc{position % 2}.pdf", "text": text}
        for position, text in enumerate(CORPUS)
    ]

    vectors = np.asarray(
        fake_model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True),
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = store_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    (store_dir / "chunks_metadata.json").write_text(
        json.dumps(
            {
                "embedding_model": "fake-model",
                "chunk_size": 256,
                "chunk_overlap": 0,
                "total_chunks": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Never bypass the cache in tests: tmp paths are reused within a session and a
    # stale handle would hide a rebuild.
    monkeypatch.setattr(faiss_store, "load_cached_index", faiss_store.read_index)
    monkeypatch.setattr(faiss_store, "get_embedding_model", lambda name=None: fake_model)

    return {
        "dir": store_dir,
        "index_path": index_path,
        "metadata_path": store_dir / "chunks_metadata.json",
        "chunks": chunks,
        "model": fake_model,
    }


@pytest.fixture
def point_config_at_vector_store(monkeypatch, vector_store) -> Dict[str, Any]:
    """Make config resolve the FAISS index to the fixture-built temp store."""
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(vector_store["index_path"]))
    monkeypatch.setattr(config, "VECTOR_DIR", str(vector_store["dir"]))
    monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "fake-model")
    return vector_store

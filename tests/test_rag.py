"""
Tests for the RAG pipeline: vector store plumbing, chunking, retrieval, evaluation.

The FAISS math is real; only the encoder is faked (a deterministic word-hash
model), so these tests verify ranking behaviour and file layout without
downloading weights or requiring a GPU.
"""

from __future__ import annotations

import faiss
import json
import numpy as np
import pytest
from pathlib import Path

import embedder
import faiss_store
import rag_tester
import retriever
from helpers import FakeEmbeddingModel


# ---------------------------------------------------------------------------
# faiss_store
# ---------------------------------------------------------------------------
class TestPathResolution:
    def test_absolute_config_path_is_used_as_is(self, tmp_path, monkeypatch):
        import config

        target = tmp_path / "custom" / "index.bin"
        monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(target))
        assert faiss_store.resolve_index_path() == target

    def test_relative_config_path_is_anchored_to_the_vector_dir(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "FAISS_INDEX_PATH", "faiss_index.bin")
        monkeypatch.setattr(config, "VECTOR_DIR", str(tmp_path / "vs"))
        assert faiss_store.resolve_index_path() == tmp_path / "vs" / "faiss_index.bin"

    def test_metadata_prefers_the_index_directory(self, vector_store):
        resolved = faiss_store.resolve_metadata_path(vector_store["index_path"])
        assert resolved == vector_store["metadata_path"]

    def test_metadata_falls_back_to_the_default_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.VECTOR_DIR", str(tmp_path / "elsewhere"))
        assert faiss_store.resolve_metadata_path(tmp_path / "nope" / "index.bin") == (
            tmp_path / "elsewhere" / "chunks_metadata.json"
        )

    def test_index_can_be_written_and_read_back(self, tmp_path):
        import faiss

        vectors = np.eye(4, dtype=np.float32)
        index = faiss.IndexFlatIP(4)
        index.add(vectors)

        target = tmp_path / "store" / "index.bin"
        faiss_store.write_index(index, target)
        reloaded = faiss_store.read_index(target)

        assert reloaded.ntotal == 4

    def test_reading_a_missing_index_explains_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Build it first via embedder.py"):
            faiss_store.read_index(tmp_path / "absent.bin")


class TestMetadata:
    def test_round_trip(self, tmp_path):
        payload = faiss_store.build_metadata_payload(
            [{"chunk_id": 0, "source_file": "a.pdf", "text": "hello"}],
            embedding_model="m",
            chunk_size=256,
            chunk_overlap=50,
        )
        path = tmp_path / "chunks_metadata.json"
        faiss_store.write_metadata(path, payload)

        loaded = faiss_store.load_metadata(path)
        assert loaded == payload
        assert loaded["total_chunks"] == 1

    def test_vietnamese_text_is_not_escaped(self, tmp_path):
        payload = faiss_store.build_metadata_payload(
            [{"chunk_id": 0, "source_file": "a.pdf", "text": "Thị trường chứng khoán"}],
            embedding_model="m",
        )
        path = tmp_path / "meta.json"
        faiss_store.write_metadata(path, payload)
        assert "Thị trường" in path.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "payload, expected",
        [
            ({}, "missing 'chunks' list"),
            ({"chunks": [{}]}, "missing required key"),
            ({"chunks": ["text instead of object"]}, "expected object"),
        ],
    )
    def test_invalid_metadata_is_rejected(self, tmp_path, payload, expected):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            faiss_store.load_metadata(path)

    def test_missing_metadata_explains_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Build index first"):
            faiss_store.load_metadata(tmp_path / "absent.json")


class TestSearch:
    def test_scores_and_positions_stay_paired(self, vector_store):
        index = faiss.read_index(str(vector_store["index_path"]))
        model = vector_store["model"]
        matches = faiss_store.search_with_scores(
            index, faiss_store.embed_query(model, vector_store["chunks"][0]["text"]), top_k=3
        )

        assert len(matches) == 3
        assert matches[0][0] == 0
        scores = [score for _, score in matches]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_clamped_to_index_size(self, vector_store):
        index = faiss.read_index(str(vector_store["index_path"]))
        matches = faiss_store.search_with_scores(
            index, faiss_store.embed_query(vector_store["model"], "anything"), top_k=99
        )
        assert len(matches) == len(vector_store["chunks"])

    def test_non_positive_top_k_returns_nothing(self, vector_store):
        index = faiss.read_index(str(vector_store["index_path"]))
        assert faiss_store.search_with_scores(index, np.ones((1, 64), np.float32), 0) == []
        assert faiss_store.search_positions(index, np.ones((1, 64), np.float32), 0) == []

    def test_query_embedding_is_normalized(self, vector_store):
        vector = faiss_store.embed_query(vector_store["model"], "force mass acceleration")
        assert vector.dtype == np.float32
        assert np.linalg.norm(vector) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# embedder
# ---------------------------------------------------------------------------
class TestChunking:
    def test_chunks_overlap_by_the_requested_amount(self):
        model = FakeEmbeddingModel()
        words = [f"w{i}" for i in range(600)]
        text = " ".join(words)

        chunks = embedder.chunk_text_by_tokens(text, model, chunk_size=256, overlap=10)
        assert len(chunks) == 3

        first, second = chunks[0].split(), chunks[1].split()
        assert first == words[:256]
        # The tail of one chunk repeats at the head of the next, so no context is
        # lost at the boundary.
        assert second[:10] == first[-10:]
        assert second == words[246:502]

    def test_short_text_yields_a_single_chunk(self):
        model = FakeEmbeddingModel()
        assert embedder.chunk_text_by_tokens("one two three", model, chunk_size=256, overlap=0) == [
            "one two three"
        ]

    def test_empty_text_yields_no_chunks(self):
        assert embedder.chunk_text_by_tokens("   ", FakeEmbeddingModel(), 256, 0) == []

    @pytest.mark.parametrize("size, overlap", [(255, 50), (513, 50), (256, 256), (256, -1)])
    def test_spec_bounds_are_enforced(self, size, overlap):
        with pytest.raises(ValueError):
            embedder.chunk_text_by_tokens("text", FakeEmbeddingModel(), size, overlap)

    def test_whitespace_is_normalized_within_chunks(self):
        chunks = embedder.chunk_text_by_tokens(
            "a\t\tb\n\n  c", FakeEmbeddingModel(), chunk_size=256, overlap=0
        )
        assert chunks == ["a b c"]

    def test_normalize_whitespace(self):
        assert embedder.normalize_whitespace("  x \n y\tz ") == "x y z"


class TestFileCollection:
    def test_only_supported_documents_are_collected(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"x")
        (tmp_path / "b.docx").write_bytes(b"x")
        (tmp_path / "c.txt").write_text("skip me", encoding="utf-8")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "d.PDF").write_bytes(b"x")

        collected = embedder.collect_source_files(tmp_path)
        assert [p.name for p in collected] == ["a.pdf", "b.docx", "d.PDF"]

    def test_missing_directory_warns_and_returns_empty(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            assert embedder.collect_source_files(tmp_path / "absent") == []
        assert "does not exist" in caplog.text


class TestExtractionDispatch:
    def test_unsupported_suffix_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported file type"):
            embedder.extract_text(tmp_path / "notes.md")

    def test_pdf_without_text_raises(self, tmp_path, monkeypatch):
        class NoTextReader:
            is_encrypted = False
            pages: list = []

        monkeypatch.setattr("pypdf.PdfReader", lambda path: NoTextReader())
        with pytest.raises(ValueError, match="No extractable text found in PDF"):
            embedder.parse_pdf_text(tmp_path / "empty.pdf")

    def test_docx_without_text_raises(self, tmp_path, monkeypatch):
        class EmptyDoc:
            paragraphs: list = []

        monkeypatch.setattr("docx.Document", lambda path: EmptyDoc())
        with pytest.raises(ValueError, match="No extractable text found in DOCX"):
            embedder.parse_docx_text(tmp_path / "empty.docx")


class TestBuildFaissIndex:
    @pytest.fixture
    def documents(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "docs"
        data_dir.mkdir()
        (data_dir / "lesson.pdf").write_bytes(b"fake pdf")
        monkeypatch.setattr(
            embedder,
            "extract_text",
            lambda path: (
                "Newton second law force equals mass times acceleration " * 3
                if path.name == "lesson.pdf"
                else "Photosynthesis converts light into glucose and oxygen"
            ),
        )
        return data_dir

    def test_index_and_metadata_are_written_consistently(
        self, documents, tmp_path, monkeypatch, fake_model
    ):
        monkeypatch.setattr(faiss_store, "get_embedding_model", lambda name=None: fake_model)
        import config

        vector_dir = tmp_path / "out"
        monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(vector_dir / "faiss_index.bin"))

        index_path, metadata_path, count = embedder.build_faiss_index(
            data_dir=documents,
            vector_dir=vector_dir,
            embedding_model_name="fake-model",
            chunk_size=256,
            overlap=50,
        )

        assert index_path == vector_dir / "faiss_index.bin"
        assert metadata_path == vector_dir / "chunks_metadata.json"
        assert count >= 1

        metadata = faiss_store.load_metadata(metadata_path)
        assert metadata["total_chunks"] == count
        assert metadata["embedding_model"] == "fake-model"
        assert all(chunk["source_file"].endswith("lesson.pdf") for chunk in metadata["chunks"])
        assert all(chunk["chunk_id"] == position for position, chunk in enumerate(metadata["chunks"]))

        index = faiss_store.read_index(index_path)
        assert index.ntotal == count, "index size must match the metadata it ships with"

    def test_unreadable_documents_are_skipped(self, documents, tmp_path, monkeypatch, fake_model):
        monkeypatch.setattr(faiss_store, "get_embedding_model", lambda name=None: fake_model)
        monkeypatch.setattr(embedder, "collect_source_files", lambda path: [path / "broken.pdf"])

        def flaky(path):
            raise ValueError("corrupted file")

        monkeypatch.setattr(embedder, "extract_text", flaky)
        with pytest.raises(ValueError, match="No valid chunks"):
            embedder.build_faiss_index(data_dir=documents, vector_dir=tmp_path / "out2")

    def test_an_absolute_index_path_wins_over_the_vector_dir(
        self, documents, tmp_path, monkeypatch, fake_model
    ):
        """Documented precedence: config points the index, vector_dir only the fallback."""
        import config

        monkeypatch.setattr(faiss_store, "get_embedding_model", lambda name=None: fake_model)
        elsewhere = tmp_path / "configured"
        monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(elsewhere / "faiss_index.bin"))

        index_path, metadata_path, _ = embedder.build_faiss_index(
            data_dir=documents, vector_dir=tmp_path / "ignored", chunk_size=256, overlap=50
        )

        assert index_path.parent == metadata_path.parent == elsewhere

    def test_no_documents_raises_with_the_searched_path(self, tmp_path, monkeypatch, fake_model):
        monkeypatch.setattr(faiss_store, "get_embedding_model", lambda name=None: fake_model)
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(FileNotFoundError, match=str(empty.resolve())):
            embedder.build_faiss_index(data_dir=empty, vector_dir=tmp_path / "out3")


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------
class TestRetrieval:
    def test_relevant_chunk_wins(self, point_config_at_vector_store):
        hits = retriever.search("force equals mass times acceleration", top_k=2)

        assert hits, "expected retrieval results"
        assert "Newton" in hits[0].text
        assert hits[0].score >= hits[-1].score

    def test_retrieve_returns_texts_only(self, point_config_at_vector_store):
        texts = retriever.retrieve("photosynthesis light glucose", top_k=1)
        assert len(texts) == 1 and "Photosynthesis" in texts[0]

    def test_retrieve_with_sources_returns_bare_filenames(self, point_config_at_vector_store):
        results = retriever.retrieve_with_sources("queue stack first in first out", top_k=1)
        assert len(results) == 1
        assert set(results[0]) == {"text", "source_file"}
        assert "/" not in results[0]["source_file"]
        assert results[0]["source_file"].endswith(".pdf")

    def test_both_entry_points_agree(self, point_config_at_vector_store):
        query = "water cycle evaporation condensation"
        assert retriever.retrieve(query, top_k=3) == [
            hit["text"] for hit in retriever.retrieve_with_sources(query, top_k=3)
        ]

    def test_top_k_larger_than_the_index_is_clamped(self, point_config_at_vector_store):
        assert len(retriever.retrieve("force", top_k=50)) == 4

    def test_blank_query_is_rejected(self, point_config_at_vector_store):
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.retrieve("   ")

    def test_non_positive_top_k_is_rejected(self, point_config_at_vector_store):
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve("force", top_k=0)

    def test_missing_index_raises_actionable_error(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "absent" / "index.bin"))
        monkeypatch.setattr(config, "VECTOR_DIR", str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="embedder.py"):
            retriever.retrieve("anything")

    def test_out_of_range_positions_are_skipped(self, point_config_at_vector_store, monkeypatch):
        vector_store = point_config_at_vector_store
        metadata = json.loads(Path(vector_store["metadata_path"]).read_text(encoding="utf-8"))
        metadata["chunks"] = metadata["chunks"][:1]  # index has 4 vectors, metadata lists 1
        Path(vector_store["metadata_path"]).write_text(json.dumps(metadata), encoding="utf-8")

        # Truncated metadata must not crash retrieval: only the valid chunk survives.
        hits = retriever.search("force mass acceleration", top_k=4)

        assert len(hits) == 1
        assert "Newton" in hits[0].text


# ---------------------------------------------------------------------------
# rag_tester
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_precision_counts_only_relevant_hits(self):
        assert rag_tester.precision_at_k([1, 2, 3], {1, 3}, top_k=3) == pytest.approx(2 / 3)
        assert rag_tester.precision_at_k([], {1}, top_k=3) == 0.0
        assert rag_tester.precision_at_k([9, 9], {1}, top_k=1) == 0.0

    def test_precision_ignores_results_beyond_k(self):
        assert rag_tester.precision_at_k([1, 2, 3], {3}, top_k=2) == 0.0

    def test_reciprocal_rank_uses_the_first_hit(self):
        assert rag_tester.reciprocal_rank([1, 2, 3], {2}) == pytest.approx(0.5)
        assert rag_tester.reciprocal_rank([1, 2, 3], {9}) == 0.0

    def test_mean(self):
        assert rag_tester.mean([0.2, 0.4]) == pytest.approx(0.3)
        assert rag_tester.mean([]) == 0.0


class TestQuerySynthesis:
    def test_query_is_built_from_the_first_meaningful_words(self):
        text = "The quick brown fox jumps over a lazy dog while cats sleep"
        query = rag_tester.build_query_from_chunk_text(text)

        assert query.startswith("What does the document explain about:")
        # Short filler words are dropped, and the same text always yields the same
        # query (the evaluator promises a deterministic test set).
        assert "quick brown" in query and " a " not in query
        assert query == rag_tester.build_query_from_chunk_text(text)

    def test_noisy_text_gets_the_generic_query(self):
        assert "study material" in rag_tester.build_query_from_chunk_text("1 2 3 !!!")

    def test_selection_spreads_across_the_corpus(self):
        indices = rag_tester.select_test_chunk_indices(100, 20)
        assert len(indices) == 20 == len(set(indices))
        assert indices[0] == 0 and indices[-1] < 100
        assert indices == sorted(indices)

    def test_too_few_chunks_is_reported(self):
        with pytest.raises(ValueError, match="Need at least 20 indexed chunks"):
            rag_tester.select_test_chunk_indices(5, 20)

    def test_build_test_cases_marks_same_document_chunks_relevant(self):
        chunks = [
            {"chunk_id": i, "source_file": f"doc{i % 2}.pdf", "text": f"alpha{i} beta{i} gamma{i}"}
            for i in range(20)
        ]
        cases = rag_tester.build_test_cases({"chunks": chunks}, test_count=20)

        assert len(cases) == 20
        for case in cases:
            assert case.target_chunk_id in case.relevant_chunk_ids
            # Relevance is document-wide, so the sibling chunk of the same doc counts.
            sibling = case.target_chunk_id + 2
            if sibling < 20:
                assert sibling in case.relevant_chunk_ids


class TestEvaluation:
    def test_evaluate_reports_perfect_scores_on_a_matching_corpus(self, point_config_at_vector_store, monkeypatch):
        vector_store = point_config_at_vector_store
        metadata = json.loads(Path(vector_store["metadata_path"]).read_text(encoding="utf-8"))

        cases = [
            rag_tester.RagTestCase(
                query=chunk["text"],
                target_chunk_id=chunk["chunk_id"],
                relevant_chunk_ids={chunk["chunk_id"]},
                source_file=chunk["source_file"],
            )
            for chunk in metadata["chunks"]
        ]

        monkeypatch.setattr(
            faiss_store,
            "resolve_index_path",
            lambda *a, **k: vector_store["index_path"],
        )
        precision, mrr, details = rag_tester.evaluate_retrieval(cases, metadata, top_k=1)

        assert precision == pytest.approx(1.0)
        assert mrr == pytest.approx(1.0)
        assert len(details) == 4
        assert {key for key in details[0]} == {
            "test_id",
            "query",
            "target_chunk_id",
            "source_file",
            "retrieved_chunk_ids",
            "relevant_hits",
            "precision_at_k",
            "reciprocal_rank",
        }

    def test_cli_rejects_a_non_standard_test_count(self):
        with pytest.raises(ValueError, match="exactly 20"):
            rag_tester.main(["--test-count", "5"])

    def test_cli_accepts_the_documented_defaults(self, capsys):
        # --test-count 20 is required by the spec; check the parser, not the index.
        args = rag_tester.parse_args(["--test-count", "20", "--top-k", "3"])
        assert (args.test_count, args.top_k) == (20, 3)

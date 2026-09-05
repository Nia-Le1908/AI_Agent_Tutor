"""
Embedding pipeline: documents -> chunks -> FAISS index + metadata.

Steps:
1. Read source documents from the data directory (PDF and DOCX).
2. Clean and normalize the extracted text.
3. Split text into overlapping token-based chunks using the same tokenizer family
   as the embedding model, so chunk boundaries stay meaningful to the model.
4. Embed chunks with sentence-transformers.
5. Persist a FAISS index plus chunk metadata through :mod:`faiss_store`.

Design goals:
- Robust behaviour with actionable errors.
- Deterministic output where possible.
- Index/metadata layout owned by faiss_store, so retriever and rag_tester read
  exactly what this module writes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

import faiss_store
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    VECTOR_DIR,
)
from validation import require_int_in_range

# Document types the pipeline understands. Kept as a constant so the collector and
# the extraction dispatcher cannot disagree about what is "supported".
SUPPORTED_SUFFIXES = {".pdf", ".docx"}

DEFAULT_DATA_DIR = Path(DATA_DIR)
DEFAULT_VECTOR_DIR = Path(VECTOR_DIR)
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_NAME
DEFAULT_METADATA_PATH = DEFAULT_VECTOR_DIR / faiss_store.METADATA_FILENAME

# Chunking bounds mirrored from the spec; config.py validates the env values.
MIN_CHUNK_SIZE = 256
MAX_CHUNK_SIZE = 512

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """A single text chunk and its provenance metadata."""

    chunk_id: int
    source_file: str
    text: str


def normalize_whitespace(text: str) -> str:
    """
    Collapse whitespace runs into single spaces.

    PDF extraction inserts irregular line breaks and spacing, which both hurts
    embedding quality and makes chunk sizes unpredictable.
    """
    return " ".join(text.split())


def parse_pdf_text(file_path: Path) -> str:
    """
    Extract normalized text from a PDF.

    Handles the common PDF issues: encrypted files (empty-password decrypt is
    attempted) and pages with no extractable text.

    Raises:
        ValueError: when no usable text can be extracted.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"Cannot decrypt PDF: {file_path}") from exc

    page_texts = [
        cleaned
        for cleaned in (normalize_whitespace(page.extract_text() or "") for page in reader.pages)
        if cleaned
    ]

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ValueError(f"No extractable text found in PDF: {file_path}")
    return full_text


def parse_docx_text(file_path: Path) -> str:
    """Extract text from a DOCX file with paragraph-level normalization."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [normalize_whitespace(p.text) for p in doc.paragraphs if p.text and p.text.strip()]

    full_text = "\n".join(paragraphs).strip()
    if not full_text:
        raise ValueError(f"No extractable text found in DOCX: {file_path}")
    return full_text


def extract_text(file_path: Path) -> str:
    """
    Extract text from any supported document type.

    Raises:
        ValueError: for an unsupported suffix.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_text(Path(file_path))
    if suffix == ".docx":
        return parse_docx_text(Path(file_path))
    raise ValueError(f"Unsupported file type: {file_path}")


# Historical alias kept for callers documented in interfaces.md.
_extract_text_for_file = extract_text


def chunk_text_by_tokens(
    text: str,
    model: Any,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into token-based overlapping chunks.

    Args:
        text: Source text.
        model: embedding model exposing ``.tokenizer`` (duck-typed so tests can
            pass a fake tokenizer).
        chunk_size: Required to be in [256, 512].
        overlap: Token overlap between consecutive chunks; must be < chunk_size.

    Returns:
        List of chunk strings, in document order.

    Raises:
        ValueError: when the sizing arguments violate the spec.
    """
    chunk_size = require_int_in_range(chunk_size, "chunk_size", MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
    overlap = require_int_in_range(overlap, "overlap", 0, chunk_size - 1)

    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []

    # Walk the token stream with a stride of (size - overlap). Decoding each window
    # keeps chunk text aligned to model tokens rather than arbitrary characters.
    for start in range(0, len(token_ids), step):
        end = start + chunk_size
        window_ids = token_ids[start:end]
        if not window_ids:
            continue

        chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(normalize_whitespace(chunk_text))

        if end >= len(token_ids):
            break

    return chunks


def collect_source_files(data_dir: Path) -> List[Path]:
    """Collect supported source files recursively, in deterministic sorted order."""
    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return []

    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def build_chunk_records(
    source_files: List[Path],
    model: Any,
    *,
    chunk_size: int,
    overlap: int,
) -> List[ChunkRecord]:
    """
    Turn documents into numbered chunk records, skipping unreadable files.

    One corrupt PDF should not abort an index rebuild, so extraction failures are
    logged and skipped.
    """
    records: List[ChunkRecord] = []

    for file_path in source_files:
        try:
            raw_text = extract_text(file_path)
        except Exception as exc:  # noqa: BLE001 - per-file resilience is intentional
            logger.warning("Skipping unreadable file %s due to: %s", file_path, exc)
            continue

        for chunk in chunk_text_by_tokens(raw_text, model, chunk_size=chunk_size, overlap=overlap):
            records.append(
                ChunkRecord(
                    chunk_id=len(records),
                    source_file=file_path.as_posix(),
                    text=chunk,
                )
            )

    return records


def build_faiss_index(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    vector_dir: Path | str = DEFAULT_VECTOR_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Tuple[Path, Path, int]:
    """
    Build and persist the FAISS index plus chunk metadata.

    Returns:
        (index_path, metadata_path, total_chunks)

    Raises:
        FileNotFoundError: when no supported documents exist under ``data_dir``.
        ValueError: when the documents yield no usable chunks.
    """
    data_path = Path(data_dir)
    vector_path = Path(vector_dir)
    vector_path.mkdir(parents=True, exist_ok=True)

    index_path = faiss_store.resolve_index_path(vector_dir=vector_path)
    metadata_path = index_path.parent / faiss_store.METADATA_FILENAME

    model = faiss_store.get_embedding_model(embedding_model_name)

    source_files = collect_source_files(data_path)
    if not source_files:
        raise FileNotFoundError(
            f"No supported documents (.pdf/.docx) found under: {data_path.resolve()}"
        )

    chunk_records = build_chunk_records(
        source_files, model, chunk_size=chunk_size, overlap=overlap
    )
    if not chunk_records:
        raise ValueError("No valid chunks were generated from input documents.")

    embeddings = model.encode(
        [record.text for record in chunk_records],
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    index = _build_index(embeddings)
    faiss_store.write_index(index, index_path)

    faiss_store.write_metadata(
        metadata_path,
        faiss_store.build_metadata_payload(
            [asdict(record) for record in chunk_records],
            embedding_model=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
    )

    logger.info("Built FAISS index at %s with %d chunks", index_path, len(chunk_records))
    logger.info("Saved chunk metadata at %s", metadata_path)

    return index_path, metadata_path, len(chunk_records)


def _build_index(embeddings: np.ndarray) -> Any:
    """
    Create a flat inner-product index over normalized vectors.

    Vectors are L2-normalized before indexing, which makes inner product equal to
    cosine similarity while staying exact (no ANN approximation to tune).
    """
    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    idx_path, meta_path, n_chunks = build_faiss_index()
    print(f"Index: {idx_path}")
    print(f"Metadata: {meta_path}")
    print(f"Chunks: {n_chunks}")

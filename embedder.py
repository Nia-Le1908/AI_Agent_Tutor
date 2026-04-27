"""
Phase 2 - Embedding pipeline for AI Tutor V5.1.

This module is responsible for:
1. Reading source documents from the data directory (PDF and DOCX).
2. Cleaning and normalizing extracted text.
3. Splitting text into overlapping token-based chunks using the same tokenizer
   family as the embedding model to keep chunk boundaries meaningful.
4. Embedding chunks with sentence-transformers (all-MiniLM-L6-v2).
5. Building and persisting a FAISS index plus chunk metadata.

Design goals:
- Robust behavior with actionable errors.
- Deterministic output where possible.
- Clean interfaces for controller/retriever integration.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Optional config import. If the config module is unavailable or does not expose
# one of these names yet, we safely fall back to defaults.
try:
    from config import CHUNK_OVERLAP, CHUNK_SIZE, FAISS_INDEX_PATH
except Exception:  # pragma: no cover
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 50
    FAISS_INDEX_PATH = "vector_store/faiss_index.bin"


DEFAULT_DATA_DIR = Path("data")
DEFAULT_VECTOR_DIR = Path("vector_store")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_METADATA_PATH = DEFAULT_VECTOR_DIR / "chunks_metadata.json"

logger = logging.getLogger(__name__)


def _safe_write_faiss_index(index: faiss.Index, index_path: Path) -> None:
    """
    Persist FAISS index with a Windows-safe fallback.

    On some Windows setups, FAISS C++ file APIs fail for Unicode paths
    (for example directories containing accented characters). We first try the
    direct path, then fallback to writing in an ASCII temp directory and copy
    the resulting binary to the target path via Python.
    """
    try:
        faiss.write_index(index, str(index_path))
        return
    except RuntimeError as exc:
        logger.warning("Direct FAISS write failed, using temp fallback: %s", exc)

    temp_dir = Path(tempfile.gettempdir()) / "ai_tutor_faiss_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_index_path = temp_dir / "faiss_index.bin"

    faiss.write_index(index, str(temp_index_path))
    shutil.copyfile(temp_index_path, index_path)


@dataclass
class ChunkRecord:
    """A single text chunk and its provenance metadata."""

    chunk_id: int
    source_file: str
    text: str


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace to improve chunk quality.

    Why this matters:
    - PDF extraction often inserts irregular line breaks and spacing.
    - Cleaner text produces better embeddings and retrieval precision.
    """
    return " ".join(text.split())


def parse_pdf_text(file_path: Path) -> str:
    """
    Extract raw text from a PDF file.

    The function handles common PDF issues:
    - encrypted files (attempts empty-password decrypt)
    - pages with no extractable text

    Raises:
        ValueError: when no usable text can be extracted.
    """
    reader = PdfReader(str(file_path))

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"Cannot decrypt PDF: {file_path}") from exc

    page_texts: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        cleaned = normalize_whitespace(extracted)
        if cleaned:
            page_texts.append(cleaned)

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ValueError(f"No extractable text found in PDF: {file_path}")
    return full_text


def parse_docx_text(file_path: Path) -> str:
    """Extract text from a DOCX file with paragraph-level normalization."""
    doc = Document(str(file_path))
    paragraphs = [normalize_whitespace(p.text) for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs).strip()
    if not full_text:
        raise ValueError(f"No extractable text found in DOCX: {file_path}")
    return full_text


def chunk_text_by_tokens(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into token-based overlapping chunks.

    Requirements enforced from spec:
    - chunk_size must be in [256, 512]
    - overlap defaults to 50

    Implementation detail:
    - We chunk in embedding-model token space by using the model tokenizer.
    - This keeps each chunk near the desired semantic size for embedding.
    """
    if chunk_size < 256 or chunk_size > 512:
        raise ValueError(f"chunk_size must be in [256, 512], got {chunk_size}")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []

    # Walk token stream with overlap. Decoding each token window yields text
    # that remains aligned to model tokenization.
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


def _collect_source_files(data_dir: Path) -> List[Path]:
    """Collect supported source files recursively from the data folder."""
    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return []

    supported_suffixes = {".pdf", ".docx"}
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported_suffixes]
    return sorted(files)


def _extract_text_for_file(file_path: Path) -> str:
    """Dispatch text extraction by file type."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_text(file_path)
    if suffix == ".docx":
        return parse_docx_text(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def build_faiss_index(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    vector_dir: Path | str = DEFAULT_VECTOR_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Tuple[Path, Path, int]:
    """
    Build and persist FAISS index + metadata.

    Returns:
        (index_path, metadata_path, total_chunks)
    """
    data_dir = Path(data_dir)
    vector_dir = Path(vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)

    # Keep index filename aligned with project conventions.
    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.is_absolute():
        index_path = vector_dir / index_path.name

    metadata_path = vector_dir / DEFAULT_METADATA_PATH.name

    # Ensure output directories exist before attempting file writes.
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(embedding_model_name)

    source_files = _collect_source_files(data_dir)
    if not source_files:
        raise FileNotFoundError(
            f"No supported documents (.pdf/.docx) found under: {data_dir.resolve()}"
        )

    chunk_records: List[ChunkRecord] = []
    next_chunk_id = 0

    for file_path in source_files:
        try:
            raw_text = _extract_text_for_file(file_path)
        except Exception as exc:
            logger.warning("Skipping unreadable file %s due to: %s", file_path, exc)
            continue

        chunks = chunk_text_by_tokens(
            text=raw_text,
            model=model,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            chunk_records.append(
                ChunkRecord(
                    chunk_id=next_chunk_id,
                    source_file=str(file_path.as_posix()),
                    text=chunk,
                )
            )
            next_chunk_id += 1

    if not chunk_records:
        raise ValueError("No valid chunks were generated from input documents.")

    chunk_texts = [record.text for record in chunk_records]

    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    # FAISS expects float32 contiguous arrays.
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Use inner product because vectors are normalized -> equivalent to cosine similarity.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _safe_write_faiss_index(index, index_path)

    metadata_payload = {
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "total_chunks": len(chunk_records),
        "chunks": [record.__dict__ for record in chunk_records],
    }
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Built FAISS index at %s with %d chunks", index_path, len(chunk_records))
    logger.info("Saved chunk metadata at %s", metadata_path)

    return index_path, metadata_path, len(chunk_records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    idx_path, meta_path, n_chunks = build_faiss_index()
    print(f"Index: {idx_path}")
    print(f"Metadata: {meta_path}")
    print(f"Chunks: {n_chunks}")

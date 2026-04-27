# AI Tutor V5.1 - Internal Interfaces Contract

This document defines exact callable interfaces shared across modules.
It is intended to reduce integration errors and establish stable boundaries.

## Type Aliases (Conceptual)

- QuestionDict: Dict[str, Any]
- WeakTopicsDict: Dict[str, Dict[str, float]]
- ChatMessage: Dict[str, str]

## controller.py

### chat

Signature:

```python
def chat(user_input: str, top_k: int = TOP_K, model_name: str = DEFAULT_MODEL) -> str
```

Contract:
- Input:
  - user_input: non-empty user message.
  - top_k: number of retrieved chunks to inject into prompt.
  - model_name: Gemini model identifier.
- Output:
  - Assistant response text.
- Raises:
  - ValueError for invalid input.
  - RuntimeError for exhausted Gemini retries.

### generate_exercise_for_user

Signature:

```python
def generate_exercise_for_user(uid: int, topic: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]
```

Contract:
- Input:
  - uid: user id.
  - topic: target topic/subject.
  - model_name: Gemini model identifier.
- Output:
  - Schema-validated question dict.
- Depends on:
  - adaptive_logic.get_next_difficulty
  - generator.generate

## sqlite_manager.py

### save_history

Signature:

```python
def save_history(uid: int, qid: int, is_correct: bool) -> None
```

Contract:
- Input:
  - uid: user id.
  - qid: question id.
  - is_correct: correctness flag.
- Side effects:
  - Inserts one row into history(uid, qid, is_correct, timestamp).

### get_question_by_diff

Signature:

```python
def get_question_by_diff(level: int) -> List[Dict[str, Any]]
```

Contract:
- Input:
  - level in range [1, 5].
- Output:
  - List of question row dicts with keys:
    - id, content, difficulty, subject, options, answer, explanation

### get_weak_topics

Signature:

```python
def get_weak_topics(uid: int) -> Dict[str, Dict[str, float]]
```

Contract:
- Input:
  - uid: user id.
- Output:
  - Dict keyed by subject, each value:
    - correct: float
    - incorrect: float
    - total: float
    - accuracy: float in [0.0, 1.0]

## generator.py

### generate

Signature:

```python
def generate(topic: str, difficulty: int, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]
```

Contract:
- Input:
  - topic: non-empty string.
  - difficulty: integer in [1, 5].
  - model_name: Gemini model name.
- Output:
  - Strict JSON dict validated against schema.json.
- Guarantees:
  - Prompt includes strict JSON instruction.
  - Defensive parsing strips markdown fences and extracts JSON object.
  - Exponential backoff is applied to Gemini calls.

## retriever.py

### retrieve

Signature:

```python
def retrieve(query: str, top_k: int = TOP_K) -> List[str]
```

Contract:
- Input:
  - query: non-empty search query string.
  - top_k: positive integer.
- Output:
  - Top-k retrieved chunk texts from FAISS metadata.

## adaptive_logic.py

### get_next_difficulty

Signature:

```python
def get_next_difficulty(uid: int) -> int
```

Contract:
- Input:
  - uid: user id.
- Output:
  - Next level clamped to [1, 5].
- Rule set:
  - 3 correct in a row => +1
  - 2 incorrect in a row => -1
  - otherwise unchanged

## json_parser.py

### parse_and_insert

Signature:

```python
def parse_and_insert(json_str: str) -> int
```

Contract:
- Input:
  - json_str: raw JSON from LLM.
- Output:
  - Inserted question row id.
- Validation behavior:
  - Raises KeyError for missing required fields.
  - Strict Draft 2020-12 schema validation against schema.json.
- Side effects:
  - Inserts validated question into questions table.

## embedder.py

### parse_pdf_text

Signature:

```python
def parse_pdf_text(file_path: Path) -> str
```

Contract:
- Input:
  - PDF path.
- Output:
  - Extracted normalized text.

### parse_docx_text

Signature:

```python
def parse_docx_text(file_path: Path) -> str
```

Contract:
- Input:
  - DOCX path.
- Output:
  - Extracted normalized text.

### chunk_text_by_tokens

Signature:

```python
def chunk_text_by_tokens(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]
```

Contract:
- Input:
  - text: source text.
  - model: embedding model with tokenizer.
  - chunk_size: integer in [256, 512].
  - overlap: integer, default 50, must be < chunk_size.
- Output:
  - Overlapping token-aligned text chunks.

### build_faiss_index

Signature:

```python
def build_faiss_index(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    vector_dir: Path | str = DEFAULT_VECTOR_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Tuple[Path, Path, int]
```

Contract:
- Output tuple:
  - index_path: Path
  - metadata_path: Path
  - total_chunks: int

## dashboard.py

### render_dashboard

Signature:

```python
def render_dashboard(uid: int) -> None
```

Contract:
- Input:
  - uid: user id.
- Side effects:
  - Renders 3 Plotly charts into Streamlit container.

## init_db.py

### initialize_database

Signature:

```python
def initialize_database() -> Path
```

Contract:
- Output:
  - Absolute path to initialized SQLite database.
- Side effects:
  - Applies schema.sql and validates required tables.

---

Change policy:
- Any signature change in these public functions must update this file in the same pull request.

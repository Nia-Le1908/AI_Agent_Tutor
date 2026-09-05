# AI Tutor V5.1 - Internal Interfaces Contract

This document defines exact callable interfaces shared across modules.
It is intended to reduce integration errors and establish stable boundaries.

Layering rules that the signatures below follow:

- Only `config.py` reads the environment; `config_db.py` is a facade over it.
- Only `sqlite_manager.py` / `auth.py` / `init_db.py` / `seed_db.py` / `generate_mock_data.py`
  write SQL, and only through `db_manager.py` helpers. UI modules call functions, never SQL.
- Only `llm_client.py` talks to the provider; only `faiss_store.py` touches FAISS paths/models.

## Type Aliases (Conceptual)

- QuestionDict: Dict[str, Any] - a question validated against schema.json
  (keys: question_id, content, difficulty, subject, options, answer, explanation).
- StoredQuestionDict: Dict[str, Any] - a row of the questions table. Same keys, except
  `id` is added and `options` is the JSON **string** stored in the column.
- WeakTopicsDict: Dict[str, Dict[str, float]]
- ChatMessage: Dict[str, str] - {"role": "user"|"assistant", "content": str}
- Row: Dict[str, Any] - every DB row is returned as a dict, never a tuple (both backends).

## config.py

### require_deepseek_api_key

```python
def require_deepseek_api_key() -> str
```

Contract:
- Output: `DEEPSEEK_API_KEY`, stripped.
- Raises:
  - `ConfigError`: unset/blank, or a value that still matches the `.env.example` placeholder.
- Note: all module constants (`DB_PATH`, `CHUNK_SIZE`, `TOP_K`, `FAISS_INDEX_PATH`, ...) are
  validated at import and raise `ConfigError` naming the offending variable.

### resolve_path / ensure_runtime_dirs

```python
def resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path
def ensure_runtime_dirs() -> None
```

Contract:
- `resolve_path`: relative values are anchored to `PROJECT_ROOT`; absolute values pass through.
- `ensure_runtime_dirs`: creates the parent directories of DB_PATH, VECTOR_DIR, DATA_DIR, LOG_PATH.

## validation.py

```python
def clamp(value: int, low: int, high: int) -> int
def require_int_in_range(value: object, name: str, low: int, high: int) -> int
def require_positive_int(value: object, name: str) -> int
def require_level(value: object, name: str = "level") -> int
def require_non_empty_str(value: object, name: str, *, max_length: int | None = None) -> str
def require_bool(value: object, name: str) -> bool
```

Contract:
- Shared by every layer so error wording stays uniform.
- Raises `TypeError` for the wrong type, `ValueError` for an out-of-range or blank value.
- `require_level` enforces the project range [1, 5]; `require_bool` rejects 1/0 on purpose.
- Input is never coerced silently: `int`-typed flags and float levels are errors, not rounding.

## schemas.py

```python
def load_schema(schema_path: Path | str | None = None) -> Dict[str, Any]
def validate_question_payload(payload: Dict[str, Any]) -> None
def validate_questions(payloads: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]
def required_fields() -> List[str]
def options_to_json(options: Any) -> str
def normalize_options(options: Any) -> Dict[str, str]
def parse_options(options: Any) -> Dict[str, str]
def is_option_set_complete(option_map: Dict[str, str]) -> bool
```

Contract:
- `load_schema`: cached per process; a corrupt schema.json is a `SchemaValidationError`
  listing every validator error; `check_schema` is verified against the metaschema.
- `validate_question_payload`: Draft 2020-12 validation, `additionalProperties: false`.
  Raises `SchemaValidationError` whose message lists every violation, never just the first.
- `validate_questions`: batch filter - keeps the valid payloads, logs and drops the rest.
  Raises `ValueError` only when nothing survives.
- `normalize_options`: accepts both accepted shapes - a 4-item array (`["a","b","c","d"]`)
  and the canonical DB object (`{"A": "a", ...}`) - and always returns `{"A","B","C","D"}`.
  Raises `ValueError` when the shape or the letters do not line up.
- `options_to_json`: canonical JSON text for the `questions.options` column (object form).
- `parse_options`: lenient UI-facing variant - returns `{}` instead of raising, so one
  malformed row can break the query, not the whole page.
- `is_option_set_complete`: True only when all four letters carry non-empty text.

## db_manager.py

```python
def to_backend_sql(sql: str, *, backend: str = DB_TYPE) -> str

class DatabaseError(RuntimeError)

class Result:
    def fetchone(self) -> Optional[Dict[str, Any]]
    def fetchall(self) -> List[Dict[str, Any]]
    lastrowid: int | None

class Connection:
    def raw(self) -> Any
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Result
    def executemany(self, sql: str, params_list: Iterable[Sequence[Any]]) -> int
    def commit(self) -> None
    def rollback(self) -> None
    def close(self) -> None

class DatabaseManager:
    def open_connection(self) -> Connection
    def connect(self) -> Iterator[Connection]
    def fetch_all(self, sql: str, params: Sequence[Any] | None = None) -> List[Dict[str, Any]]
    def fetch_one(self, sql: str, params: Sequence[Any] | None = None) -> Optional[Dict[str, Any]]
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int
    def execute_many(self, sql: str, params_list: Iterable[Sequence[Any]]) -> int
    def insert_returning_id(self, table: str, values: Mapping[str, Any]) -> int
    def insert_ignore(self, table: str, values: Mapping[str, Any]) -> bool
    def health_check(self) -> bool

def get_manager() -> DatabaseManager
def reset_manager_for_tests(manager: DatabaseManager | None = None) -> None
```

Module-level shortcuts forward to the process-wide manager, so callers never build
connections themselves:

```python
def get_db_connection() -> Connection
def execute_query(query: str, params: Optional[Tuple[Any, ...]] = None, fetch: bool = False) -> Optional[List[Dict[str, Any]]]
def fetch_all(query: str, params: Sequence[Any] | None = None) -> List[Dict[str, Any]]
def fetch_one(query: str, params: Sequence[Any] | None = None) -> Optional[Dict[str, Any]]
def execute(query: str, params: Sequence[Any] | None = None) -> int
def execute_many(query: str, params_list: Iterable[Sequence[Any]]) -> int
def insert_returning_id(table: str, values: Mapping[str, Any]) -> int
def insert_ignore(table: str, values: Mapping[str, Any]) -> bool
def health_check() -> bool
```

Contract - portable SQL rules every caller must obey:
- Write `?` placeholders only; `to_backend_sql` rewrites them to `%s` on PostgreSQL.
- Never write `INSERT OR IGNORE`; call `insert_ignore` (PostgreSQL has no such syntax).
- Never read `cursor.lastrowid` yourself: `Result.lastrowid` is populated per backend
  (`RETURNING id` on PostgreSQL, `lastrowid` on SQLite).
- Never put SQLite date functions (`datetime('now','localtime')`, `julianday(...)`) in a
  query: either leave a `timestamp` column to its schema default (`CURRENT_TIMESTAMP`, as
  `save_history` does) or pass a Python-side tz-aware value (`auth._now()`).
- Order by `id`, never by `rowid` (PostgreSQL has no `rowid`).
- Every failure surfaces as `DatabaseError` with the driver message attached; `connect()`
  commits on success and rolls back on error.
- PostgreSQL connections come from a lazily built `psycopg2` pool and are returned to it
  on `close()`; SQLite connections are closed. Missing `psycopg2` is an install-time
  message, not an ImportError at the call site.

## config_db.py

```python
class DatabaseConfig:
    db_type: str
    def postgres_password(self) -> str          # hidden from repr
    def get_connection_string(self) -> str
    def get_driver_name(self) -> Literal["sqlite", "postgresql"]
    def is_sqlite(self) -> bool
    def is_postgresql(self) -> bool

def get_db_type() -> str
def is_postgresql() -> bool
def is_sqlite() -> bool
```

Contract:
- Compatibility facade: it derives everything from `config.py` and re-parses nothing
  (no `load_dotenv`, no `os.getenv`), so it can never disagree with the app.
- `get_connection_string()` returns the libpq form for PostgreSQL and the SQLite file path
  otherwise; it is used by diagnostics, while `db_manager` opens connections itself.
- `repr()`/`str()` never include the password.

## sqlite_manager.py

```python
def get_user_level(uid: int) -> Optional[int]
def set_user_level(uid: int, level: int) -> None
def get_or_create_user(name: str) -> tuple[int, int]
def get_question_by_diff(level: int) -> List[Dict[str, Any]]
def get_questions_filtered(level: int, subject: str | None = None, exclude_uid: int | None = None) -> List[Dict[str, Any]]
def get_all_subjects() -> List[str]
def insert_question(question: Dict[str, Any]) -> int
def insert_questions(questions: List[Dict[str, Any]]) -> int
def save_history(uid: int, qid: int, is_correct: bool) -> None
def fetch_recent_outcomes(uid: int, limit: int = MAX_STREAK_HISTORY) -> List[int]
def count_consecutive(values: List[int], expected: int) -> int
def count_streak(outcomes: List[int], expected: int = 1) -> int
def get_user_stats(uid: int) -> Dict[str, Any]
def get_weak_topics(uid: int) -> Dict[str, Dict[str, float]]
def get_progress_timeline(uid: int) -> List[Dict[str, Any]]
def get_difficulty_scores(uid: int) -> List[Dict[str, Any]]
```

Contract:
- Input:
  - `uid`: positive int (validated by `require_positive_int`).
  - `level`: int in [1, 5] (validated by `require_level`).
- Output:
  - Question rows carry keys: id, content, difficulty, subject, options, answer, explanation
    (plus is_active where the table has it). `options` is the raw stored JSON string.
  - `get_user_level` returns None for an unknown user; the caller decides the default.
  - `get_or_create_user(name)` returns `(uid, level)`; the name is trimmed and matched
    exactly, creating the user at level 1 when absent.
  - `insert_questions` returns how many rows were written; invalid payloads are skipped and
    logged (partial success is intentional - one bad question must not lose the batch).
  - `fetch_recent_outcomes` returns flags newest-first (`1` correct, `0` incorrect);
    `count_streak` is the same rule without a DB round-trip.
- `get_weak_topics` values: correct / incorrect / total / accuracy (accuracy in [0.0, 1.0]).
- Question read paths do NOT filter on `is_active`: deactivating a question must not erase
  a learner's history with it (the shipped schema has no admin "list active only" query yet;
  adding one belongs here, not in the UI).

## llm_client.py

```python
class LLMError(RuntimeError)
class LLMRateLimitError(LLMError)
class LLMConfigurationError(LLMError)

def is_rate_limit_error(error: BaseException | str) -> bool

def chat(
    prompt: str = "",
    *,
    model: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    json_mode: bool = False,
    temperature: float = 0.7,
    max_attempts: int = 5,
) -> str
```

Contract:
- The only module that talks to the provider (OpenAI-compatible SDK, DeepSeek base URL).
- Input: either `prompt` (wrapped as one user message) or a full `messages` list; providing
  both is a `ValueError`.
- `json_mode=True` requests `response_format={"type": "json_object"}`.
- Retries with exponential backoff on transient failures (`LLM_BACKOFF_FACTOR`,
  `LLM_MAX_BACKOFF_SECONDS`); a rate-limit/quota response is detected by
  `is_rate_limit_error` and raised as `LLMRateLimitError` without burning the whole budget.
- Output: assistant text, markdown fences NOT stripped here (that is `json_parser`'s job).
- Raises:
  - `LLMConfigurationError`: missing API key or missing `openai` package.
  - `LLMRateLimitError` / `LLMError`: provider unavailable after retries.

## generator.py

```python
def build_prompt(topic: str, difficulty: int) -> str
def build_batch_prompt(topic: str, difficulty: int, count: int) -> str
def generate(topic: str, difficulty: int, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]
def generate_batch(topic: str, difficulty: int, count: int = 5, model_name: str = DEFAULT_MODEL) -> List[Dict[str, Any]]
```

Contract:
- Input: topic non-empty; difficulty in [1, 5]; `count` in [1, config.MAX_BATCH_SIZE].
- Prompt wire format is fixed (Vietnamese instructions, JSON-only reply) - the two builders
  are the only place it is defined, and tests compare them verbatim.
- Output: schema-validated question dict(s) in schema.json shape, not yet stored.
- Guarantees:
  - The provider is asked for JSON via `json_mode`; the reply is hardened by `json_parser`
    (fence stripping, balanced-object extraction) before validation.
  - `generate_batch` may return fewer than `count` items; `ValueError` only when none are valid.
  - `LLMRateLimitError` propagates so the caller can degrade instead of retrying.

## json_parser.py

```python
def strip_markdown_fences(text: str) -> str
def extract_first_json_object(text: str) -> str
def extract_all_json_objects(text: str) -> List[Dict[str, Any]]
def safe_parse_json(raw_text: str) -> Dict[str, Any]
def safe_parse_json_list(raw_text: str) -> List[Dict[str, Any]]
def validate_required_fields(payload: Dict[str, Any], required_fields: List[str]) -> None
def validate_payload(payload: Dict[str, Any]) -> None
def parse_and_insert(json_str: str) -> int
```

Contract:
- Extraction is balance-aware: prose around the JSON, trailing commentary and a nested
  `options` array are all handled; a `not-json` string yields `ValueError`/`[]`, never a guess.
- `parse_and_insert` returns the new `questions.id`; raises `KeyError` for missing required
  fields (as the spec requires) and `SchemaValidationError` for a schema violation.
- Side effects: one INSERT into `questions` through `db_manager`, options stored via
  `schemas.options_to_json`.

## controller.py

```python
def build_chat_prompt(user_input: str, context_chunks: List[str]) -> str
def build_quota_fallback_response(context_chunks: List[str]) -> str
def retrieve_context(query: str, top_k: int) -> tuple[List[str], List[str]]
def append_citations(answer: str, sources: List[str]) -> str
def chat(user_input: str, top_k: int = TOP_K, model_name: str = DEFAULT_MODEL) -> str
def generate_exercise_for_user(uid: int, topic: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]
def grade_answer(question: Dict[str, Any], selected_answer: str) -> bool
def record_answer(uid: int, question: Dict[str, Any], selected_answer: str) -> Dict[str, Any]
```

Contract:
- `chat`:
  - Input: non-empty `user_input`, `top_k` in [1, 50].
  - Output: assistant answer with a citation footer when sources are known.
  - Retrieval failures never abort the answer; an `LLMRateLimitError` returns
    `build_quota_fallback_response(...)` (retrieved context + a notice) instead of raising.
  - Any other provider failure propagates as `LLMError` for the UI to display.
- `generate_exercise_for_user`: level from `adaptive_logic.get_next_difficulty(uid)`,
  question from `generator.generate`; no DB write (drafts are persisted only on save).
- `grade_answer` compares option letters case-insensitively; unknown letters are False.
- `record_answer` is the single write path for an answer: grade -> `save_history` ->
  recompute the level. Output: `{"is_correct": bool, "next_level": int, "correct_answer": str}`.
- UI code must not re-implement any of these steps.

## retriever.py

```python
@dataclass(frozen=True)
class ChunkHit:
    text: str
    source_file: str
    chunk_id: int
    score: float

def search(query: str, top_k: int = TOP_K) -> List[ChunkHit]
def retrieve(query: str, top_k: int = TOP_K) -> List[str]
def retrieve_with_sources(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]
```

Contract:
- `search` is the one implementation: index + metadata resolution via `faiss_store`,
  inner-product over normalized vectors, best-first ordering.
- `retrieve` = texts only; `retrieve_with_sources` = `[{"text", "source", "chunk_id"}]` for
  citations (kept for existing callers).
- Missing index/metadata: `search` raises `FileNotFoundError` naming the file to build
  (`python embedder.py`); the two convenience wrappers return `[]` so the UI degrades quietly.

## faiss_store.py

```python
def default_vector_dir() -> Path
def resolve_index_path(index_path: str | Path | None = None, *, vector_dir: str | Path | None = None) -> Path
def resolve_metadata_path(index_path: Path | None = None) -> Path
def write_index(index: Any, index_path: Path) -> None
def read_index(index_path: Path) -> Any
def load_cached_index(index_path: Path) -> Any
def load_metadata(metadata_path: Path) -> Dict[str, Any]
def write_metadata(metadata_path: Path, payload: Dict[str, Any]) -> None
def build_metadata_payload(chunks: List[Dict[str, Any]], *, embedding_model: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> Dict[str, Any]
def get_embedding_model(model_name: str | None = None) -> Any
def load_index_for(metadata_path: Path | None = None, index_path: Path | None = None) -> Any
def embed_query(model: Any, query: str) -> Any
def search_positions(index: Any, query_vector: Any, top_k: int) -> List[int]
def search_with_scores(index: Any, query_vector: Any, top_k: int) -> List[tuple[int, float]]
```

Contract:
- Path precedence (both resolvers): an absolute configured path wins; a bare filename is
  anchored to the configured vector dir. Config values are read at **call** time so a test
  or a runtime override cannot be shadowed by an import-time constant.
- Metadata is written next to the index by `embedder.build_faiss_index`, so the two never drift.
- `write_index`/`read_index` fall back to an ASCII temp directory when the configured path
  breaks FAISS' C++ loader (Windows non-ASCII paths), then move the file into place.
- `get_embedding_model` caches per model name under a lock; `load_cached_index` keys on
  `(path, mtime)` so a rebuild is picked up without a restart.
- `embed_query` returns a normalized float32 row; scores are therefore cosine similarities.

## embedder.py

```python
@dataclass
class ChunkRecord:
    chunk_id: int
    source_file: str
    text: str

def normalize_whitespace(text: str) -> str
def parse_pdf_text(file_path: Path) -> str
def parse_docx_text(file_path: Path) -> str
def extract_text(file_path: Path) -> str
def chunk_text_by_tokens(text: str, model: Any, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]
def collect_source_files(data_dir: Path) -> List[Path]
def build_chunk_records(source_files: List[Path], model: Any, chunk_size: int, overlap: int) -> List[ChunkRecord]
def build_faiss_index(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    vector_dir: Path | str = DEFAULT_VECTOR_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Tuple[Path, Path, int]
```

Contract:
- `chunk_text_by_tokens`: `chunk_size` in [256, 512], `overlap < chunk_size`, both validated;
  chunks never overlap by more than `overlap` tokens and never drop text.
- Unreadable documents are skipped with a warning (a corrupt PDF must not fail the build).
- `build_faiss_index` writes `<vector_dir>/faiss_index.bin` and `chunks_metadata.json`, and
  returns `(index_path, metadata_path, total_chunks)`; the index is `IndexFlatIP` over
  normalized vectors. Model loading/index IO go through `faiss_store`.

## adaptive_logic.py

```python
def compute_next_level(current_level: int, outcomes: List[int], *, promote_after: int = 3, demote_after: int = 2) -> int
def get_next_difficulty(uid: int) -> int
```

Contract:
- `compute_next_level` is pure: no DB, no clock. `outcomes` is newest-first (`1`/`0`).
- Rule set (counted on the **newest** consecutive run, not the whole history):
  - promote_after (3) newest correct in a row => +1
  - demote_after (2) newest incorrect in a row => -1
  - otherwise unchanged
  - result clamped to [1, 5]
- `get_next_difficulty(uid)` reads the stored level (unknown user => 1), applies the rule and
  persists the new level only when it actually changed (no pointless UPDATE).

## dashboard.py

```python
def format_timestamp(raw: Any) -> str
def build_subject_pie(subject_stats: Dict[str, Dict[str, float]]) -> go.Figure
def build_weak_topic_radar(subject_stats: Dict[str, Dict[str, float]]) -> go.Figure
def build_progress_line(timeline: List[Dict[str, Any]]) -> go.Figure
def build_difficulty_bar(difficulty_rows: List[Dict[str, Any]]) -> go.Figure
def render_summary_metrics(stats: Dict[str, Any]) -> None
def render_dashboard(uid: Optional[int]) -> None
```

Contract:
- The four builders are pure `(rows) -> Figure`: testable without Streamlit, and each returns
  an explicit "no data yet" figure instead of a Plotly error when its input is empty.
- `build_progress_line` plots **cumulative** accuracy over attempts (per-attempt accuracy is noise).
- `render_dashboard(None)` renders the "pick a user" state and issues no query.

## init_db.py

```python
class SchemaBootstrapError(RuntimeError)
def read_schema_sql(schema_path: Path = SCHEMA_SQL_PATH) -> str
def verify_tables(conn: sqlite3.Connection, required: frozenset[str] = REQUIRED_TABLES) -> None
def initialize_database() -> Path
def main() -> int
```

Contract:
- `read_schema_sql` is the single place schema.sql is loaded (also used by the test fixtures);
  an empty/missing file raises `SchemaBootstrapError`.
- `initialize_database()` creates parents, applies the schema idempotently, verifies the
  required tables and returns the absolute DB path. `DB_PATH` is read from `config` at call time.
- With `DB_TYPE=postgresql` it raises `SchemaBootstrapError` telling the operator to run the
  equivalent migration, instead of silently writing a SQLite file nobody reads.
- `main()` returns a process exit code and prints `[ERROR] ...` without a traceback.

## auth.py

```python
class AuthenticationError(Exception)
class AuthorizationError(Exception)
class SecurityUnavailableError(RuntimeError)

def hash_password(password: str) -> str
def verify_password(password: str, hashed: str) -> bool
def needs_password_rehash(hashed: str) -> bool
def generate_token(user_id: int, role: str = ROLE_USER, expires_hours: Optional[int] = None) -> str
def verify_token(token: str) -> Dict[str, Any]
def require_auth(func: Callable[..., Any]) -> Callable[..., Any]
def require_role(required_role: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]

class UserManager:
    def create_user(self, name: str, email: Optional[str] = None, password: Optional[str] = None, level: int = 1) -> int
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]
    def get_user_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]
    def authenticate_user(self, identifier: str, password: str) -> Optional[str]
    def touch_last_login(self, user_id: int) -> None
    def get_user_roles(self, user_id: int) -> List[str]
    def assign_role(self, user_id: int, role_name: str) -> bool
    def deactivate_user(self, user_id: int) -> bool
```

Contract:
- All reads/writes go through `db_manager` (`users`, `roles`, `user_roles`); `auth.py` opens
  no connection and embeds no SQL string outside those tables' statements.
- Hashing is bcrypt. Unsalted SHA-256 hashes written by earlier builds are accepted only
  while `ALLOW_LEGACY_PASSWORD_FALLBACK=true`, and `authenticate_user` upgrades them to
  bcrypt on the next successful login.
- Missing `bcrypt`/`PyJWT` raise `SecurityUnavailableError` - never a silent SHA-256 fallback.
- `generate_token` payload: `{"user_id", "role", "iat", "exp", "type": "access"}`; expiry from
  `SESSION_TIMEOUT_HOURS` unless overridden. `verify_token` raises `AuthenticationError` for
  expired/invalid tokens.
- `require_auth` accepts the token as `token=` kwarg or first positional arg and injects
  `user_id` / `user_role` into the call; `require_role` lets admins pass any role check.

## rag_tester.py

```python
@dataclass
class RagTestCase:
    query: str
    target_chunk_id: int
    relevant_chunk_ids: Set[int]
    source_file: str

def precision_at_k(retrieved: Sequence[int], relevant: Set[int], top_k: int) -> float
def reciprocal_rank(retrieved: Sequence[int], relevant: Set[int]) -> float
def mean(values: Sequence[float]) -> float
def build_query_from_chunk_text(text: str) -> str
def select_test_chunk_indices(total_chunks: int, test_count: int) -> List[int]
def build_test_cases(metadata: Dict, test_count: int = DEFAULT_TEST_COUNT) -> List[RagTestCase]
def evaluate_retrieval(test_cases: Sequence[RagTestCase], metadata: Dict, *, top_k: int = DEFAULT_TOP_K, index_path: Path | None = None) -> Tuple[float, float, List[Dict]]
def print_report(mean_p_at_k: float, mrr: float, details: Sequence[Dict], top_k: int) -> None
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace
def main(argv: Sequence[str] | None = None) -> None
```

Contract:
- The three metric functions are pure (`top_k == 0` or empty input => 0.0, never ZeroDivisionError).
- Test selection is deterministic and evenly spread, so two runs are comparable.
- `--test-count` is clamped to `DEFAULT_TEST_COUNT` (20) - pinned by tests on purpose.

## seed_db.py / generate_mock_data.py

```python
def seed_questions(topics: List[Dict[str, object]] | None = None) -> int
def main() -> int
```

Contract:
- `seed_questions` returns the number of stored questions and writes through
  `sqlite_manager.insert_questions` (no SQL of its own).
- `generate_mock_data.py` targets `mock_data/`; its `chat_history` inserts use the real
  `schema.sql` columns `(uid, role, content, created_at)` and it does not redeclare that table.

---

Change policy:
- Any signature change in these public functions must update this file in the same pull request.
- New cross-module helpers belong here too: if a function is imported by another module,
  it is part of the contract.

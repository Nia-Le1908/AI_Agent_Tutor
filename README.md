# AI Tutor V5.1

AI Tutor V5.1 is a personalized learning assistant built with Python, Streamlit, SQLite, DeepSeek, and a local RAG pipeline (Sentence Transformers + FAISS).

## Features

- Chat tutor grounded by local document retrieval (RAG), with source citations.
- Auto-generated multiple-choice questions with strict JSON validation.
- Adaptive difficulty logic based on user answer streaks.
- Streamlit UI with persistent session state.
- Learning analytics dashboard with Plotly charts.
- Optional PostgreSQL backend and JWT/bcrypt authentication layer.

## Project Structure

Data access and configuration are centralized: `config.py` reads the environment once, `db_manager.py` owns connections, and `sqlite_manager.py` owns the SQL. The UI never writes SQL.

- .env.example: Environment variable template.
- config.py: Single source of runtime config (paths, provider, RAG tuning, validation).
- validation.py: Shared argument checks (uid/level ranges) used by every layer.
- schemas.py: schema.json loading, question validation, A/B/C/D option normalization.
- db_manager.py: Database connection layer (SQLite + pooled PostgreSQL), backend-agnostic rows.
- config_db.py: Compatibility facade over `config` for database backend selection.
- schema.sql: SQLite schema (DDL is SQLite-specific; see "Database backends").
- init_db.py: Database initializer.
- sqlite_manager.py: Repository layer - all business SQL (users, questions, history, analytics).
- json_parser.py: LLM output hardening (markdown fences, JSON extraction) and question insertion.
- llm_client.py: Single LLM entry point (OpenAI-compatible) with retries and backoff.
- generator.py: Question generation with strict JSON safety.
- adaptive_logic.py: Dynamic difficulty adjustment (pure rules + persistence).
- faiss_store.py: Shared FAISS index/metadata IO and embedding-model cache.
- embedder.py: PDF/DOCX parsing, chunking, embedding, FAISS build.
- retriever.py: Top-k retrieval from the FAISS index.
- controller.py: Orchestration for chat, generation, and answer recording.
- app.py: Streamlit app (rendering only).
- dashboard.py: Plotly dashboard rendering.
- auth.py: Password hashing, JWT, RBAC decorators, user management.
- rag_tester.py: Precision@K and MRR evaluation for retrieval.
- generate_mock_data.py: Mock JSON and mock SQLite generator for demos.
- tests/: Pytest suite (hermetic - no network, no model downloads).
- file_md/interfaces.md: Internal cross-module function contracts.

## Requirements

- Python 3.10+
- pip
- Internet access for package installation and LLM API calls

## Quick Start (Linux/macOS)

1. Clone or copy project.
2. Enter project folder.
3. Run:

```bash
bash run.sh
```

The script will:
- create .venv if missing
- install dependencies from requirements.txt
- create .env from .env.example if missing
- initialize SQLite database
- start Streamlit app

## Manual Setup (All Platforms)

### 1) Create virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Configure environment variables

Create .env from template:

```bash
cp .env.example .env
```

Windows PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Then edit .env and set at least:

- DEEPSEEK_API_KEY: Required for question generation and chat.
- DB_PATH: SQLite location (default data/ai_tutor_v5.db).
- FAISS_INDEX_PATH: FAISS index file path.
- LOG_PATH: Log output file path.
- CHUNK_SIZE: Must be in [256, 512].
- CHUNK_OVERLAP: Must be less than CHUNK_SIZE (default 50).
- TOP_K: Retrieval top-k (default 3).
- EMBEDDING_MODEL_NAME: Default all-MiniLM-L6-v2.

Example:

```env
DEEPSEEK_API_KEY=your_real_key_here
DB_PATH=data/ai_tutor_v5.db
FAISS_INDEX_PATH=vector_store/faiss_index.bin
LOG_PATH=logs/app.log
CHUNK_SIZE=256
CHUNK_OVERLAP=50
TOP_K=3
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

Every value above is validated at import: an out-of-range `CHUNK_SIZE` or an
`CHUNK_OVERLAP` that is not smaller than `CHUNK_SIZE` raises `ConfigError` with the
offending name instead of failing later inside the pipeline.

### 4) Initialize database

```bash
python init_db.py
```

### 5) Build vector index (required for RAG chat)

1. Put PDF/DOCX documents into data folder.
2. Run:

```bash
python embedder.py
```

This creates:
- vector_store/faiss_index.bin
- vector_store/chunks_metadata.json

### 6) Run app

```bash
streamlit run app.py
```

## Running Individual Components

### Generate and store starter questions

```bash
python seed_db.py
```

### Generate mock data

```bash
python generate_mock_data.py
```

Outputs:
- mock_data/mock_questions.json
- mock_data/mock_db.sqlite

### Evaluate RAG retrieval quality

```bash
python rag_tester.py
```

Outputs:
- Precision@3
- MRR
- Per-test retrieval details for 20 deterministic test queries

## Tests

The suite is hermetic: it fakes the LLM transport and the embedding model, and runs
every database test against a throwaway SQLite file created from `schema.sql`.

```bash
pip install pytest
python -m pytest
```

Coverage includes the adaptive streak rules, schema/option validation, LLM output
hardening, retry and rate-limit fallback behaviour, FAISS IO, and the Streamlit
flows driven through `streamlit.testing.v1.AppTest`.

## Database Backends

- SQLite (default): zero setup; `python init_db.py` creates and upgrades the schema.
- PostgreSQL: set `DB_TYPE=postgresql` plus the `POSTGRES_*` variables and install
  `psycopg2-binary`. All queries go through `db_manager.py`, which translates
  placeholders and reads inserted ids per backend.

`schema.sql` is SQLite DDL only (PRAGMA, AUTOINCREMENT, triggers, INSERT OR IGNORE).
A PostgreSQL deployment needs equivalent migrations applied out of band; running
`init_db.py` with `DB_TYPE=postgresql` fails fast with that message rather than
silently creating a SQLite file nobody reads.

## Internal API Contract

See file_md/interfaces.md for exact function signatures and expected return types
used across modules. It is kept in the same commit as any signature change.

## Notes and Troubleshooting

- If Streamlit launches but chat/generation fails, verify DEEPSEEK_API_KEY in .env.
- If retrieval fails, make sure the FAISS index and chunk metadata exist by running `python embedder.py`.
- Chat still answers (retrieval-only, with a clear notice) when the provider is rate limited or out of quota.
- If your machine cannot install faiss-cpu directly, use a supported Python version and platform wheel.
- On Windows, PowerShell execution policy may block activation scripts; run with a permitted policy or use cmd activation.
- FAISS paths containing non-ASCII characters are handled by an ASCII temp-file fallback in `faiss_store.py`.

## Security

- Do not commit .env to source control.
- Treat DEEPSEEK_API_KEY and JWT_SECRET_KEY as sensitive.
- Passwords are hashed with bcrypt; the legacy unsalted SHA-256 hashes written by
  earlier builds are still accepted for verification (and upgraded on next login) so
  existing accounts are not locked out. Set `ALLOW_LEGACY_PASSWORD_FALLBACK=false`
  to disable that compatibility path once migrated.

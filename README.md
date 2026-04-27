# AI Tutor V5.1

AI Tutor V5.1 is a personalized learning assistant built with Python, Streamlit, SQLite, Gemini API, and a local RAG pipeline (Sentence Transformers + FAISS).

## Features

- Chat tutor grounded by local document retrieval (RAG).
- Auto-generated multiple-choice questions with strict JSON validation.
- Adaptive difficulty logic based on user answer streaks.
- Streamlit UI with persistent session state.
- Learning analytics dashboard with Plotly charts.

## Project Structure

- .env.example: Environment variable template.
- config.py: Centralized runtime config and constants.
- schema.sql: SQLite schema.
- init_db.py: Database initializer.
- sqlite_manager.py: DB access functions.
- json_parser.py: Strict schema validation and question insertion.
- embedder.py: PDF/DOCX parsing, chunking, embedding, FAISS build.
- retriever.py: Top-k retrieval from FAISS index.
- generator.py: Gemini question generation with strict JSON safety.
- adaptive_logic.py: Dynamic difficulty adjustment.
- controller.py: Main orchestration for chat + generation.
- app.py: Streamlit app.
- dashboard.py: Plotly dashboard rendering.
- rag_tester.py: Precision@3 and MRR evaluation for retrieval.
- generate_mock_data.py: TV6 mock JSON and mock SQLite generator.
- interfaces.md: Internal cross-module function contracts.

## Requirements

- Python 3.10+
- pip
- Internet access for package installation and Gemini API calls

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

- GEMINI_API_KEY: Required for Gemini generation/chat modules.
- DB_PATH: SQLite location (default data/ai_tutor_v5.db).
- FAISS_INDEX_PATH: FAISS index file path.
- LOG_PATH: Log output file path.
- CHUNK_SIZE: Must be in [256, 512].
- CHUNK_OVERLAP: Must be less than CHUNK_SIZE (default 50).
- TOP_K: Retrieval top-k (default 3).
- EMBEDDING_MODEL_NAME: Default all-MiniLM-L6-v2.

Example:

```env
GEMINI_API_KEY=your_real_key_here
DB_PATH=data/ai_tutor_v5.db
FAISS_INDEX_PATH=vector_store/faiss_index.bin
LOG_PATH=logs/app.log
CHUNK_SIZE=256
CHUNK_OVERLAP=50
TOP_K=3
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

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

## Internal API Contract

See interfaces.md for exact function signatures and expected return types used across modules.

## Notes and Troubleshooting

- If Streamlit launches but chat/generation fails, verify GEMINI_API_KEY in .env.
- If retrieval fails, make sure FAISS index and chunk metadata exist by running embedder.py.
- If your machine cannot install faiss-cpu directly, use a supported Python version and platform wheel.
- On Windows, PowerShell execution policy may block activation scripts; run with a permitted policy or use cmd activation.

## Security

- Do not commit .env to source control.
- Treat GEMINI_API_KEY as sensitive.

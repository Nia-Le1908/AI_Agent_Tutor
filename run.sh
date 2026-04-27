#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# AI Tutor V5.1 bootstrap script
# - Checks Python and pip
# - Creates/uses virtual environment
# - Installs dependencies
# - Ensures .env exists
# - Initializes SQLite database
# - Starts Streamlit app
# ================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[INFO] Project directory: $PROJECT_DIR"

# -------------------------
# Resolve Python executable
# -------------------------
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[ERROR] Python is not installed. Install Python 3.10+ and retry."
  exit 1
fi

echo "[INFO] Using Python: $PYTHON_BIN"

# Quick version check (3.10+ required)
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("[ERROR] Python 3.10+ is required.")
print(f"[INFO] Python version OK: {sys.version.split()[0]}")
PY

# -------------------------
# Create venv if missing
# -------------------------
if [[ ! -d ".venv" ]]; then
  echo "[INFO] Creating virtual environment in .venv"
  "$PYTHON_BIN" -m venv .venv
else
  echo "[INFO] Reusing existing virtual environment: .venv"
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Ensure pip is available in venv
if ! command -v pip >/dev/null 2>&1; then
  echo "[ERROR] pip not found in virtual environment."
  exit 1
fi

echo "[INFO] Upgrading pip"
pip install --upgrade pip

echo "[INFO] Installing dependencies from requirements.txt"
pip install -r requirements.txt

# -------------------------
# Ensure .env exists
# -------------------------
if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp .env.example .env
    echo "[WARN] .env was missing, created from .env.example"
    echo "[WARN] Please edit .env and set GEMINI_API_KEY for full functionality"
  else
    echo "[ERROR] .env.example not found. Cannot auto-create .env"
    exit 1
  fi
else
  echo "[INFO] Found existing .env"
fi

# -------------------------
# Initialize database
# -------------------------
echo "[INFO] Initializing database"
python init_db.py

# -------------------------
# Start Streamlit
# -------------------------
if ! command -v streamlit >/dev/null 2>&1; then
  echo "[ERROR] streamlit command not found after installation."
  exit 1
fi

echo "[INFO] Starting Streamlit app"
echo "[INFO] Open browser at: http://localhost:8501"
streamlit run app.py

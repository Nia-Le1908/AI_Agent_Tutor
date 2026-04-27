"""
TV6 - Mock data generator for AI Tutor V5.1.

This script generates:
1) mock_data/mock_questions.json
   - Exactly 10 sample questions.
   - Each question is strictly validated against schema.json.

2) mock_data/mock_db.sqlite
   - Base tables from schema.sql.
   - Sample users, questions, sessions, answer history.
   - Additional mock-only tables:
       - chat_history (sample conversation traces)
       - weak_topics (materialized weak-topic stats for quick inspection)

Why an additional weak_topics table in mock DB?
- The production app computes weak topics dynamically via sqlite_manager.get_weak_topics.
- For TV6 infrastructure/demo work, we also materialize a snapshot so stakeholders
  can inspect weak-topic data immediately with a simple SELECT.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parent
MOCK_DIR = PROJECT_ROOT / "mock_data"
SCHEMA_JSON_PATH = PROJECT_ROOT / "schema.json"
SCHEMA_SQL_PATH = PROJECT_ROOT / "schema.sql"
OUTPUT_QUESTIONS_PATH = MOCK_DIR / "mock_questions.json"
OUTPUT_DB_PATH = MOCK_DIR / "mock_db.sqlite"


def load_schema() -> Dict[str, Any]:
    """Load schema.json and ensure the schema itself is valid."""
    if not SCHEMA_JSON_PATH.exists():
        raise FileNotFoundError(f"schema.json not found at: {SCHEMA_JSON_PATH}")

    schema = json.loads(SCHEMA_JSON_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def build_mock_questions() -> List[Dict[str, Any]]:
    """
    Build exactly 10 questions matching schema.json fields and constraints.

    Output format intentionally keeps options as a 4-item list because that is
    what schema.json requires.
    """
    return [
        {
            "question_id": 1001,
            "content": "What is 5 + 7?",
            "difficulty": 1,
            "subject": "Mathematics",
            "options": ["10", "11", "12", "13"],
            "answer": "C",
            "explanation": "5 + 7 equals 12.",
        },
        {
            "question_id": 1002,
            "content": "Which gas is most abundant in Earth's atmosphere?",
            "difficulty": 2,
            "subject": "Science",
            "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
            "answer": "B",
            "explanation": "Nitrogen makes up about 78% of Earth's atmosphere.",
        },
        {
            "question_id": 1003,
            "content": "Which continent is Egypt located in?",
            "difficulty": 1,
            "subject": "Geography",
            "options": ["Asia", "Europe", "Africa", "South America"],
            "answer": "C",
            "explanation": "Egypt is located in northeastern Africa.",
        },
        {
            "question_id": 1004,
            "content": "What does HTTP stand for?",
            "difficulty": 3,
            "subject": "Computer Science",
            "options": [
                "HyperText Transfer Protocol",
                "HighText Transit Package",
                "Hyperlink Text Transport Program",
                "Host Transfer Text Process",
            ],
            "answer": "A",
            "explanation": "HTTP is short for HyperText Transfer Protocol.",
        },
        {
            "question_id": 1005,
            "content": "In a right triangle, what is the side opposite the right angle called?",
            "difficulty": 2,
            "subject": "Mathematics",
            "options": ["Adjacent", "Hypotenuse", "Median", "Altitude"],
            "answer": "B",
            "explanation": "The longest side opposite the right angle is the hypotenuse.",
        },
        {
            "question_id": 1006,
            "content": "Who wrote 'Romeo and Juliet'?",
            "difficulty": 1,
            "subject": "Literature",
            "options": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Leo Tolstoy"],
            "answer": "B",
            "explanation": "Romeo and Juliet is a tragedy written by William Shakespeare.",
        },
        {
            "question_id": 1007,
            "content": "Which process do plants use to make food?",
            "difficulty": 2,
            "subject": "Biology",
            "options": ["Respiration", "Photosynthesis", "Digestion", "Fermentation"],
            "answer": "B",
            "explanation": "Plants produce glucose by photosynthesis using light energy.",
        },
        {
            "question_id": 1008,
            "content": "What is the derivative of x^2 with respect to x?",
            "difficulty": 4,
            "subject": "Mathematics",
            "options": ["x", "2x", "x^2", "2"],
            "answer": "B",
            "explanation": "Using the power rule, d/dx(x^2) = 2x.",
        },
        {
            "question_id": 1009,
            "content": "Which data structure follows First-In-First-Out order?",
            "difficulty": 3,
            "subject": "Computer Science",
            "options": ["Stack", "Queue", "Tree", "Graph"],
            "answer": "B",
            "explanation": "A queue processes elements in FIFO order.",
        },
        {
            "question_id": 1010,
            "content": "What is the primary function of red blood cells?",
            "difficulty": 2,
            "subject": "Biology",
            "options": [
                "Fight infections",
                "Carry oxygen",
                "Produce hormones",
                "Digest nutrients",
            ],
            "answer": "B",
            "explanation": "Red blood cells carry oxygen from the lungs to body tissues.",
        },
    ]


def validate_questions_strictly(questions: List[Dict[str, Any]], schema: Dict[str, Any]) -> None:
    """
    Strictly validate every question against schema.json.

    This fails fast with precise per-question diagnostics if any field violates
    type, range, required keys, or additionalProperties constraints.
    """
    validator = Draft202012Validator(schema)

    if len(questions) != 10:
        raise ValueError(f"Expected exactly 10 questions, got {len(questions)}")

    for index, question in enumerate(questions, start=1):
        errors = sorted(validator.iter_errors(question), key=lambda err: list(err.path))
        if errors:
            joined = "; ".join(err.message for err in errors)
            raise ValueError(f"Question #{index} failed schema validation: {joined}")


def write_mock_questions_json(questions: List[Dict[str, Any]]) -> None:
    """Write questions list to mock_data/mock_questions.json in UTF-8."""
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_QUESTIONS_PATH.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_schema_sql() -> str:
    """Read schema.sql used to create core SQLite tables."""
    if not SCHEMA_SQL_PATH.exists():
        raise FileNotFoundError(f"schema.sql not found at: {SCHEMA_SQL_PATH}")

    sql_text = SCHEMA_SQL_PATH.read_text(encoding="utf-8").strip()
    if not sql_text:
        raise ValueError("schema.sql is empty")
    return sql_text


def _options_list_to_json_object(options: List[str]) -> str:
    """Convert [A, B, C, D] list into {A:..., B:..., C:..., D:...} JSON string."""
    payload = {
        "A": options[0],
        "B": options[1],
        "C": options[2],
        "D": options[3],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_mock_database(questions: List[Dict[str, Any]]) -> None:
    """
    Build mock_data/mock_db.sqlite with schema + sample records.

    Records included:
    - users
    - questions (10 items from mock_questions.json)
    - sessions
    - history (crafted to create weak topics)
    - chat_history (mock-only)
    - weak_topics (mock-only materialized stats)
    """
    MOCK_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DB_PATH.exists():
        OUTPUT_DB_PATH.unlink()

    conn = sqlite3.connect(str(OUTPUT_DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        # 1) Create core schema from project contract.
        conn.executescript(_load_schema_sql())

        # 2) Create mock-only chat and weak-topic tables.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (uid) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS weak_topics (
                uid INTEGER NOT NULL,
                subject TEXT NOT NULL,
                correct_count INTEGER NOT NULL,
                incorrect_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                FOREIGN KEY (uid) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )

        # 3) Insert sample users.
        users = [
            (1, "Alice", 2),
            (2, "Bob", 1),
            (3, "Charlie", 4),
        ]
        conn.executemany(
            "INSERT INTO users (id, name, level) VALUES (?, ?, ?)",
            users,
        )

        # 4) Insert validated questions.
        question_rows = []
        for q in questions:
            question_rows.append(
                (
                    int(q["question_id"]),
                    q["content"],
                    int(q["difficulty"]),
                    q["subject"],
                    _options_list_to_json_object(q["options"]),
                    q["answer"],
                    q["explanation"],
                )
            )

        conn.executemany(
            """
            INSERT INTO questions (id, content, difficulty, subject, options, answer, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            question_rows,
        )

        # 5) Insert sessions (high-level attempt summaries).
        sessions = [
            (1, "2026-04-20 09:00:00", 0.72),
            (1, "2026-04-24 14:30:00", 0.68),
            (2, "2026-04-22 11:10:00", 0.55),
            (3, "2026-04-23 16:45:00", 0.83),
        ]
        conn.executemany(
            "INSERT INTO sessions (uid, start_time, score) VALUES (?, ?, ?)",
            sessions,
        )

        # 6) Insert answer history. These rows are crafted so user 1 has weak
        # topics in Mathematics and Computer Science.
        history_rows = [
            (1, 1001, 1, "2026-04-20 09:05:00"),
            (1, 1005, 0, "2026-04-20 09:10:00"),
            (1, 1008, 0, "2026-04-20 09:12:00"),
            (1, 1004, 0, "2026-04-24 14:35:00"),
            (1, 1009, 0, "2026-04-24 14:39:00"),
            (1, 1002, 1, "2026-04-24 14:43:00"),
            (1, 1007, 1, "2026-04-24 14:47:00"),
            (2, 1001, 0, "2026-04-22 11:15:00"),
            (2, 1003, 1, "2026-04-22 11:18:00"),
            (2, 1006, 0, "2026-04-22 11:22:00"),
            (2, 1010, 1, "2026-04-22 11:26:00"),
            (3, 1004, 1, "2026-04-23 16:50:00"),
            (3, 1009, 1, "2026-04-23 16:54:00"),
            (3, 1008, 1, "2026-04-23 16:58:00"),
            (3, 1002, 0, "2026-04-23 17:02:00"),
        ]
        conn.executemany(
            "INSERT INTO history (uid, qid, is_correct, timestamp) VALUES (?, ?, ?, ?)",
            history_rows,
        )

        # 7) Insert sample chat history for UI/demo.
        chat_rows = [
            (1, "user", "Can you explain the Pythagorean theorem?", "2026-04-24 14:31:00"),
            (1, "assistant", "Sure. In a right triangle, a^2 + b^2 = c^2.", "2026-04-24 14:31:10"),
            (1, "user", "Give me one easy example.", "2026-04-24 14:31:25"),
            (1, "assistant", "If sides are 3 and 4, hypotenuse is 5.", "2026-04-24 14:31:37"),
            (2, "user", "What is photosynthesis?", "2026-04-22 11:12:00"),
            (2, "assistant", "Plants use light to make glucose and oxygen.", "2026-04-22 11:12:14"),
            (3, "user", "Explain queue vs stack quickly.", "2026-04-23 16:47:00"),
            (3, "assistant", "Queue is FIFO, stack is LIFO.", "2026-04-23 16:47:12"),
        ]
        conn.executemany(
            "INSERT INTO chat_history (uid, role, content, timestamp) VALUES (?, ?, ?, ?)",
            chat_rows,
        )

        # 8) Materialize weak-topic snapshot from history + questions.
        conn.execute("DELETE FROM weak_topics")
        conn.execute(
            """
            INSERT INTO weak_topics (uid, subject, correct_count, incorrect_count, total_count, accuracy)
            SELECT
                h.uid,
                q.subject,
                SUM(CASE WHEN h.is_correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN h.is_correct = 0 THEN 1 ELSE 0 END) AS incorrect_count,
                COUNT(*) AS total_count,
                CASE
                    WHEN COUNT(*) = 0 THEN 0.0
                    ELSE CAST(SUM(CASE WHEN h.is_correct = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
                END AS accuracy
            FROM history h
            INNER JOIN questions q ON q.id = h.qid
            GROUP BY h.uid, q.subject
            ORDER BY h.uid ASC, accuracy ASC, q.subject ASC
            """
        )

        conn.commit()
    finally:
        conn.close()


def main() -> None:
    """Generate both mock questions and mock database in one run."""
    schema = load_schema()
    questions = build_mock_questions()

    validate_questions_strictly(questions, schema)
    write_mock_questions_json(questions)
    build_mock_database(questions)

    print(f"[OK] Wrote questions file: {OUTPUT_QUESTIONS_PATH}")
    print(f"[OK] Wrote mock database: {OUTPUT_DB_PATH}")


if __name__ == "__main__":
    main()

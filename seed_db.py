"""
Seed the question bank with LLM-generated questions.

One question per topic is generated at difficulty 1 and stored through the
repository layer, so the seeding path writes exactly what the UI reads.

Usage:
    python seed_db.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from config import DB_PATH
from generator import generate
from init_db import initialize_database
from logging_setup import configure_logging
from sqlite_manager import insert_question
from validation import require_level

logger = logging.getLogger(__name__)

# Topics to seed, each with the difficulty level to generate at.
DEFAULT_TOPICS: List[Dict[str, object]] = [
    {"subject": "Mã độc File Infector cơ bản", "difficulty": 1},
    {"subject": "Thuật toán đồ thị BFS", "difficulty": 1},
    {"subject": "Ngữ pháp TOEIC: Thì hiện tại đơn", "difficulty": 1},
    {"subject": "Bảo mật hệ thống thông tin", "difficulty": 1},
]


def seed_questions(topics: List[Dict[str, object]] | None = None) -> int:
    """
    Generate and store one question per topic.

    Returns:
        Number of questions saved. Individual failures are logged and skipped so
        one bad generation cannot abort the whole seed run.
    """
    topics = topics or DEFAULT_TOPICS
    saved = 0

    for entry in topics:
        subject = str(entry["subject"])
        difficulty = require_level(entry["difficulty"], "difficulty")
        logger.info("Generating level %s question for topic: %s", difficulty, subject)

        try:
            question = generate(topic=subject, difficulty=difficulty)
            insert_question(question)
            saved += 1
        except Exception as exc:  # noqa: BLE001 - one failure must not abort seeding
            logger.error("Failed to generate/store question for '%s': %s", subject, exc)

    return saved


def main() -> int:
    """CLI entry point."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate and store starter questions via the LLM.")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create/upgrade the database schema before seeding.",
    )
    args = parser.parse_args()

    if args.init:
        initialize_database()
    elif not Path(DB_PATH).exists():
        print(f"Không tìm thấy file Database tại {DB_PATH}. Hãy chạy python init_db.py trước.")
        return 1

    print("🚀 Đang sinh câu hỏi bằng LLM. Thời gian phụ thuộc vào tốc độ mạng/máy tính...")
    saved = seed_questions()
    print(f"\n🎉 Hoàn tất: đã lưu {saved}/{len(DEFAULT_TOPICS)} câu hỏi.")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

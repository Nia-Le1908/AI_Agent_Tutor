-- ================================================================
-- AI Tutor V5.1 - SQLite schema (Phase 1)
-- ================================================================
-- Tables requested by the specification:
--   1) Users
--   2) Questions
--   3) History
--   4) Sessions

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    level       INTEGER NOT NULL DEFAULT 1 CHECK (level BETWEEN 1 AND 5)
);

CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    difficulty  INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    subject     TEXT NOT NULL,
    -- JSON string storing answer options A/B/C/D.
    options     TEXT NOT NULL,
    answer      TEXT NOT NULL,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS history (
    uid         INTEGER NOT NULL,
    qid         INTEGER NOT NULL,
    is_correct  INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (uid) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (qid) REFERENCES questions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    uid         INTEGER NOT NULL,
    start_time  TEXT NOT NULL DEFAULT (datetime('now')),
    score       REAL,
    FOREIGN KEY (uid) REFERENCES users (id) ON DELETE CASCADE
);

-- Helpful indexes for query performance.
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions (difficulty);
CREATE INDEX IF NOT EXISTS idx_questions_subject ON questions (subject);
CREATE INDEX IF NOT EXISTS idx_history_uid ON history (uid);
CREATE INDEX IF NOT EXISTS idx_history_qid ON history (qid);
CREATE INDEX IF NOT EXISTS idx_history_uid_qid ON history (uid, qid);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history (timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_uid ON sessions (uid);

COMMIT;

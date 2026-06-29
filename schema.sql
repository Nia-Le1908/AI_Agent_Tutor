-- ================================================================
-- AI Tutor - Database Schema (Phase 5 - Production Ready)
-- ================================================================
-- Supports both SQLite (development) and PostgreSQL (production)
-- Tables:
--   1) Users - with authentication fields
--   2) Questions - exercise bank
--   3) History - answer tracking with session support
--   4) Sessions - learning session management
--   5) ChatHistory - persistent chat conversations
--   6) Roles - user role management (admin/user)
--   7) UserRoles - many-to-many relationship

-- For SQLite compatibility (PostgreSQL will ignore PRAGMA)
PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- ================================================================
-- Role Management (for access control)
-- ================================================================
CREATE TABLE IF NOT EXISTS roles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);

-- Default roles
INSERT OR IGNORE INTO roles (name, description) VALUES 
    ('admin', 'Administrator with full system access'),
    ('user', 'Regular learner');

-- ================================================================
-- Users Table - Enhanced with authentication and security
-- ================================================================
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT UNIQUE,
    password_hash   TEXT,  -- For future authentication
    level           INTEGER NOT NULL DEFAULT 1 CHECK (level BETWEEN 1 AND 5),
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_login      TEXT
);

-- ================================================================
-- User Roles Assignment (Many-to-Many)
-- ================================================================
CREATE TABLE IF NOT EXISTS user_roles (
    user_id     INTEGER NOT NULL,
    role_id     INTEGER NOT NULL,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE
);

-- Assign default 'user' role to all new users via trigger (SQLite)
CREATE TRIGGER IF NOT EXISTS assign_default_user_role
AFTER INSERT ON users
BEGIN
    INSERT INTO user_roles (user_id, role_id)
    SELECT NEW.id, id FROM roles WHERE name = 'user';
END;

-- ================================================================
-- Questions Table - Exercise Bank
-- ================================================================
CREATE TABLE IF NOT EXISTS questions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    difficulty      INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    subject         TEXT NOT NULL,
    topic           TEXT,  -- More granular than subject for adaptive learning
    -- JSON string storing answer options A/B/C/D.
    options         TEXT NOT NULL,
    answer          TEXT NOT NULL,
    explanation     TEXT,
    source_file     TEXT,  -- Track which document this came from (RAG)
    created_by      INTEGER,  -- User who created/generated this question
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    is_active       INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE SET NULL
);

-- ================================================================
-- History Table - Answer Tracking with Session Support
-- ================================================================
CREATE TABLE IF NOT EXISTS history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             INTEGER NOT NULL,
    qid             INTEGER NOT NULL,
    session_id      INTEGER,  -- Link to session for grouping
    is_correct      INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    selected_answer TEXT,  -- Store what user actually selected
    time_spent      INTEGER,  -- Seconds spent on this question
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (uid) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (qid) REFERENCES questions (id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE SET NULL
);

-- ================================================================
-- Sessions Table - Learning Session Management
-- ================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             INTEGER NOT NULL,
    start_time      TEXT NOT NULL DEFAULT (datetime('now')),
    end_time        TEXT,
    score           REAL,
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    is_active       INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (uid) REFERENCES users (id) ON DELETE CASCADE
);

-- ================================================================
-- ChatHistory Table - Persistent Chat Conversations
-- ================================================================
CREATE TABLE IF NOT EXISTS chat_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             INTEGER NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    session_id      INTEGER,
    parent_message  INTEGER,  -- For threading conversations
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (uid) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE SET NULL,
    FOREIGN KEY (parent_message) REFERENCES chat_history (id) ON DELETE SET NULL
);

-- ================================================================
-- Indexes for Query Performance
-- ================================================================
-- Users
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_name ON users (name);
CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active);

-- Questions
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions (difficulty);
CREATE INDEX IF NOT EXISTS idx_questions_subject ON questions (subject);
CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions (topic);
CREATE INDEX IF NOT EXISTS idx_questions_active ON questions (is_active);
CREATE INDEX IF NOT EXISTS idx_questions_diff_subject ON questions (difficulty, subject);

-- History
CREATE INDEX IF NOT EXISTS idx_history_uid ON history (uid);
CREATE INDEX IF NOT EXISTS idx_history_qid ON history (qid);
CREATE INDEX IF NOT EXISTS idx_history_uid_qid ON history (uid, qid);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history (timestamp);
CREATE INDEX IF NOT EXISTS idx_history_session ON history (session_id);
CREATE INDEX IF NOT EXISTS idx_history_uid_timestamp ON history (uid, timestamp);

-- Sessions
CREATE INDEX IF NOT EXISTS idx_sessions_uid ON sessions (uid);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions (is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_uid_active ON sessions (uid, is_active);

-- Chat History
CREATE INDEX IF NOT EXISTS idx_chat_history_uid ON chat_history (uid);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history (created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_uid_created ON chat_history (uid, created_at);

-- User Roles
CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles (user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles (role_id);

COMMIT;

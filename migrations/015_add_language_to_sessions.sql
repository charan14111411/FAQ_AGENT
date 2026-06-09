-- Migration 015: Add language preference column to sessions table
-- Tracks which language the user selected during the session.
-- NULL means English (default). Possible values: english, tamil, hindi, kannada, telugu, urdu

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS language VARCHAR(20) DEFAULT NULL;

COMMENT ON COLUMN sessions.language IS
    'User-selected response language for this session. NULL = English (default).';

-- Migration 015: Make phone and email nullable to support temp user creation at session start
ALTER TABLE users ALTER COLUMN phone DROP NOT NULL;
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

-- Migration 006: Make phone nullable so users who skip phone can still register
ALTER TABLE users ALTER COLUMN phone DROP NOT NULL;

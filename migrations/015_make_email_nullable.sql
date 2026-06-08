-- Migration 015: Make email nullable in users table
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

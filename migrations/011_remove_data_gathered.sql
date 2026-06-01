-- Drop the unused legacy data_gathered column from the users table
ALTER TABLE users DROP COLUMN IF EXISTS data_gathered;

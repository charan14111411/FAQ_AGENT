-- Revert users table to strict NOT NULL constraints
ALTER TABLE users 
    ALTER COLUMN name SET NOT NULL,
    ALTER COLUMN phone SET NOT NULL,
    ALTER COLUMN email SET NOT NULL;

-- Revert sessions table to strict NOT NULL constraints
ALTER TABLE sessions
    ALTER COLUMN category SET NOT NULL;

-- 1. Drop existing unique email constraint
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_email_key;
DROP INDEX IF EXISTS idx_users_email;
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 2. Add unique constraint on phone column
ALTER TABLE users ADD CONSTRAINT users_phone_key UNIQUE (phone);
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);

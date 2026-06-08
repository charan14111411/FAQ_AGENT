-- Relax constraints to allow eager creation of rows during step 1 of onboarding

ALTER TABLE users ALTER COLUMN name DROP NOT NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS data_gathered INTEGER DEFAULT 0;

ALTER TABLE sessions ALTER COLUMN category DROP NOT NULL;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_category_check;
ALTER TABLE sessions ADD CONSTRAINT sessions_category_check CHECK (category IS NULL OR category IN ('grower','corporate','investor','agritech','exploring'));

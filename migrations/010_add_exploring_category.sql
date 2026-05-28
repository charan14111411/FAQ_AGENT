-- Add 'exploring' to the sessions category check constraint
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_category_check;
ALTER TABLE sessions ADD CONSTRAINT sessions_category_check
    CHECK (category IS NULL OR category IN ('grower', 'corporate', 'investor', 'agritech', 'exploring'));

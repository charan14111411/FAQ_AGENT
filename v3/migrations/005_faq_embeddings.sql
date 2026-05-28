CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS faq_embeddings (
  id        SERIAL PRIMARY KEY,
  category  VARCHAR(30) NOT NULL,
  question  TEXT NOT NULL UNIQUE,
  answer    TEXT NOT NULL,
  embedding vector(384)
);
CREATE INDEX IF NOT EXISTS idx_faq_embeddings_category ON faq_embeddings(category);

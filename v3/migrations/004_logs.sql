CREATE TABLE IF NOT EXISTS logs (
  id         BIGSERIAL PRIMARY KEY,
  level      VARCHAR(10) NOT NULL,
  event      VARCHAR(80) NOT NULL,
  user_id    UUID,
  session_id UUID,
  meta       JSONB DEFAULT '{}',
  message    TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_logs_event      ON logs(event);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_level      ON logs(level);

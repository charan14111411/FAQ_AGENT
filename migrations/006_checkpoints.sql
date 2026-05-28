CREATE TABLE IF NOT EXISTS checkpoints (
  id                   BIGSERIAL PRIMARY KEY,
  turn_id              UUID NOT NULL,
  checkpoint_type      VARCHAR(40) NOT NULL,
  status               VARCHAR(20) NOT NULL DEFAULT 'ok',
  user_id              UUID,
  session_id           UUID,
  category             VARCHAR(30),
  agent                VARCHAR(60),
  user_message_id      UUID,
  assistant_message_id UUID,
  metadata             JSONB NOT NULL DEFAULT '{}',
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_turn_id
  ON checkpoints(turn_id);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session_id
  ON checkpoints(session_id);

CREATE INDEX IF NOT EXISTS idx_checkpoints_user_id
  ON checkpoints(user_id);

CREATE INDEX IF NOT EXISTS idx_checkpoints_type
  ON checkpoints(checkpoint_type);

CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at
  ON checkpoints(created_at DESC);

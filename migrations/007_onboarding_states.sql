CREATE TABLE IF NOT EXISTS onboarding_states (
  conversation_id UUID PRIMARY KEY,
  step            VARCHAR(20) NOT NULL,
  profile         JSONB NOT NULL DEFAULT '{}',
  user_id         UUID,
  session_id      UUID,
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_onboarding_states_updated_at
  ON onboarding_states(updated_at DESC);

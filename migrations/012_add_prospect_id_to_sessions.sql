-- Add prospect_id column to sessions to store the CRM prospect reference
-- returned by the BusinessCentral createProspect API call on session creation.
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS prospect_id VARCHAR(50) DEFAULT NULL;

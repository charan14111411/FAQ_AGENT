-- Create prospects table to map sequential prospect_ids to unique phone numbers locally
CREATE TABLE IF NOT EXISTS prospects (
    prospect_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    phone_number VARCHAR(50) UNIQUE NOT NULL,
    source VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

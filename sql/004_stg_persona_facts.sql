-- Structured persona facts grounded in stg.messages (evidence ids).
-- English-only DDL comments.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE TABLE IF NOT EXISTS stg.persona_facts (
    id                 BIGSERIAL PRIMARY KEY,
    actor_key          TEXT NOT NULL,
    fact_key           TEXT NOT NULL,
    statement          TEXT NOT NULL,
    evidence_message_ids BIGINT[] NOT NULL DEFAULT '{}',
    confidence         REAL NOT NULL DEFAULT 0.7,
    status             TEXT NOT NULL DEFAULT 'active',
    source             TEXT NOT NULL DEFAULT 'extract_stg_messages',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (actor_key, fact_key)
);

CREATE INDEX IF NOT EXISTS idx_persona_facts_actor_status
    ON stg.persona_facts (actor_key, status);

COMMENT ON TABLE stg.persona_facts IS
    'Normalized claims about an actor; statement must be supported by evidence_message_ids in stg.messages.';

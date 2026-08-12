-- Persona governance objects previously kept as private JSON.
-- English-only DDL comments.

CREATE SCHEMA IF NOT EXISTS stg;

-- Official topic slots (was data/private/fact_topics/<actor>.json)
CREATE TABLE IF NOT EXISTS stg.persona_topic_specs (
    actor_key   TEXT NOT NULL,
    fact_key    TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'preference',
    match_any   TEXT[] NOT NULL DEFAULT '{}',
    prefer_any  TEXT[] NOT NULL DEFAULT '{}',
    min_len     INT NOT NULL DEFAULT 4,
    max_len     INT NOT NULL DEFAULT 80,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    source      TEXT NOT NULL DEFAULT 'manual',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (actor_key, fact_key)
);

CREATE INDEX IF NOT EXISTS idx_persona_topic_specs_actor_enabled
    ON stg.persona_topic_specs (actor_key, enabled);

-- Rejected / blocked topic keys and phrases
CREATE TABLE IF NOT EXISTS stg.persona_topic_blocks (
    actor_key   TEXT NOT NULL,
    block_kind  TEXT NOT NULL CHECK (block_kind IN ('fact_key', 'phrase')),
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (actor_key, block_kind, value)
);

-- Pending topic candidates (was topics_pending JSON)
CREATE TABLE IF NOT EXISTS stg.persona_topic_candidates (
    actor_key             TEXT NOT NULL,
    fact_key              TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'pending',
    kind                  TEXT NOT NULL DEFAULT 'preference',
    phrase                TEXT,
    match_any             TEXT[] NOT NULL DEFAULT '{}',
    prefer_any            TEXT[] NOT NULL DEFAULT '{}',
    min_len               INT NOT NULL DEFAULT 4,
    max_len               INT NOT NULL DEFAULT 80,
    hit_count             INT,
    score                 REAL,
    latest_epoch          BIGINT,
    example_message_ids   BIGINT[] NOT NULL DEFAULT '{}',
    example_excerpts      TEXT[] NOT NULL DEFAULT '{}',
    source                TEXT NOT NULL DEFAULT 'topic_mine_v1',
    enabled               BOOLEAN NOT NULL DEFAULT TRUE,
    meta                  JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (actor_key, fact_key)
);

CREATE TABLE IF NOT EXISTS stg.persona_topic_extract_runs (
    actor_key     TEXT PRIMARY KEY,
    generated_at  TIMESTAMPTZ,
    meta          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Rejected fact keys for extract skip
CREATE TABLE IF NOT EXISTS stg.persona_fact_blocks (
    actor_key   TEXT NOT NULL,
    fact_key    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (actor_key, fact_key)
);

-- Pending fact candidates (was facts_pending JSON)
CREATE TABLE IF NOT EXISTS stg.persona_fact_candidates (
    actor_key              TEXT NOT NULL,
    fact_key               TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'pending',
    statement              TEXT NOT NULL DEFAULT '',
    evidence_message_ids   BIGINT[] NOT NULL DEFAULT '{}',
    evidence_count_scanned INT,
    create_time_epoch      BIGINT,
    score                  REAL,
    score_breakdown        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source                 TEXT,
    op                     TEXT NOT NULL DEFAULT 'upsert',
    meta                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (actor_key, fact_key)
);

CREATE TABLE IF NOT EXISTS stg.persona_fact_extract_runs (
    actor_key     TEXT PRIMARY KEY,
    generated_at  TIMESTAMPTZ,
    meta          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Persona cards (was data/private/personas/<actor>.json)
CREATE TABLE IF NOT EXISTS stg.persona_cards (
    actor_key     TEXT PRIMARY KEY,
    display_name  TEXT,
    relationship  TEXT,
    voice_notes   JSONB NOT NULL DEFAULT '[]'::jsonb,
    known_facts   JSONB NOT NULL DEFAULT '[]'::jsonb,
    boundaries    JSONB NOT NULL DEFAULT '[]'::jsonb,
    extra_rules   JSONB NOT NULL DEFAULT '[]'::jsonb,
    card          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE stg.persona_topic_specs IS
    'Approved topic slots for fact extract; replaces private fact_topics JSON.';
COMMENT ON TABLE stg.persona_topic_candidates IS
    'Human-review queue for mined topic slots.';
COMMENT ON TABLE stg.persona_fact_candidates IS
    'Human-review queue for extracted fact propositions.';
COMMENT ON TABLE stg.persona_cards IS
    'Persona card / voice boundaries; card JSONB keeps full document.';

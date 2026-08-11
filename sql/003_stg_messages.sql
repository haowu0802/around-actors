-- Unified conversation staging: one row per raw message, with model-facing text.
-- Built from raw.messages LEFT JOIN stg.voice_messages.
-- Grain: (actor_key, create_time_text, local_id_text) == raw message identity.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE TABLE IF NOT EXISTS stg.messages (
    id                  BIGSERIAL PRIMARY KEY,

    actor_key           TEXT NOT NULL,
    raw_message_id      BIGINT NOT NULL,
    source_file         TEXT,
    voice_stg_id        BIGINT,

    create_time_text    TEXT NOT NULL,
    local_id_text       TEXT NOT NULL,
    create_time_epoch   BIGINT,
    local_id_num        BIGINT,
    sort_seq            TEXT,
    local_type          TEXT,
    msg_kind            TEXT NOT NULL,

    sender_username     TEXT,
    sender_display      TEXT,
    is_mine             INTEGER,
    speaker_role        TEXT NOT NULL DEFAULT 'unknown',

    text_raw            TEXT,
    text_content        TEXT,
    text_source         TEXT NOT NULL,
    has_semantic_text   BOOLEAN NOT NULL DEFAULT FALSE,

    asr_status          TEXT,
    wav_path            TEXT,
    duration_sec        DOUBLE PRECISION,
    server_id           TEXT,

    staged_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT stg_messages_raw_uid UNIQUE (raw_message_id),
    CONSTRAINT stg_messages_actor_msg_uid
        UNIQUE (actor_key, create_time_text, local_id_text),
    CONSTRAINT stg_messages_msg_kind_chk
        CHECK (msg_kind IN ('text', 'voice', 'image', 'video', 'sticker', 'call', 'system', 'other')),
    CONSTRAINT stg_messages_text_source_chk
        CHECK (text_source IN ('export_text', 'asr', 'placeholder', 'empty')),
    CONSTRAINT stg_messages_speaker_role_chk
        CHECK (speaker_role IN ('self', 'actor', 'unknown'))
);

CREATE INDEX IF NOT EXISTS stg_messages_actor_time_idx
    ON stg.messages (actor_key, create_time_epoch);

CREATE INDEX IF NOT EXISTS stg_messages_actor_semantic_idx
    ON stg.messages (actor_key, has_semantic_text);

CREATE INDEX IF NOT EXISTS stg_messages_actor_kind_idx
    ON stg.messages (actor_key, msg_kind);

CREATE INDEX IF NOT EXISTS stg_messages_voice_stg_idx
    ON stg.messages (voice_stg_id);

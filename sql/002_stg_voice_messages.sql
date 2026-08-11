-- Staging table for WeChat voice messages (audio assets + future ASR).
-- Grain: one row per voice message per actor_key.
-- Join back to raw.messages on (actor_key, create_time, local_id) text keys.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE TABLE IF NOT EXISTS stg.voice_messages (
    id                  BIGSERIAL PRIMARY KEY,

    -- Identity / lineage
    actor_key           TEXT NOT NULL,
    raw_message_id      BIGINT,
    source_file         TEXT,
    audio_bundle_key    TEXT,
    source_system       TEXT NOT NULL DEFAULT 'wechat_silk_wav',

    -- Stable join keys to raw.messages (keep text form exactly as exported)
    create_time_text    TEXT NOT NULL,
    local_id_text       TEXT NOT NULL,
    create_time_epoch   BIGINT,
    local_id_num        BIGINT,
    local_type          TEXT NOT NULL DEFAULT '34',
    server_id           TEXT,

    -- Speakers (copied from raw at stage time; is_mine may be unreliable)
    sender_username     TEXT,
    sender_display      TEXT,
    is_mine             INTEGER,

    -- Voice XML / container metadata (optional)
    voice_duration_ms   INTEGER,
    voice_payload_bytes INTEGER,

    -- Audio asset (decoded WAV; do not overwrite with enhanced copies)
    wav_path            TEXT,
    wav_sha256          TEXT,
    sample_rate         INTEGER,
    channels            INTEGER,
    pcm_bits            INTEGER,
    duration_sec        DOUBLE PRECISION,

    -- ASR (filled by a later Whisper job)
    asr_status          TEXT NOT NULL DEFAULT 'pending',
    asr_transcript      TEXT,
    asr_language        TEXT,
    asr_model           TEXT,
    asr_device          TEXT,
    asr_error           TEXT,
    asr_at              TIMESTAMPTZ,

    staged_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT stg_voice_messages_actor_msg_uid
        UNIQUE (actor_key, create_time_text, local_id_text),

    CONSTRAINT stg_voice_messages_asr_status_chk
        CHECK (asr_status IN ('pending', 'ok', 'failed', 'skipped'))
);

CREATE INDEX IF NOT EXISTS stg_voice_messages_actor_status_idx
    ON stg.voice_messages (actor_key, asr_status);

CREATE INDEX IF NOT EXISTS stg_voice_messages_actor_time_idx
    ON stg.voice_messages (actor_key, create_time_epoch);

CREATE INDEX IF NOT EXISTS stg_voice_messages_raw_id_idx
    ON stg.voice_messages (raw_message_id);

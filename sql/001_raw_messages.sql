-- Raw landing table for WeChat-export-tool JSON messages (1:1 field mapping).
-- Apply: via ingest script --ensure-schema, or any SQL client.

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.messages (
    id                      BIGSERIAL PRIMARY KEY,
    actor_key               TEXT NOT NULL,
    source_file             TEXT NOT NULL,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT now(),

    wcdb_ct_message_content TEXT,
    wcdb_ct_source          TEXT,
    db_path                 TEXT,
    compress_content        TEXT,
    create_time             TEXT,
    download_status         TEXT,
    local_id                TEXT,
    local_type              TEXT,
    message_content         TEXT,
    origin_source           TEXT,
    packed_info_data        TEXT,
    real_sender_id          TEXT,
    sender_username         TEXT,
    server_id               TEXT,
    server_seq              TEXT,
    sort_seq                TEXT,
    source                  TEXT,
    status                  TEXT,
    table_name              TEXT,
    upload_status           TEXT,
    is_mine                 INTEGER,
    sender_display          TEXT,
    time_str                TEXT
);

CREATE INDEX IF NOT EXISTS raw_messages_actor_source_idx
    ON raw.messages (actor_key, source_file);

CREATE INDEX IF NOT EXISTS raw_messages_local_type_idx
    ON raw.messages (local_type);

CREATE INDEX IF NOT EXISTS raw_messages_actor_type_idx
    ON raw.messages (actor_key, local_type);

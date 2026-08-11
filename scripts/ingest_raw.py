#!/usr/bin/env python3
"""Load a WeChat-export-tool JSON array into Postgres raw.messages.

No private paths are hardcoded. Pass --file / --actor-key or env vars.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

# JSON key -> DB column (export uses WCDB_* / _db_path names)
FIELD_MAP = [
    ("WCDB_CT_message_content", "wcdb_ct_message_content"),
    ("WCDB_CT_source", "wcdb_ct_source"),
    ("_db_path", "db_path"),
    ("compress_content", "compress_content"),
    ("create_time", "create_time"),
    ("download_status", "download_status"),
    ("local_id", "local_id"),
    ("local_type", "local_type"),
    ("message_content", "message_content"),
    ("origin_source", "origin_source"),
    ("packed_info_data", "packed_info_data"),
    ("real_sender_id", "real_sender_id"),
    ("sender_username", "sender_username"),
    ("server_id", "server_id"),
    ("server_seq", "server_seq"),
    ("sort_seq", "sort_seq"),
    ("source", "source"),
    ("status", "status"),
    ("table_name", "table_name"),
    ("upload_status", "upload_status"),
    ("is_mine", "is_mine"),
    ("sender_display", "sender_display"),
    ("time_str", "time_str"),
]

INSERT_COLS = ["actor_key", "source_file", "ingested_at"] + [c for _, c in FIELD_MAP]
INSERT_SQL = (
    "INSERT INTO raw.messages ("
    + ", ".join(INSERT_COLS)
    + ") VALUES ("
    + ", ".join(["%s"] * len(INSERT_COLS))
    + ")"
)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def empty_to_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return value


def row_from_message(msg: dict, actor_key: str, source_file: str, ingested_at: datetime):
    values = [actor_key, source_file, ingested_at]
    for json_key, _ in FIELD_MAP:
        val = msg.get(json_key)
        if json_key == "is_mine":
            if val is None or val == "":
                values.append(None)
            else:
                values.append(int(val))
        else:
            # Preserve exporter typing: most fields arrive as strings.
            values.append(empty_to_none(val if val is None or isinstance(val, str) else str(val)))
    return tuple(values)


def ensure_schema(conn: psycopg.Connection, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest export JSON into raw.messages")
    p.add_argument(
        "--file",
        default=os.environ.get("EXPORT_FILE"),
        help="Path to export JSON (or set EXPORT_FILE)",
    )
    p.add_argument(
        "--actor-key",
        default=os.environ.get("ACTOR_KEY"),
        help="Stable actor id slug, e.g. person_a (or set ACTOR_KEY)",
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres URL (or set DATABASE_URL)",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows for this actor_key + source_file before insert",
    )
    p.add_argument(
        "--ensure-schema",
        action="store_true",
        help="Apply sql/001_raw_messages.sql before ingest",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Insert batch size (default 1000)",
    )
    return p.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    args = parse_args()
    if not args.file or not args.actor_key or not args.database_url:
        print(
            "Missing --file / --actor-key / --database-url (or EXPORT_FILE, ACTOR_KEY, DATABASE_URL).",
            file=sys.stderr,
        )
        return 2

    path = Path(args.file).expanduser().resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        print("Export JSON must be a top-level array of message objects.", file=sys.stderr)
        return 2

    source_name = path.name
    ingested_at = datetime.now(timezone.utc)
    rows = [row_from_message(m, args.actor_key, source_name, ingested_at) for m in payload]

    sql_path = repo_root / "sql" / "001_raw_messages.sql"

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if args.ensure_schema:
            ensure_schema(conn, sql_path)

        with conn.cursor() as cur:
            if args.replace:
                cur.execute(
                    "DELETE FROM raw.messages WHERE actor_key = %s AND source_file = %s",
                    (args.actor_key, source_name),
                )
                deleted = cur.rowcount
            else:
                deleted = 0

            batch: list[tuple] = []
            inserted = 0
            for row in rows:
                batch.append(row)
                if len(batch) >= args.batch_size:
                    cur.executemany(INSERT_SQL, batch)
                    inserted += len(batch)
                    batch.clear()
            if batch:
                cur.executemany(INSERT_SQL, batch)
                inserted += len(batch)

            cur.execute(
                """
                SELECT local_type, COUNT(*) AS n
                FROM raw.messages
                WHERE actor_key = %s AND source_file = %s
                GROUP BY local_type
                ORDER BY n DESC
                LIMIT 10
                """,
                (args.actor_key, source_name),
            )
            top_types = cur.fetchall()

        conn.commit()

    print(f"actor_key={args.actor_key}")
    print(f"source_file={source_name}")
    print(f"deleted={deleted}")
    print(f"inserted={inserted}")
    print("top_local_types:")
    for r in top_types:
        print(f"  {r['local_type']}: {r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

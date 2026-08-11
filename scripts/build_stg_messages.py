#!/usr/bin/env python3
"""Build stg.messages from raw.messages + stg.voice_messages.

Produces a time-ordered conversation table with model-facing text_content.
No private paths hardcoded.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

LOCAL_TYPE_KIND = {
    "1": "text",
    "3": "image",
    "34": "voice",
    "43": "video",
    "47": "sticker",
    "50": "call",
    "10000": "system",
}

PLACEHOLDER = {
    "image": "[图片]",
    "video": "[视频]",
    "sticker": "[表情]",
    "call": "[通话]",
    "system": "[系统消息]",
    "other": "[其它消息]",
    "voice": "[语音]",
}

UPSERT_SQL = """
INSERT INTO stg.messages (
    actor_key,
    raw_message_id,
    source_file,
    voice_stg_id,
    create_time_text,
    local_id_text,
    create_time_epoch,
    local_id_num,
    sort_seq,
    local_type,
    msg_kind,
    sender_username,
    sender_display,
    is_mine,
    speaker_role,
    text_raw,
    text_content,
    text_source,
    has_semantic_text,
    asr_status,
    wav_path,
    duration_sec,
    server_id,
    staged_at,
    updated_at
) VALUES (
    %(actor_key)s,
    %(raw_message_id)s,
    %(source_file)s,
    %(voice_stg_id)s,
    %(create_time_text)s,
    %(local_id_text)s,
    %(create_time_epoch)s,
    %(local_id_num)s,
    %(sort_seq)s,
    %(local_type)s,
    %(msg_kind)s,
    %(sender_username)s,
    %(sender_display)s,
    %(is_mine)s,
    %(speaker_role)s,
    %(text_raw)s,
    %(text_content)s,
    %(text_source)s,
    %(has_semantic_text)s,
    %(asr_status)s,
    %(wav_path)s,
    %(duration_sec)s,
    %(server_id)s,
    %(staged_at)s,
    %(updated_at)s
)
ON CONFLICT (raw_message_id) DO UPDATE SET
    actor_key = EXCLUDED.actor_key,
    source_file = EXCLUDED.source_file,
    voice_stg_id = EXCLUDED.voice_stg_id,
    create_time_text = EXCLUDED.create_time_text,
    local_id_text = EXCLUDED.local_id_text,
    create_time_epoch = EXCLUDED.create_time_epoch,
    local_id_num = EXCLUDED.local_id_num,
    sort_seq = EXCLUDED.sort_seq,
    local_type = EXCLUDED.local_type,
    msg_kind = EXCLUDED.msg_kind,
    sender_username = EXCLUDED.sender_username,
    sender_display = EXCLUDED.sender_display,
    is_mine = EXCLUDED.is_mine,
    speaker_role = EXCLUDED.speaker_role,
    text_raw = EXCLUDED.text_raw,
    text_content = EXCLUDED.text_content,
    text_source = EXCLUDED.text_source,
    has_semantic_text = EXCLUDED.has_semantic_text,
    asr_status = EXCLUDED.asr_status,
    wav_path = EXCLUDED.wav_path,
    duration_sec = EXCLUDED.duration_sec,
    server_id = EXCLUDED.server_id,
    updated_at = EXCLUDED.updated_at
"""


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def ensure_schema(conn: psycopg.Connection, sql_path: Path) -> None:
    conn.execute(sql_path.read_text(encoding="utf-8"))
    conn.commit()


def looks_like_xml(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("<") and ">" in s[:200]


def clean_export_text(text: str | None) -> str:
    if not text:
        return ""
    t = str(text).replace("\x00", "").strip()
    if not t or looks_like_xml(t):
        return ""
    return t


def msg_kind_for(local_type: str | None) -> str:
    if local_type is None:
        return "other"
    return LOCAL_TYPE_KIND.get(str(local_type), "other")


def parse_csv_env_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def match_speaker(
    sender_username: str | None,
    sender_display: str | None,
    actor_matchers: list[str],
    self_matchers: list[str],
) -> str:
    fields = [sender_username or "", sender_display or ""]
    for m in self_matchers:
        for f in fields:
            if m and m in f:
                return "self"
    for m in actor_matchers:
        for f in fields:
            if m and m in f:
                return "actor"
    return "unknown"


def compose_text(kind: str, text_raw: str | None, asr_status: str | None, asr_transcript: str | None):
    if kind == "text":
        body = clean_export_text(text_raw)
        if body:
            return body, "export_text", True
        return "", "empty", False

    if kind == "voice":
        transcript = (asr_transcript or "").strip()
        if asr_status == "ok" and transcript:
            return transcript, "asr", True
        # Keep a placeholder so timeline stays intact without polluting RAG by default.
        return PLACEHOLDER["voice"], "placeholder", False

    if kind == "system":
        body = clean_export_text(text_raw)
        if body:
            return body, "export_text", False
        return PLACEHOLDER["system"], "placeholder", False

    return PLACEHOLDER.get(kind, PLACEHOLDER["other"]), "placeholder", False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stg.messages unified conversation table")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--actor-key",
        action="append",
        dest="actor_keys",
        default=None,
        help="Actor key to build (repeatable). Default: distinct actor_key in raw.messages",
    )
    p.add_argument(
        "--actor-match",
        action="append",
        default=None,
        help="Substring match against sender_username/display => speaker_role=actor (repeatable)",
    )
    p.add_argument(
        "--self-match",
        action="append",
        default=None,
        help="Substring match against sender_username/display => speaker_role=self (repeatable)",
    )
    p.add_argument("--replace", action="store_true", help="Delete existing stg.messages for selected actors first")
    p.add_argument("--ensure-schema", action="store_true", help="Apply sql/003_stg_messages.sql first")
    p.add_argument("--batch-size", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()
    if not args.database_url:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    actor_matchers = [x for x in (args.actor_match or []) if x] or parse_csv_env_list(
        os.environ.get("ACTOR_SPEAKER_MATCH")
    )
    self_matchers = [x for x in (args.self_match or []) if x] or parse_csv_env_list(
        os.environ.get("SELF_SPEAKER_MATCH")
    )
    now = datetime.now(timezone.utc)
    sql_path = repo_root / "sql" / "003_stg_messages.sql"

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if args.ensure_schema:
            ensure_schema(conn, sql_path)

        with conn.cursor() as cur:
            if args.actor_keys:
                actors = args.actor_keys
            else:
                cur.execute("SELECT DISTINCT actor_key FROM raw.messages ORDER BY 1")
                actors = [r["actor_key"] for r in cur.fetchall()]

            if not actors:
                print("no_actors_in_raw")
                return 0

            for actor in actors:
                if args.replace:
                    cur.execute("DELETE FROM stg.messages WHERE actor_key = %s", (actor,))
                    deleted = cur.rowcount
                else:
                    deleted = 0

                cur.execute(
                    """
                    SELECT
                        r.id AS raw_message_id,
                        r.actor_key,
                        r.source_file,
                        r.create_time,
                        r.local_id,
                        r.sort_seq,
                        r.local_type,
                        r.sender_username,
                        r.sender_display,
                        r.is_mine,
                        r.message_content,
                        r.server_id,
                        v.id AS voice_stg_id,
                        v.asr_status,
                        v.asr_transcript,
                        v.wav_path,
                        v.duration_sec
                    FROM raw.messages r
                    LEFT JOIN stg.voice_messages v
                      ON v.actor_key = r.actor_key
                     AND v.create_time_text = r.create_time
                     AND v.local_id_text = r.local_id
                    WHERE r.actor_key = %s
                    ORDER BY r.id
                    """,
                    (actor,),
                )
                rows = cur.fetchall()

                batch = []
                upserted = 0
                semantic = 0
                kind_counts: dict[str, int] = {}
                role_counts: dict[str, int] = {}
                source_counts: dict[str, int] = {}

                for r in rows:
                    ct = str(r["create_time"] or "")
                    lid = str(r["local_id"] or "")
                    local_type = str(r["local_type"]) if r["local_type"] is not None else None
                    kind = msg_kind_for(local_type)
                    text_content, text_source, has_sem = compose_text(
                        kind,
                        r["message_content"],
                        r["asr_status"],
                        r["asr_transcript"],
                    )
                    role = match_speaker(
                        r["sender_username"],
                        r["sender_display"],
                        actor_matchers,
                        self_matchers,
                    )
                    # Fallback: trust is_mine only when no explicit matchers hit.
                    if role == "unknown" and r["is_mine"] in (0, 1) and not actor_matchers and not self_matchers:
                        role = "self" if int(r["is_mine"]) == 1 else "actor"

                    payload = {
                        "actor_key": actor,
                        "raw_message_id": r["raw_message_id"],
                        "source_file": r["source_file"],
                        "voice_stg_id": r["voice_stg_id"],
                        "create_time_text": ct,
                        "local_id_text": lid,
                        "create_time_epoch": int(ct) if ct.isdigit() else None,
                        "local_id_num": int(lid) if lid.isdigit() else None,
                        "sort_seq": r["sort_seq"],
                        "local_type": local_type,
                        "msg_kind": kind,
                        "sender_username": r["sender_username"],
                        "sender_display": r["sender_display"],
                        "is_mine": r["is_mine"],
                        "speaker_role": role,
                        "text_raw": r["message_content"],
                        "text_content": text_content,
                        "text_source": text_source,
                        "has_semantic_text": has_sem,
                        "asr_status": r["asr_status"],
                        "wav_path": r["wav_path"],
                        "duration_sec": r["duration_sec"],
                        "server_id": r["server_id"],
                        "staged_at": now,
                        "updated_at": now,
                    }
                    batch.append(payload)
                    kind_counts[kind] = kind_counts.get(kind, 0) + 1
                    role_counts[role] = role_counts.get(role, 0) + 1
                    source_counts[text_source] = source_counts.get(text_source, 0) + 1
                    if has_sem:
                        semantic += 1

                    if len(batch) >= args.batch_size:
                        cur.executemany(UPSERT_SQL, batch)
                        upserted += len(batch)
                        batch.clear()

                if batch:
                    cur.executemany(UPSERT_SQL, batch)
                    upserted += len(batch)

                conn.commit()

                print(f"actor_key={actor}")
                print(f"deleted={deleted}")
                print(f"raw_rows={len(rows)}")
                print(f"upserted={upserted}")
                print(f"semantic_rows={semantic}")
                print(f"msg_kind={dict(sorted(kind_counts.items(), key=lambda x: -x[1]))}")
                print(f"speaker_role={role_counts}")
                print(f"text_source={source_counts}")

            cur.execute(
                """
                SELECT actor_key, count(*) AS n,
                       count(*) FILTER (WHERE has_semantic_text) AS semantic_n,
                       count(*) FILTER (WHERE msg_kind = 'voice' AND text_source = 'asr') AS voice_asr_n
                FROM stg.messages
                GROUP BY 1
                ORDER BY 1
                """
            )
            print("summary:")
            for r in cur.fetchall():
                print(
                    f"  {r['actor_key']}: rows={r['n']} semantic={r['semantic_n']} voice_asr={r['voice_asr_n']}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

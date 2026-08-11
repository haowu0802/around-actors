#!/usr/bin/env python3
"""Register decoded WeChat voice WAVs into stg.voice_messages.

Joins raw.messages (local_type=34) to files named {create_time}_{local_id}.wav.
Does not run ASR. No private paths are hardcoded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import psycopg
from psycopg.rows import dict_row

WAV_NAME_RE = re.compile(r"^(\d+)_(\d+)\.wav$", re.IGNORECASE)

UPSERT_SQL = """
INSERT INTO stg.voice_messages (
    actor_key,
    raw_message_id,
    source_file,
    audio_bundle_key,
    source_system,
    create_time_text,
    local_id_text,
    create_time_epoch,
    local_id_num,
    local_type,
    server_id,
    sender_username,
    sender_display,
    is_mine,
    voice_duration_ms,
    voice_payload_bytes,
    wav_path,
    wav_sha256,
    sample_rate,
    channels,
    pcm_bits,
    duration_sec,
    asr_status,
    staged_at,
    updated_at
) VALUES (
    %(actor_key)s,
    %(raw_message_id)s,
    %(source_file)s,
    %(audio_bundle_key)s,
    %(source_system)s,
    %(create_time_text)s,
    %(local_id_text)s,
    %(create_time_epoch)s,
    %(local_id_num)s,
    %(local_type)s,
    %(server_id)s,
    %(sender_username)s,
    %(sender_display)s,
    %(is_mine)s,
    %(voice_duration_ms)s,
    %(voice_payload_bytes)s,
    %(wav_path)s,
    %(wav_sha256)s,
    %(sample_rate)s,
    %(channels)s,
    %(pcm_bits)s,
    %(duration_sec)s,
    'pending',
    %(staged_at)s,
    %(updated_at)s
)
ON CONFLICT (actor_key, create_time_text, local_id_text) DO UPDATE SET
    raw_message_id = EXCLUDED.raw_message_id,
    source_file = EXCLUDED.source_file,
    audio_bundle_key = EXCLUDED.audio_bundle_key,
    source_system = EXCLUDED.source_system,
    create_time_epoch = EXCLUDED.create_time_epoch,
    local_id_num = EXCLUDED.local_id_num,
    local_type = EXCLUDED.local_type,
    server_id = EXCLUDED.server_id,
    sender_username = EXCLUDED.sender_username,
    sender_display = EXCLUDED.sender_display,
    is_mine = EXCLUDED.is_mine,
    voice_duration_ms = EXCLUDED.voice_duration_ms,
    voice_payload_bytes = EXCLUDED.voice_payload_bytes,
    wav_path = EXCLUDED.wav_path,
    wav_sha256 = EXCLUDED.wav_sha256,
    sample_rate = EXCLUDED.sample_rate,
    channels = EXCLUDED.channels,
    pcm_bits = EXCLUDED.pcm_bits,
    duration_sec = EXCLUDED.duration_sec,
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


def parse_voice_xml(message_content: str | None) -> tuple[int | None, int | None]:
    """Return (voice_duration_ms, voice_payload_bytes) from <voicemsg .../> if present."""
    if not message_content:
        return None, None
    text = message_content.strip()
    if "voicemsg" not in text.lower():
        return None, None
    # Exporter may wrap fragments; try a small searchable tree.
    try:
        root = ET.fromstring(text if text.startswith("<") else f"<root>{text}</root>")
    except ET.ParseError:
        # Fallback: attribute scrape
        dur = re.search(r'voicelength\s*=\s*"(\d+)"', text, re.I)
        nbytes = re.search(r'\blength\s*=\s*"(\d+)"', text, re.I)
        return (int(dur.group(1)) if dur else None, int(nbytes.group(1)) if nbytes else None)

    node = None
    if root.tag.lower().endswith("voicemsg"):
        node = root
    else:
        for el in root.iter():
            if el.tag.lower().endswith("voicemsg"):
                node = el
                break
    if node is None:
        return None, None

    def attr_int(*names: str) -> int | None:
        for name in names:
            for k, v in node.attrib.items():
                if k.lower() == name.lower() and str(v).isdigit():
                    return int(v)
        return None

    return attr_int("voicelength"), attr_int("length")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def wav_probe(path: Path) -> tuple[int | None, int | None, int | None, float | None]:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            pcm_bits = wf.getsampwidth() * 8
            nframes = wf.getnframes()
            duration = float(nframes) / float(sample_rate) if sample_rate else None
            return sample_rate, channels, pcm_bits, duration
    except wave.Error:
        return None, None, None, None


def index_wav_dir(wav_dir: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for path in sorted(wav_dir.glob("*.wav")):
        m = WAV_NAME_RE.match(path.name)
        if not m:
            continue
        out[(m.group(1), m.group(2))] = path.resolve()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register voice WAVs into stg.voice_messages")
    p.add_argument("--actor-key", default=os.environ.get("ACTOR_KEY"), help="Actor key in raw.messages")
    p.add_argument(
        "--wav-dir",
        default=os.environ.get("VOICE_WAV_DIR"),
        help="Directory of {create_time}_{local_id}.wav files (or VOICE_WAV_DIR)",
    )
    p.add_argument(
        "--audio-bundle-key",
        default=os.environ.get("AUDIO_BUNDLE_KEY"),
        help="Optional label for this audio folder (defaults to wav dir name)",
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres URL (or DATABASE_URL)",
    )
    p.add_argument(
        "--source-system",
        default=os.environ.get("VOICE_SOURCE_SYSTEM", "wechat_silk_wav"),
        help="Provenance label for staged rows",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing stg rows for this actor_key (+ audio_bundle_key if set) before upsert",
    )
    p.add_argument(
        "--ensure-schema",
        action="store_true",
        help="Apply sql/002_stg_voice_messages.sql before staging",
    )
    return p.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    if not args.actor_key or not args.wav_dir or not args.database_url:
        print(
            "Missing --actor-key / --wav-dir / --database-url "
            "(or ACTOR_KEY, VOICE_WAV_DIR, DATABASE_URL).",
            file=sys.stderr,
        )
        return 2

    wav_dir = Path(args.wav_dir).expanduser().resolve()
    if not wav_dir.is_dir():
        print(f"WAV directory not found: {wav_dir}", file=sys.stderr)
        return 2

    bundle_key = args.audio_bundle_key or wav_dir.name
    wav_index = index_wav_dir(wav_dir)
    now = datetime.now(timezone.utc)
    sql_path = repo_root / "sql" / "002_stg_voice_messages.sql"

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if args.ensure_schema:
            ensure_schema(conn, sql_path)

        with conn.cursor() as cur:
            if args.replace:
                if args.audio_bundle_key:
                    cur.execute(
                        """
                        DELETE FROM stg.voice_messages
                        WHERE actor_key = %s AND audio_bundle_key = %s
                        """,
                        (args.actor_key, bundle_key),
                    )
                else:
                    cur.execute(
                        "DELETE FROM stg.voice_messages WHERE actor_key = %s",
                        (args.actor_key,),
                    )
                deleted = cur.rowcount
            else:
                deleted = 0

            cur.execute(
                """
                SELECT
                    id,
                    source_file,
                    create_time,
                    local_id,
                    local_type,
                    server_id,
                    sender_username,
                    sender_display,
                    is_mine,
                    message_content
                FROM raw.messages
                WHERE actor_key = %s
                  AND local_type = '34'
                ORDER BY create_time, local_id
                """,
                (args.actor_key,),
            )
            raw_voices = cur.fetchall()

            upserted = 0
            matched = 0
            missing_wav = 0
            matched_keys: set[tuple[str, str]] = set()

            for row in raw_voices:
                ct = str(row["create_time"] or "")
                lid = str(row["local_id"] or "")
                key = (ct, lid)
                wav_path = wav_index.get(key)
                if wav_path is None:
                    missing_wav += 1
                    continue

                matched += 1
                matched_keys.add(key)
                sample_rate, channels, pcm_bits, duration_sec = wav_probe(wav_path)
                voice_duration_ms, voice_payload_bytes = parse_voice_xml(row["message_content"])

                payload = {
                    "actor_key": args.actor_key,
                    "raw_message_id": row["id"],
                    "source_file": row["source_file"],
                    "audio_bundle_key": bundle_key,
                    "source_system": args.source_system,
                    "create_time_text": ct,
                    "local_id_text": lid,
                    "create_time_epoch": int(ct) if ct.isdigit() else None,
                    "local_id_num": int(lid) if lid.isdigit() else None,
                    "local_type": str(row["local_type"] or "34"),
                    "server_id": row["server_id"],
                    "sender_username": row["sender_username"],
                    "sender_display": row["sender_display"],
                    "is_mine": row["is_mine"],
                    "voice_duration_ms": voice_duration_ms,
                    "voice_payload_bytes": voice_payload_bytes,
                    "wav_path": str(wav_path),
                    "wav_sha256": sha256_file(wav_path),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "pcm_bits": pcm_bits,
                    "duration_sec": duration_sec,
                    "staged_at": now,
                    "updated_at": now,
                }
                cur.execute(UPSERT_SQL, payload)
                upserted += 1

            orphan_wav = len(set(wav_index) - matched_keys)

            cur.execute(
                """
                SELECT asr_status, COUNT(*) AS n
                FROM stg.voice_messages
                WHERE actor_key = %s
                GROUP BY asr_status
                ORDER BY n DESC
                """,
                (args.actor_key,),
            )
            status_counts = cur.fetchall()

            cur.execute(
                """
                SELECT COUNT(*) AS n, COALESCE(SUM(duration_sec), 0) AS dur
                FROM stg.voice_messages
                WHERE actor_key = %s AND audio_bundle_key = %s
                """,
                (args.actor_key, bundle_key),
            )
            summary = cur.fetchone()

        conn.commit()

    print(f"actor_key={args.actor_key}")
    print(f"wav_dir={wav_dir}")
    print(f"audio_bundle_key={bundle_key}")
    print(f"raw_voice_rows={len(raw_voices)}")
    print(f"wav_files_indexed={len(wav_index)}")
    print(f"deleted={deleted}")
    print(f"matched={matched}")
    print(f"upserted={upserted}")
    print(f"missing_wav={missing_wav}")
    print(f"orphan_wav={orphan_wav}")
    print(f"staged_rows_for_bundle={summary['n']}")
    print(f"staged_duration_sec={round(float(summary['dur']), 1)}")
    print("asr_status_counts:")
    for r in status_counts:
        print(f"  {r['asr_status']}: {r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

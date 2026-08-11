#!/usr/bin/env python3
"""Run Whisper ASR for staged voice rows and write transcripts into stg.voice_messages.

Reads pending rows with wav_path; updates asr_* columns. No private paths hardcoded.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR staged voice WAVs into stg.voice_messages")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--actor-key",
        action="append",
        dest="actor_keys",
        default=None,
        help="Limit to actor_key (repeatable). Default: all actors with pending rows",
    )
    p.add_argument(
        "--limit-per-actor",
        type=int,
        default=0,
        help="Max rows per actor (0 = no limit). Useful for pilots.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", "large-v3"),
        help="faster-whisper model size/name (default large-v3)",
    )
    p.add_argument(
        "--device",
        default=os.environ.get("WHISPER_DEVICE", "cuda"),
        choices=["cuda", "cpu", "auto"],
        help="Inference device (default cuda)",
    )
    p.add_argument(
        "--compute-type",
        default=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
        help="ctranslate2 compute type, e.g. float16 / int8_float16 / int8",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("WHISPER_LANGUAGE", "zh"),
        help="Language code passed to Whisper (default zh)",
    )
    p.add_argument(
        "--status",
        default="pending",
        help="Only rows with this asr_status (default pending)",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Include asr_status=failed in addition to --status",
    )
    return p.parse_args()


def resolve_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type_override_or_empty)."""
    if requested != "auto":
        return requested, ""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", ""
    except Exception:
        pass
    return "cpu", "int8"


def load_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    print(f"loading_model={model_name} device={device} compute_type={compute_type}")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def fetch_batch(
    cur,
    actor_key: str,
    status: str,
    retry_failed: bool,
    limit: int,
) -> list[dict]:
    statuses = [status]
    if retry_failed and "failed" not in statuses:
        statuses.append("failed")
    sql = """
        SELECT id, actor_key, wav_path, create_time_text, local_id_text
        FROM stg.voice_messages
        WHERE actor_key = %s
          AND asr_status = ANY(%s)
          AND wav_path IS NOT NULL
          AND wav_path <> ''
        ORDER BY create_time_epoch NULLS LAST, local_id_num NULLS LAST, id
    """
    params: list = [actor_key, statuses]
    if limit and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    cur.execute(sql, params)
    return list(cur.fetchall())


_T2S = None


def to_simplified(text: str) -> str:
    """Convert Traditional Chinese characters to Simplified (no-op for already-simplified)."""
    global _T2S
    if not text:
        return text
    if _T2S is None:
        from opencc import OpenCC

        _T2S = OpenCC("t2s")
    return _T2S.convert(text)


def transcribe_one(model, wav_path: str, language: str) -> str:
    segments, info = model.transcribe(
        wav_path,
        language=language or None,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    parts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
    text = "".join(parts).strip()
    # Whisper zh outputs may mix Traditional; normalize to Simplified for downstream.
    return to_simplified(text)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    if not args.database_url:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    device, auto_ct = resolve_device(args.device)
    compute_type = auto_ct or args.compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    try:
        model = load_model(args.model, device, compute_type)
    except Exception as e:
        if device == "cuda":
            print(f"cuda_load_failed={e!r}; falling back to cpu/int8", file=sys.stderr)
            device = "cpu"
            compute_type = "int8"
            model = load_model(args.model, device, compute_type)
        else:
            raise

    now = datetime.now(timezone.utc)

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if args.actor_keys:
                actors = args.actor_keys
            else:
                cur.execute(
                    """
                    SELECT DISTINCT actor_key
                    FROM stg.voice_messages
                    WHERE asr_status = %s
                       OR (%s AND asr_status = 'failed')
                    ORDER BY actor_key
                    """,
                    (args.status, args.retry_failed),
                )
                actors = [r["actor_key"] for r in cur.fetchall()]

            if not actors:
                print("no_actors_with_matching_rows")
                return 0

            totals = {"ok": 0, "failed": 0, "missing_file": 0}

            for actor in actors:
                rows = fetch_batch(
                    cur,
                    actor,
                    args.status,
                    args.retry_failed,
                    args.limit_per_actor,
                )
                print(f"actor={actor} batch={len(rows)}")
                for i, row in enumerate(rows, 1):
                    wav = Path(row["wav_path"])
                    if not wav.is_file():
                        cur.execute(
                            """
                            UPDATE stg.voice_messages
                            SET asr_status = 'failed',
                                asr_error = %s,
                                asr_model = %s,
                                asr_device = %s,
                                asr_language = %s,
                                asr_at = %s,
                                updated_at = %s
                            WHERE id = %s
                            """,
                            (
                                f"wav_missing: {wav}",
                                args.model,
                                device,
                                args.language,
                                now,
                                now,
                                row["id"],
                            ),
                        )
                        conn.commit()
                        totals["missing_file"] += 1
                        print(f"  [{i}/{len(rows)}] id={row['id']} FAIL missing_file")
                        continue

                    try:
                        text = transcribe_one(model, str(wav), args.language)
                        cur.execute(
                            """
                            UPDATE stg.voice_messages
                            SET asr_status = 'ok',
                                asr_transcript = %s,
                                asr_error = NULL,
                                asr_model = %s,
                                asr_device = %s,
                                asr_language = %s,
                                asr_at = %s,
                                updated_at = %s
                            WHERE id = %s
                            """,
                            (
                                text,
                                args.model,
                                device,
                                args.language,
                                now,
                                now,
                                row["id"],
                            ),
                        )
                        conn.commit()
                        totals["ok"] += 1
                        preview = text if len(text) <= 80 else text[:80] + "…"
                        print(f"  [{i}/{len(rows)}] id={row['id']} OK {preview!r}")
                    except Exception as e:
                        cur.execute(
                            """
                            UPDATE stg.voice_messages
                            SET asr_status = 'failed',
                                asr_error = %s,
                                asr_model = %s,
                                asr_device = %s,
                                asr_language = %s,
                                asr_at = %s,
                                updated_at = %s
                            WHERE id = %s
                            """,
                            (
                                repr(e)[:1000],
                                args.model,
                                device,
                                args.language,
                                now,
                                now,
                                row["id"],
                            ),
                        )
                        conn.commit()
                        totals["failed"] += 1
                        print(f"  [{i}/{len(rows)}] id={row['id']} FAIL {e!r}")

        # summary counts
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT actor_key, asr_status, COUNT(*) AS n
                FROM stg.voice_messages
                GROUP BY 1, 2
                ORDER BY 1, 2
                """
            )
            summary = cur.fetchall()

    print(f"device_used={device} compute_type={compute_type} model={args.model}")
    print(f"totals={totals}")
    print("status_by_actor:")
    for r in summary:
        print(f"  {r['actor_key']} {r['asr_status']}: {r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

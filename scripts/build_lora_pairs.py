#!/usr/bin/env python3
"""Build LoRA-style (context -> actor reply) JSONL pairs from stg.messages.

Hold-out is by time quantile so eval chats are never trained on.
Writes under data/private/lora/<actor_key>/ by default (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    p = argparse.ArgumentParser(description="Build LoRA chat pairs from stg.messages")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--actor-key", required=True)
    p.add_argument("--context-turns", type=int, default=3, help="Prior turns before actor reply")
    p.add_argument("--max-gap-sec", type=int, default=3600, help="Max seconds between adjacent turns")
    p.add_argument("--min-reply-len", type=int, default=4)
    p.add_argument("--max-reply-len", type=int, default=120)
    p.add_argument("--holdout-fraction", type=float, default=0.1)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output dir (default: data/private/lora/<actor_key>)",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional cap after filtering (0=all)")
    return p.parse_args()


def is_noisy(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.startswith("[") and t.endswith("]"):
        return True
    # Mostly punctuation / emoji placeholders
    alnum = sum(ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") for ch in t)
    return alnum < 2


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()
    if not args.database_url:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "data" / "private" / "lora" / args.actor_key
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, speaker_role, sender_display, text_content, create_time_epoch
                FROM stg.messages
                WHERE actor_key = %s
                  AND has_semantic_text = TRUE
                  AND text_content IS NOT NULL
                  AND text_content <> ''
                  AND create_time_epoch IS NOT NULL
                ORDER BY create_time_epoch ASC, id ASC
                """,
                (args.actor_key,),
            )
            rows = list(cur.fetchall())

    # Build contiguous windows ending on actor replies.
    pairs: list[dict] = []
    for i, row in enumerate(rows):
        if row["speaker_role"] != "actor":
            continue
        reply = (row["text_content"] or "").strip().replace("\n", " ")
        if is_noisy(reply):
            continue
        if not (args.min_reply_len <= len(reply) <= args.max_reply_len):
            continue
        if reply.startswith("[") or "<" in reply:
            continue

        ctx: list[dict] = []
        j = i - 1
        prev_ts = int(row["create_time_epoch"])
        while j >= 0 and len(ctx) < args.context_turns:
            prev = rows[j]
            ts = int(prev["create_time_epoch"] or 0)
            if prev_ts - ts > args.max_gap_sec:
                break
            text = (prev["text_content"] or "").strip().replace("\n", " ")
            if text and not is_noisy(text):
                role = "user" if prev["speaker_role"] == "self" else "assistant"
                # Only keep self/actor turns in the supervised window.
                if prev["speaker_role"] in {"self", "actor"}:
                    ctx.append({"role": role, "content": text[:200]})
                    prev_ts = ts
            j -= 1
        ctx.reverse()
        if not ctx:
            continue
        # Require at least one user (self) turn in context.
        if not any(m["role"] == "user" for m in ctx):
            continue

        pairs.append(
            {
                "id": f"{args.actor_key}-{row['id']}",
                "actor_key": args.actor_key,
                "reply_message_id": int(row["id"]),
                "create_time_epoch": int(row["create_time_epoch"]),
                "messages": ctx + [{"role": "assistant", "content": reply}],
            }
        )

    if args.limit and len(pairs) > args.limit:
        pairs = pairs[: args.limit]

    if not pairs:
        print("No pairs built", file=sys.stderr)
        return 1

    pairs.sort(key=lambda p: p["create_time_epoch"])
    cut = max(1, int(len(pairs) * (1.0 - args.holdout_fraction)))
    if cut >= len(pairs):
        cut = len(pairs) - 1
    train, holdout = pairs[:cut], pairs[cut:]

    def write_jsonl(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = out_dir / "train.jsonl"
    holdout_path = out_dir / "holdout.jsonl"
    meta_path = out_dir / "meta.json"
    write_jsonl(train_path, train)
    write_jsonl(holdout_path, holdout)
    meta = {
        "actor_key": args.actor_key,
        "context_turns": args.context_turns,
        "max_gap_sec": args.max_gap_sec,
        "min_reply_len": args.min_reply_len,
        "max_reply_len": args.max_reply_len,
        "holdout_fraction": args.holdout_fraction,
        "train_n": len(train),
        "holdout_n": len(holdout),
        "total_n": len(pairs),
        "train_path": str(train_path),
        "holdout_path": str(holdout_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build stg.persona_facts from stg.messages (topic heuristics + evidence ids).

Does not invent prose: each statement is a real actor utterance (optionally trimmed).
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
    p = argparse.ArgumentParser(description="Extract persona facts from stg.messages")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--actor-key", required=True)
    p.add_argument("--ensure-schema", action="store_true")
    p.add_argument("--replace", action="store_true", help="Delete existing facts for actor first")
    p.add_argument("--evidence-limit", type=int, default=5)
    p.add_argument(
        "--topics-file",
        default=None,
        help="JSON list of topic specs (default: data/private/fact_topics/<actor>.json if present)",
    )
    p.add_argument(
        "--export-json",
        default=None,
        help="Also write extracted facts JSON (optional local path)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_topics(repo_root: Path, actor_key: str, explicit: str | None) -> tuple[list[dict], Path]:
    """Load topic specs from JSON. Language-specific match strings stay out of this script."""
    path: Path | None = None
    if explicit:
        path = Path(explicit)
    else:
        candidate = repo_root / "data" / "private" / "fact_topics" / f"{actor_key}.json"
        if candidate.is_file():
            path = candidate
        else:
            example = repo_root / "data" / "samples" / "fact_topics.example.json"
            raise SystemExit(
                f"No topics file for actor_key={actor_key!r}. "
                f"Expected {candidate} (or pass --topics-file). "
                f"See example: {example}"
            )
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        specs = data.get("topics") or data.get("TOPIC_SPECS") or []
    else:
        specs = data
    if not isinstance(specs, list) or not specs:
        raise SystemExit(f"No topics found in {path}")
    return specs, path


def ensure_schema(conn: psycopg.Connection, repo_root: Path) -> None:
    ddl = (repo_root / "sql" / "004_stg_persona_facts.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def fetch_candidates(
    conn: psycopg.Connection,
    actor_key: str,
    terms: list[str],
    min_len: int,
    max_len: int,
    limit: int,
) -> list[dict]:
    clauses = ["text_content ILIKE %s" for _ in terms]
    params: list = [actor_key, min_len, max_len, *[f"%{t}%" for t in terms], limit]
    sql = f"""
        SELECT id, text_content, create_time_epoch
        FROM stg.messages
        WHERE actor_key = %s
          AND speaker_role = 'actor'
          AND has_semantic_text = TRUE
          AND char_length(text_content) BETWEEN %s AND %s
          AND ({' OR '.join(clauses)})
        ORDER BY create_time_epoch DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def pick_statement(rows: list[dict], prefer_any: list[str]) -> tuple[str, list[int]]:
    if not rows:
        return "", []
    scored: list[tuple[float, dict]] = []
    for r in rows:
        text = (r["text_content"] or "").strip().replace("\n", " ")
        if not text:
            continue
        score = 0.0
        for p in prefer_any:
            if p in text:
                score += 3.0
        # Prefer shorter, clearer chat lines.
        n = len(text)
        if 6 <= n <= 40:
            score += 1.0
        elif n <= 60:
            score += 0.4
        scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], -(x[1].get("create_time_epoch") or 0)))
    best = scored[0][1]
    evidence_ids = [int(r["id"]) for _, r in scored[:5]]
    statement = (best["text_content"] or "").strip().replace("\n", " ")
    if len(statement) > 120:
        statement = statement[:120] + "…"
    return statement, evidence_ids


def upsert_fact(
    conn: psycopg.Connection,
    actor_key: str,
    fact_key: str,
    statement: str,
    evidence_ids: list[int],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_facts (
                actor_key, fact_key, statement, evidence_message_ids,
                confidence, status, source, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, 'active', 'extract_stg_messages', now())
            ON CONFLICT (actor_key, fact_key) DO UPDATE SET
                statement = EXCLUDED.statement,
                evidence_message_ids = EXCLUDED.evidence_message_ids,
                confidence = EXCLUDED.confidence,
                status = 'active',
                source = EXCLUDED.source,
                updated_at = now()
            """,
            (actor_key, fact_key, statement, evidence_ids, 0.75),
        )


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

    extracted: list[dict] = []
    topics, topics_path = resolve_topics(repo_root, args.actor_key, args.topics_file)
    print(f"topics_file={topics_path} topics_n={len(topics)} actor_key={args.actor_key}")
    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if args.ensure_schema:
            ensure_schema(conn, repo_root)

        if args.replace and not args.dry_run:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM stg.persona_facts WHERE actor_key = %s",
                    (args.actor_key,),
                )
            conn.commit()

        for spec in topics:
            rows = fetch_candidates(
                conn,
                args.actor_key,
                spec["match_any"],
                int(spec.get("min_len", 4)),
                int(spec.get("max_len", 80)),
                max(args.evidence_limit * 3, 15),
            )
            statement, evidence_ids = pick_statement(rows, spec.get("prefer_any") or [])
            if not statement or not evidence_ids:
                print(f"SKIP {spec['fact_key']}: no evidence")
                continue
            item = {
                "fact_key": spec["fact_key"],
                "statement": statement,
                "evidence_message_ids": evidence_ids[: args.evidence_limit],
                "evidence_count_scanned": len(rows),
            }
            extracted.append(item)
            print(
                f"FACT {spec['fact_key']}: {statement!r} "
                f"evidence={item['evidence_message_ids']}"
            )
            if not args.dry_run:
                upsert_fact(
                    conn,
                    args.actor_key,
                    spec["fact_key"],
                    statement,
                    item["evidence_message_ids"],
                )
        if not args.dry_run:
            conn.commit()

    if args.export_json:
        path = Path(args.export_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"actor_key": args.actor_key, "facts": extracted},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"export_json={path}")

    print(f"done actor_key={args.actor_key} facts={len(extracted)} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

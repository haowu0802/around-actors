#!/usr/bin/env python3
"""Extract persona-fact candidates from stg.messages into a pending file; apply approved rows.

Phase 1 workflow:
  extract -> data/private/facts_pending/<actor>.json  (scored, status=pending)
  (human sets status=approved on chosen rows)
  apply   -> upsert approved rows into stg.persona_facts

Language-specific topic match strings live in data/private/fact_topics/, not in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    p = argparse.ArgumentParser(
        description="Pending persona-fact extract/apply (human review gate)"
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--actor-key", required=True)
    p.add_argument(
        "--mode",
        choices=("extract", "apply", "list"),
        default="extract",
        help="extract=write pending JSON; apply=write approved to DB; list=print pending",
    )
    p.add_argument("--ensure-schema", action="store_true")
    p.add_argument("--evidence-limit", type=int, default=5)
    p.add_argument("--limit", type=int, default=10, help="Top-N candidates to keep on extract")
    p.add_argument(
        "--topics-file",
        default=None,
        help="Topic specs JSON (default: data/private/fact_topics/<actor>.json)",
    )
    p.add_argument(
        "--pending-file",
        default=None,
        help="Pending JSON path (default: data/private/facts_pending/<actor>.json)",
    )
    p.add_argument(
        "--replace-active",
        action="store_true",
        help="On apply: delete existing active facts for actor before upsert (dangerous)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_topics(repo_root: Path, actor_key: str, explicit: str | None) -> tuple[list[dict], Path]:
    if explicit:
        path = Path(explicit)
    else:
        path = repo_root / "data" / "private" / "fact_topics" / f"{actor_key}.json"
        if not path.is_file():
            example = repo_root / "data" / "samples" / "fact_topics.example.json"
            raise SystemExit(
                f"No topics file for actor_key={actor_key!r}. "
                f"Expected {path} (or pass --topics-file). See {example}"
            )
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        specs = data.get("topics") or []
    else:
        specs = data
    if not isinstance(specs, list) or not specs:
        raise SystemExit(f"No topics found in {path}")
    return specs, path


def default_pending_path(repo_root: Path, actor_key: str) -> Path:
    return repo_root / "data" / "private" / "facts_pending" / f"{actor_key}.json"


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


def score_candidate(
    *,
    statement: str,
    prefer_any: list[str],
    evidence_count_scanned: int,
    evidence_ids: list[int],
    create_time_epoch: int | None,
) -> float:
    """Simple explainable rule score in 0..100 (phase 1)."""
    score = 40.0
    score += min(30.0, float(evidence_count_scanned) * 4.0)
    score += min(10.0, float(len(evidence_ids)) * 2.0)
    if any(p and p in statement for p in prefer_any):
        score += 15.0
    n = len(statement)
    if 6 <= n <= 40:
        score += 10.0
    elif 41 <= n <= 60:
        score += 5.0
    elif n > 100:
        score -= 5.0
    # Mild recency bump if timestamp looks present (absolute scale ignored).
    if create_time_epoch:
        score += 3.0
    # Question / shrug soft penalty.
    if "？" in statement or "?" in statement:
        score -= 8.0
    if statement.count("哈") >= 3:
        score -= 4.0
    return max(0.0, min(100.0, score))


def pick_statement(rows: list[dict], prefer_any: list[str]) -> tuple[str, list[int], int | None]:
    if not rows:
        return "", [], None
    scored: list[tuple[float, dict]] = []
    for r in rows:
        text = (r["text_content"] or "").strip().replace("\n", " ")
        if not text:
            continue
        local = 0.0
        for p in prefer_any:
            if p in text:
                local += 3.0
        n = len(text)
        if 6 <= n <= 40:
            local += 1.0
        elif n <= 60:
            local += 0.4
        scored.append((local, r))
    scored.sort(key=lambda x: (-x[0], -(x[1].get("create_time_epoch") or 0)))
    best = scored[0][1]
    evidence_ids = [int(r["id"]) for _, r in scored[:5]]
    statement = (best["text_content"] or "").strip().replace("\n", " ")
    if len(statement) > 120:
        statement = statement[:120] + "…"
    ts = best.get("create_time_epoch")
    return statement, evidence_ids, int(ts) if ts is not None else None


def load_pending(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_pending(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upsert_fact(
    conn: psycopg.Connection,
    actor_key: str,
    fact_key: str,
    statement: str,
    evidence_ids: list[int],
    confidence: float,
    source: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_facts (
                actor_key, fact_key, statement, evidence_message_ids,
                confidence, status, source, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, 'active', %s, now())
            ON CONFLICT (actor_key, fact_key) DO UPDATE SET
                statement = EXCLUDED.statement,
                evidence_message_ids = EXCLUDED.evidence_message_ids,
                confidence = EXCLUDED.confidence,
                status = 'active',
                source = EXCLUDED.source,
                updated_at = now()
            """,
            (actor_key, fact_key, statement, evidence_ids, confidence, source),
        )


def cmd_extract(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    topics: list[dict],
    topics_path: Path,
    pending_path: Path,
    evidence_limit: int,
    limit: int,
    dry_run: bool,
) -> int:
    # Preserve prior human decisions when fact_key+statement match.
    prior = load_pending(pending_path)
    prior_status: dict[str, str] = {}
    for c in prior.get("candidates") or []:
        key = f"{c.get('fact_key')}||{c.get('statement')}"
        st = c.get("status")
        if st in {"approved", "rejected", "applied"}:
            prior_status[key] = st

    candidates: list[dict[str, Any]] = []
    for spec in topics:
        fact_key = str(spec.get("fact_key") or "").strip()
        if not fact_key:
            continue
        rows = fetch_candidates(
            conn,
            actor_key,
            list(spec.get("match_any") or []),
            int(spec.get("min_len", 4)),
            int(spec.get("max_len", 80)),
            max(evidence_limit * 3, 15),
        )
        prefer = list(spec.get("prefer_any") or [])
        statement, evidence_ids, ts = pick_statement(rows, prefer)
        if not statement or not evidence_ids:
            print(f"SKIP {fact_key}: no evidence")
            continue
        evidence_ids = evidence_ids[:evidence_limit]
        score = score_candidate(
            statement=statement,
            prefer_any=prefer,
            evidence_count_scanned=len(rows),
            evidence_ids=evidence_ids,
            create_time_epoch=ts,
        )
        status_key = f"{fact_key}||{statement}"
        status = prior_status.get(status_key, "pending")
        item = {
            "id": f"{actor_key}-{fact_key}",
            "op": "upsert",
            "fact_key": fact_key,
            "statement": statement,
            "evidence_message_ids": evidence_ids,
            "evidence_count_scanned": len(rows),
            "create_time_epoch": ts,
            "score": round(score, 2),
            "status": status,
            "source": "topic_heuristic_v1",
        }
        candidates.append(item)
        print(f"CANDIDATE {fact_key} score={score:.1f} status={status}: {statement!r}")

    candidates.sort(key=lambda c: (-float(c["score"]), str(c["fact_key"])))
    if limit > 0:
        candidates = candidates[:limit]

    payload = {
        "version": 1,
        "actor_key": actor_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topics_file": str(topics_path),
        "limit": limit,
        "candidates": candidates,
        "notes": "Set status to approved|rejected then run --mode apply",
    }
    if dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        write_pending(pending_path, payload)
        print(f"pending_file={pending_path}")
    print(f"extract_done actor_key={actor_key} candidates={len(candidates)}")
    return 0


def cmd_list(pending_path: Path) -> int:
    data = load_pending(pending_path)
    if not data:
        print(f"No pending file: {pending_path}")
        return 1
    print(f"pending_file={pending_path} actor_key={data.get('actor_key')}")
    for c in data.get("candidates") or []:
        print(
            f"  [{c.get('status')}] score={c.get('score')} "
            f"{c.get('fact_key')}: {c.get('statement')!r}"
        )
    return 0


def cmd_apply(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    pending_path: Path,
    replace_active: bool,
    dry_run: bool,
) -> int:
    data = load_pending(pending_path)
    if not data:
        print(f"No pending file: {pending_path}", file=sys.stderr)
        return 2
    if data.get("actor_key") and data["actor_key"] != actor_key:
        print(
            f"Pending actor_key={data.get('actor_key')!r} != --actor-key {actor_key!r}",
            file=sys.stderr,
        )
        return 2

    approved = [c for c in (data.get("candidates") or []) if c.get("status") == "approved"]
    if not approved:
        print("No candidates with status=approved", file=sys.stderr)
        return 1

    if replace_active and not dry_run:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM stg.persona_facts WHERE actor_key = %s",
                (actor_key,),
            )
        conn.commit()

    applied = 0
    for c in approved:
        fact_key = str(c.get("fact_key") or "").strip()
        statement = str(c.get("statement") or "").strip()
        evidence_ids = [int(x) for x in (c.get("evidence_message_ids") or [])]
        if not fact_key or not statement:
            continue
        conf = float(c.get("score") or 50.0) / 100.0
        conf = max(0.05, min(0.99, conf))
        source = str(c.get("source") or "pending_apply")
        print(f"APPLY {fact_key}: {statement!r} confidence={conf:.2f}")
        if not dry_run:
            upsert_fact(conn, actor_key, fact_key, statement, evidence_ids, conf, source)
            c["status"] = "applied"
        applied += 1

    if not dry_run:
        conn.commit()
        write_pending(pending_path, data)
    print(f"apply_done actor_key={actor_key} applied={applied} pending_file={pending_path}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()
    if not args.database_url and args.mode in {"extract", "apply"}:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    pending_path = (
        Path(args.pending_file)
        if args.pending_file
        else default_pending_path(repo_root, args.actor_key)
    )

    if args.mode == "list":
        return cmd_list(pending_path)

    topics, topics_path = resolve_topics(repo_root, args.actor_key, args.topics_file)

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if args.ensure_schema:
            ensure_schema(conn, repo_root)
        if args.mode == "extract":
            print(f"topics_file={topics_path} topics_n={len(topics)}")
            return cmd_extract(
                conn,
                actor_key=args.actor_key,
                topics=topics,
                topics_path=topics_path,
                pending_path=pending_path,
                evidence_limit=args.evidence_limit,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        if args.mode == "apply":
            return cmd_apply(
                conn,
                actor_key=args.actor_key,
                pending_path=pending_path,
                replace_active=args.replace_active,
                dry_run=args.dry_run,
            )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

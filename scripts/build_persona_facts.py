#!/usr/bin/env python3
"""Extract persona-fact candidates from stg.messages into a pending file; apply approved rows.

Workflow:
  extract -> data/private/facts_pending/<actor>.json  (scored, status=pending)
  set-status / approve / reject  (optional CLI helpers)
  apply   -> upsert approved rows into stg.persona_facts

Language-specific topic match strings live in data/private/fact_topics/, not in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

WS_RE = re.compile(r"\s+")


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
        choices=("extract", "apply", "list", "set-status"),
        default="extract",
        help="extract|apply|list|set-status",
    )
    p.add_argument("--ensure-schema", action="store_true")
    p.add_argument("--evidence-limit", type=int, default=5)
    p.add_argument("--limit", type=int, default=10, help="Top-N candidates to keep on extract")
    p.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Only use messages from the last N days (0=all)",
    )
    p.add_argument(
        "--include-active-keys",
        action="store_true",
        help="Keep candidates whose fact_key already exists as active (default: skip)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop candidates below this rule score on extract",
    )
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
        "--keys",
        default="",
        help="Comma-separated fact_key list for --mode set-status",
    )
    p.add_argument(
        "--status",
        default="",
        choices=("", "pending", "approved", "rejected"),
        help="Target status for --mode set-status",
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


def normalize_text(text: str) -> str:
    return WS_RE.sub("", (text or "").strip().lower())


def text_overlap_ratio(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # Char-bigram Jaccard for short CJK/latin lines.
    def bigrams(s: str) -> set[str]:
        if len(s) < 2:
            return {s}
        return {s[i : i + 2] for i in range(len(s) - 1)}

    ba, bb = bigrams(na), bigrams(nb)
    inter = len(ba & bb)
    union = len(ba | bb) or 1
    return inter / union


def fetch_active_facts(conn: psycopg.Connection, actor_key: str) -> list[dict]:
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT fact_key, statement
                FROM stg.persona_facts
                WHERE actor_key = %s AND status = 'active'
                """,
                (actor_key,),
            )
            return list(cur.fetchall())
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return []


def fetch_candidates(
    conn: psycopg.Connection,
    actor_key: str,
    terms: list[str],
    min_len: int,
    max_len: int,
    limit: int,
    min_epoch: int | None,
) -> list[dict]:
    clauses = ["text_content ILIKE %s" for _ in terms]
    params: list = [actor_key, min_len, max_len]
    time_sql = ""
    if min_epoch is not None:
        time_sql = " AND create_time_epoch >= %s "
        params.append(min_epoch)
    params.extend(f"%{t}%" for t in terms)
    params.append(limit)
    sql = f"""
        SELECT id, text_content, create_time_epoch
        FROM stg.messages
        WHERE actor_key = %s
          AND speaker_role = 'actor'
          AND has_semantic_text = TRUE
          AND char_length(text_content) BETWEEN %s AND %s
          {time_sql}
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
    now_epoch: int,
    active_facts: list[dict],
    fact_key: str,
) -> tuple[float, dict[str, float]]:
    """Explainable rule score in 0..100 with breakdown (phase 2)."""
    parts: dict[str, float] = {
        "base": 20.0,
        "evidence_volume": 0.0,
        "evidence_ids": 0.0,
        "prefer_hit": 0.0,
        "length": 0.0,
        "recency": 0.0,
        "question_penalty": 0.0,
        "laugh_penalty": 0.0,
        "active_key_penalty": 0.0,
        "active_text_penalty": 0.0,
    }
    # Volume: diminishing returns so many topics do not all pin at 100.
    parts["evidence_volume"] = min(25.0, 8.0 * (evidence_count_scanned**0.5))
    parts["evidence_ids"] = min(8.0, float(len(evidence_ids)) * 1.5)
    if any(p and p in statement for p in prefer_any):
        parts["prefer_hit"] = 12.0
    n = len(statement)
    if 8 <= n <= 36:
        parts["length"] = 12.0
    elif 6 <= n <= 50:
        parts["length"] = 8.0
    elif 51 <= n <= 80:
        parts["length"] = 3.0
    elif n > 100:
        parts["length"] = -6.0
    if create_time_epoch:
        age_days = max(0.0, (now_epoch - int(create_time_epoch)) / 86400.0)
        if age_days <= 30:
            parts["recency"] = 10.0
        elif age_days <= 180:
            parts["recency"] = 6.0
        elif age_days <= 365:
            parts["recency"] = 3.0
        else:
            parts["recency"] = 0.0
    if "？" in statement or "?" in statement:
        parts["question_penalty"] = -10.0
    laugh_n = statement.count("哈") + statement.count("haha".lower())
    if laugh_n >= 4:
        parts["laugh_penalty"] = -8.0
    elif laugh_n >= 2:
        parts["laugh_penalty"] = -3.0

    for af in active_facts:
        if af.get("fact_key") == fact_key:
            parts["active_key_penalty"] = -35.0
        ov = text_overlap_ratio(statement, str(af.get("statement") or ""))
        if ov >= 0.85:
            parts["active_text_penalty"] = min(parts["active_text_penalty"], -40.0)
        elif ov >= 0.65:
            parts["active_text_penalty"] = min(parts["active_text_penalty"], -20.0)

    score = sum(parts.values())
    score = max(0.0, min(100.0, score))
    return score, parts


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
        if "？" in text or "?" in text:
            local -= 0.5
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
    since_days: int,
    include_active_keys: bool,
    min_score: float,
    dry_run: bool,
) -> int:
    prior = load_pending(pending_path)
    prior_status: dict[str, str] = {}
    for c in prior.get("candidates") or []:
        key = f"{c.get('fact_key')}||{c.get('statement')}"
        st = c.get("status")
        if st in {"approved", "rejected", "applied"}:
            prior_status[key] = st

    active_facts = fetch_active_facts(conn, actor_key)
    active_keys = {str(a.get("fact_key") or "") for a in active_facts}
    now_epoch = int(time.time())
    min_epoch = now_epoch - since_days * 86400 if since_days > 0 else None

    skipped_active = 0
    skipped_low = 0
    candidates: list[dict[str, Any]] = []
    for spec in topics:
        fact_key = str(spec.get("fact_key") or "").strip()
        if not fact_key:
            continue
        if not include_active_keys and fact_key in active_keys:
            print(f"SKIP {fact_key}: already active")
            skipped_active += 1
            continue
        rows = fetch_candidates(
            conn,
            actor_key,
            list(spec.get("match_any") or []),
            int(spec.get("min_len", 4)),
            int(spec.get("max_len", 80)),
            max(evidence_limit * 3, 15),
            min_epoch,
        )
        prefer = list(spec.get("prefer_any") or [])
        statement, evidence_ids, ts = pick_statement(rows, prefer)
        if not statement or not evidence_ids:
            print(f"SKIP {fact_key}: no evidence")
            continue
        evidence_ids = evidence_ids[:evidence_limit]
        score, breakdown = score_candidate(
            statement=statement,
            prefer_any=prefer,
            evidence_count_scanned=len(rows),
            evidence_ids=evidence_ids,
            create_time_epoch=ts,
            now_epoch=now_epoch,
            active_facts=active_facts,
            fact_key=fact_key,
        )
        if score < min_score:
            print(f"SKIP {fact_key}: score={score:.1f} < min_score={min_score}")
            skipped_low += 1
            continue
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
            "score_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
            "status": status,
            "source": "topic_heuristic_v2",
        }
        candidates.append(item)
        print(
            f"CANDIDATE {fact_key} score={score:.1f} status={status}: {statement!r} "
            f"breakdown={item['score_breakdown']}"
        )

    candidates.sort(key=lambda c: (-float(c["score"]), str(c["fact_key"])))
    if limit > 0:
        candidates = candidates[:limit]

    if not candidates:
        print(
            f"extract_done actor_key={actor_key} candidates=0 "
            f"skipped_active={skipped_active} skipped_low={skipped_low} "
            "(kept existing pending file if any)"
        )
        if prior.get("candidates") and not dry_run:
            print(f"pending_file={pending_path} (unchanged)")
        return 0

    payload = {
        "version": 2,
        "actor_key": actor_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topics_file": str(topics_path),
        "limit": limit,
        "since_days": since_days,
        "min_score": min_score,
        "skipped_active_keys": skipped_active,
        "skipped_low_score": skipped_low,
        "active_fact_keys": sorted(active_keys),
        "candidates": candidates,
        "notes": (
            "Set status via --mode set-status --keys a,b --status approved "
            "then --mode apply"
        ),
    }
    if dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        write_pending(pending_path, payload)
        print(f"pending_file={pending_path}")
    print(
        f"extract_done actor_key={actor_key} candidates={len(candidates)} "
        f"skipped_active={skipped_active} skipped_low={skipped_low}"
    )
    return 0


def cmd_list(pending_path: Path) -> int:
    data = load_pending(pending_path)
    if not data:
        print(f"No pending file: {pending_path}")
        return 1
    print(f"pending_file={pending_path} actor_key={data.get('actor_key')}")
    for c in data.get("candidates") or []:
        bd = c.get("score_breakdown") or {}
        print(
            f"  [{c.get('status')}] score={c.get('score')} "
            f"{c.get('fact_key')}: {c.get('statement')!r}"
        )
        if bd:
            print(f"           breakdown={bd}")
    return 0


def cmd_set_status(pending_path: Path, keys_csv: str, status: str, dry_run: bool) -> int:
    if not status:
        print("--status is required for set-status", file=sys.stderr)
        return 2
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if not keys:
        print("--keys is required (comma-separated fact_key)", file=sys.stderr)
        return 2
    data = load_pending(pending_path)
    if not data:
        print(f"No pending file: {pending_path}", file=sys.stderr)
        return 2
    keyset = set(keys)
    n = 0
    for c in data.get("candidates") or []:
        if c.get("fact_key") in keyset:
            print(f"SET {c.get('fact_key')}: {c.get('status')} -> {status}")
            if not dry_run:
                c["status"] = status
            n += 1
    if n == 0:
        print(f"No matching keys in pending: {sorted(keyset)}", file=sys.stderr)
        return 1
    if not dry_run:
        write_pending(pending_path, data)
    print(f"set_status_done updated={n} status={status}")
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

    pending_path = (
        Path(args.pending_file)
        if args.pending_file
        else default_pending_path(repo_root, args.actor_key)
    )

    if args.mode == "list":
        return cmd_list(pending_path)
    if args.mode == "set-status":
        return cmd_set_status(pending_path, args.keys, args.status, args.dry_run)

    if not args.database_url:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

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
                since_days=args.since_days,
                include_active_keys=args.include_active_keys,
                min_score=args.min_score,
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

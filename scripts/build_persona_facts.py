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
import random
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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from persona_store import (  # noqa: E402
    add_fact_block,
    ensure_governance_schema,
    fact_candidate_to_dict,
    list_fact_blocks,
    list_fact_candidates,
    list_topic_specs,
    load_facts_pending_view,
    remove_fact_block,
    replace_fact_candidates,
    set_fact_blocks,
    upsert_fact_candidate,
)
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
        choices=("extract", "apply", "list", "set-status", "db-list", "db-set-status", "db-delete"),
        default="extract",
        help="extract|apply|list|set-status|db-list|db-set-status|db-delete",
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
        "--include-rejected-keys",
        action="store_true",
        help="Re-extract fact_keys previously rejected in pending (default: skip)",
    )
    p.add_argument(
        "--diversify",
        action="store_true",
        help="Pick statement randomly among top local matches (default: deterministic best)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used only with --diversify (0=time-based)",
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
        help="Optional topic specs JSON override for extract (default: stg.persona_topic_specs)",
    )
    p.add_argument(
        "--pending-file",
        default=None,
        help="Deprecated; fact pending now lives in Postgres",
    )
    p.add_argument(
        "--keys",
        default="",
        help="Comma-separated fact_key list for --mode set-status",
    )
    p.add_argument(
        "--status",
        default="",
        choices=("", "pending", "approved", "rejected", "active", "inactive"),
        help="Target status for set-status / db-set-status",
    )
    p.add_argument(
        "--replace-active",
        action="store_true",
        help="On apply: delete existing active facts for actor before upsert (dangerous)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_topics(
    conn: psycopg.Connection,
    actor_key: str,
    topics_file: str | None = None,
) -> list[dict]:
    if topics_file:
        path = Path(topics_file)
        if not path.is_file():
            raise SystemExit(f"topics file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        raw = data.get("topics") if isinstance(data, dict) else data
        if not isinstance(raw, list) or not raw:
            raise SystemExit(f"No topics found in {path}")
        specs = [s for s in raw if isinstance(s, dict) and s.get("enabled", True)]
        if not specs:
            raise SystemExit(f"No enabled topics found in {path}")
        source_label = str(path)
    else:
        specs = list_topic_specs(conn, actor_key, enabled_only=True)
        source_label = "stg.persona_topic_specs"
        if not specs:
            raise SystemExit(
                f"No enabled topics in {source_label} for actor_key={actor_key!r}. "
                "Mine/approve/apply topics first, or pass --topics-file."
            )
    return [
        {
            "fact_key": s.get("fact_key"),
            "kind": s.get("kind") or "preference",
            "match_any": list(s.get("match_any") or []),
            "prefer_any": list(s.get("prefer_any") or []),
            "min_len": int(s.get("min_len") or 4),
            "max_len": int(s.get("max_len") or 80),
            "_source": source_label,
        }
        for s in specs
    ]


def default_pending_path(repo_root: Path, actor_key: str) -> Path:
    # Legacy path kept for migrate/import only; runtime uses Postgres.
    return repo_root / "data" / "private" / "facts_pending" / f"{actor_key}.json"


def ensure_schema(conn: psycopg.Connection, repo_root: Path) -> None:
    ddl = (repo_root / "sql" / "004_stg_persona_facts.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    ensure_governance_schema(conn, repo_root)
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
    return fetch_facts(conn, actor_key, status="active")


def fetch_facts(
    conn: psycopg.Connection,
    actor_key: str,
    status: str | None = None,
) -> list[dict]:
    with conn.cursor() as cur:
        try:
            if status:
                cur.execute(
                    """
                    SELECT id, fact_key, statement, confidence, status, source, updated_at
                    FROM stg.persona_facts
                    WHERE actor_key = %s AND status = %s
                    ORDER BY fact_key
                    """,
                    (actor_key, status),
                )
            else:
                cur.execute(
                    """
                    SELECT id, fact_key, statement, confidence, status, source, updated_at
                    FROM stg.persona_facts
                    WHERE actor_key = %s
                    ORDER BY status, fact_key
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


def set_fact_status(
    conn: psycopg.Connection,
    actor_key: str,
    fact_key: str,
    status: str,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE stg.persona_facts
            SET status = %s, updated_at = now()
            WHERE actor_key = %s AND fact_key = %s
            """,
            (status, actor_key, fact_key),
        )
        n = cur.rowcount
    conn.commit()
    return n > 0


def delete_fact(conn: psycopg.Connection, actor_key: str, fact_key: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM stg.persona_facts
            WHERE actor_key = %s AND fact_key = %s
            """,
            (actor_key, fact_key),
        )
        n = cur.rowcount
    conn.commit()
    return n > 0


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


def pick_statement(
    rows: list[dict],
    prefer_any: list[str],
    *,
    diversify: bool = False,
    rng: random.Random | None = None,
) -> tuple[str, list[int], int | None]:
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
    if diversify and rng is not None and len(scored) > 1:
        top_n = min(3, len(scored))
        best = rng.choice([r for _, r in scored[:top_n]])
    else:
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
    evidence_limit: int,
    limit: int,
    since_days: int,
    include_active_keys: bool,
    include_rejected_keys: bool,
    min_score: float,
    diversify: bool,
    seed: int,
    dry_run: bool,
) -> int:
    prior = load_facts_pending_view(conn, actor_key)
    prior_status: dict[str, str] = {}
    blocked_keys: set[str] = set(str(x) for x in (prior.get("blocked_fact_keys") or []))
    for c in prior.get("candidates") or []:
        key = f"{c.get('fact_key')}||{c.get('statement')}"
        st = c.get("status")
        fk = str(c.get("fact_key") or "")
        if st in {"approved", "rejected", "applied"}:
            prior_status[key] = st
        if st == "rejected" and fk:
            blocked_keys.add(fk)
    blocked_keys |= list_fact_blocks(conn, actor_key)

    active_facts = fetch_active_facts(conn, actor_key)
    active_keys = {str(a.get("fact_key") or "") for a in active_facts}
    now_epoch = int(time.time())
    min_epoch = now_epoch - since_days * 86400 if since_days > 0 else None
    rng = random.Random(seed if seed else now_epoch)

    skipped_active = 0
    skipped_rejected = 0
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
        if not include_rejected_keys and fact_key in blocked_keys:
            print(f"SKIP {fact_key}: previously rejected (blocked)")
            skipped_rejected += 1
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
        statement, evidence_ids, ts = pick_statement(
            rows, prefer, diversify=diversify, rng=rng
        )
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
        if status == "rejected" and not include_rejected_keys:
            print(f"SKIP {fact_key}: exact statement was rejected")
            skipped_rejected += 1
            blocked_keys.add(fact_key)
            continue
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

    run_meta = {
        "limit": limit,
        "since_days": since_days,
        "min_score": min_score,
        "diversify": diversify,
        "skipped_active_keys": skipped_active,
        "skipped_rejected_keys": skipped_rejected,
        "skipped_low_score": skipped_low,
        "active_fact_keys": sorted(active_keys),
    }

    if not candidates:
        if not dry_run:
            set_fact_blocks(conn, actor_key, blocked_keys)
            conn.commit()
        print(
            f"extract_done actor_key={actor_key} candidates=0 "
            f"skipped_active={skipped_active} skipped_rejected={skipped_rejected} "
            f"skipped_low={skipped_low} storage=postgres"
        )
        return 0

    if dry_run:
        print(json.dumps({"candidates": candidates, "meta": run_meta}, ensure_ascii=False, indent=2))
    else:
        set_fact_blocks(conn, actor_key, blocked_keys)
        replace_fact_candidates(conn, actor_key, candidates, run_meta=run_meta)
        conn.commit()
        print("storage=postgres table=stg.persona_fact_candidates")
    print(
        f"extract_done actor_key={actor_key} candidates={len(candidates)} "
        f"skipped_active={skipped_active} skipped_rejected={skipped_rejected} "
        f"skipped_low={skipped_low}"
    )
    return 0



def cmd_list(conn: psycopg.Connection, actor_key: str) -> int:
    data = load_facts_pending_view(conn, actor_key)
    cands = data.get("candidates") or []
    if not cands:
        print(f"No pending fact candidates for actor_key={actor_key}")
        return 1
    print(f"storage=postgres actor_key={actor_key}")
    for c in cands:
        bd = c.get("score_breakdown") or {}
        print(
            f"  [{c.get('status')}] score={c.get('score')} "
            f"{c.get('fact_key')}: {c.get('statement')!r}"
        )
        if bd:
            print(f"           breakdown={bd}")
    return 0


def cmd_set_status(
    conn: psycopg.Connection, actor_key: str, keys_csv: str, status: str, dry_run: bool
) -> int:
    if not status:
        print("--status is required for set-status", file=sys.stderr)
        return 2
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if not keys:
        print("--keys is required (comma-separated fact_key)", file=sys.stderr)
        return 2
    keyset = set(keys)
    n = 0
    for row in list_fact_candidates(conn, actor_key):
        fk = str(row.get("fact_key") or "")
        if fk not in keyset:
            continue
        c = fact_candidate_to_dict(row)
        print(f"SET {fk}: {c.get('status')} -> {status}")
        if not dry_run:
            c["status"] = status
            upsert_fact_candidate(conn, actor_key, c)
            if status == "rejected":
                add_fact_block(conn, actor_key, fk)
            elif status in {"pending", "approved"}:
                remove_fact_block(conn, actor_key, fk)
        n += 1
    if n == 0:
        print(f"No matching keys in pending: {sorted(keyset)}", file=sys.stderr)
        return 1
    if not dry_run:
        conn.commit()
    print(f"set_status_done updated={n} status={status}")
    return 0


def cmd_apply(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    replace_active: bool,
    dry_run: bool,
) -> int:
    data = load_facts_pending_view(conn, actor_key)
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
            upsert_fact_candidate(conn, actor_key, c)
        applied += 1

    if not dry_run:
        conn.commit()
    print(f"apply_done actor_key={actor_key} applied={applied} storage=postgres")
    return 0


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

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if args.ensure_schema or args.mode in {
            "extract",
            "apply",
            "list",
            "set-status",
            "db-list",
            "db-set-status",
            "db-delete",
        }:
            ensure_schema(conn, repo_root)

        if args.mode == "list":
            return cmd_list(conn, args.actor_key)
        if args.mode == "set-status":
            return cmd_set_status(
                conn, args.actor_key, args.keys, args.status, args.dry_run
            )

        if args.mode == "db-list":
            rows = fetch_facts(conn, args.actor_key, status=None)
            for r in rows:
                print(
                    f"[{r.get('status')}] {r.get('fact_key')}: "
                    f"{(r.get('statement') or '')[:80]!r} conf={r.get('confidence')}"
                )
            print(f"db_list_done n={len(rows)}")
            return 0

        if args.mode == "db-set-status":
            if args.status not in {"active", "inactive"}:
                print("db-set-status requires --status active|inactive", file=sys.stderr)
                return 2
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
            if not keys:
                print("--keys required", file=sys.stderr)
                return 2
            n = 0
            for k in keys:
                if args.dry_run:
                    print(f"DRY {k} -> {args.status}")
                elif set_fact_status(conn, args.actor_key, k, args.status):
                    print(f"SET DB {k} -> {args.status}")
                    n += 1
                else:
                    print(f"MISS {k}", file=sys.stderr)
            print(f"db_set_status_done updated={n}")
            return 0 if n or args.dry_run else 1

        if args.mode == "db-delete":
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
            if not keys:
                print("--keys required", file=sys.stderr)
                return 2
            n = 0
            for k in keys:
                if args.dry_run:
                    print(f"DRY DELETE {k}")
                elif delete_fact(conn, args.actor_key, k):
                    print(f"DELETE DB {k}")
                    n += 1
                else:
                    print(f"MISS {k}", file=sys.stderr)
            print(f"db_delete_done deleted={n}")
            return 0 if n or args.dry_run else 1

        if args.mode == "extract":
            topics = resolve_topics(conn, args.actor_key, args.topics_file)
            src = (topics[0].get("_source") if topics else "none")
            print(f"topics_n={len(topics)} topics_source={src}")
            return cmd_extract(
                conn,
                actor_key=args.actor_key,
                topics=topics,
                evidence_limit=args.evidence_limit,
                limit=args.limit,
                since_days=args.since_days,
                include_active_keys=args.include_active_keys,
                include_rejected_keys=args.include_rejected_keys,
                min_score=args.min_score,
                diversify=args.diversify,
                seed=args.seed,
                dry_run=args.dry_run,
            )
        if args.mode == "apply":
            return cmd_apply(
                conn,
                actor_key=args.actor_key,
                replace_active=args.replace_active,
                dry_run=args.dry_run,
            )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

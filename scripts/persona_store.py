#!/usr/bin/env python3
"""Postgres helpers for persona governance (topics, pending queues, cards).

Replaces private JSON stores for runnable governance data.
Locale / LoRA / eval fixtures stay on disk by design.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def ensure_governance_schema(conn: psycopg.Connection, repo_root: Path) -> None:
    ddl = (repo_root / "sql" / "005_persona_governance.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ----- topic specs (official) -----


def list_topic_specs(
    conn: psycopg.Connection,
    actor_key: str,
    *,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    sql = """
        SELECT actor_key, fact_key, kind, match_any, prefer_any, min_len, max_len,
               enabled, source, created_at, updated_at
        FROM stg.persona_topic_specs
        WHERE actor_key = %s
    """
    if enabled_only:
        sql += " AND enabled = TRUE"
    sql += " ORDER BY fact_key"
    with conn.cursor() as cur:
        cur.execute(sql, (actor_key,))
        return list(cur.fetchall())


def upsert_topic_spec(conn: psycopg.Connection, actor_key: str, spec: dict[str, Any]) -> None:
    fk = str(spec.get("fact_key") or "").strip()
    if not fk:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_topic_specs (
                actor_key, fact_key, kind, match_any, prefer_any,
                min_len, max_len, enabled, source, updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
            ON CONFLICT (actor_key, fact_key) DO UPDATE SET
                kind = EXCLUDED.kind,
                match_any = EXCLUDED.match_any,
                prefer_any = EXCLUDED.prefer_any,
                min_len = EXCLUDED.min_len,
                max_len = EXCLUDED.max_len,
                enabled = EXCLUDED.enabled,
                source = EXCLUDED.source,
                updated_at = now()
            """,
            (
                actor_key,
                fk,
                str(spec.get("kind") or "preference"),
                list(spec.get("match_any") or []),
                list(spec.get("prefer_any") or []),
                int(spec.get("min_len") or 4),
                int(spec.get("max_len") or 80),
                bool(spec.get("enabled", True)),
                str(spec.get("source") or "manual"),
            ),
        )


def set_topic_spec_enabled(
    conn: psycopg.Connection, actor_key: str, fact_key: str, enabled: bool
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE stg.persona_topic_specs
            SET enabled = %s, updated_at = now()
            WHERE actor_key = %s AND fact_key = %s
            """,
            (enabled, actor_key, fact_key),
        )
        return cur.rowcount > 0


def replace_topic_specs(
    conn: psycopg.Connection, actor_key: str, specs: list[dict[str, Any]]
) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM stg.persona_topic_specs WHERE actor_key = %s", (actor_key,))
    for spec in specs:
        upsert_topic_spec(conn, actor_key, spec)


# ----- topic blocks / candidates -----


def list_topic_blocks(conn: psycopg.Connection, actor_key: str) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    phrases: set[str] = set()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT block_kind, value FROM stg.persona_topic_blocks
            WHERE actor_key = %s
            """,
            (actor_key,),
        )
        for r in cur.fetchall():
            if r["block_kind"] == "fact_key":
                keys.add(str(r["value"]))
            else:
                phrases.add(str(r["value"]))
    return keys, phrases


def set_topic_blocks(
    conn: psycopg.Connection,
    actor_key: str,
    *,
    fact_keys: set[str],
    phrases: set[str],
) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM stg.persona_topic_blocks WHERE actor_key = %s", (actor_key,))
        for k in sorted(fact_keys):
            cur.execute(
                """
                INSERT INTO stg.persona_topic_blocks (actor_key, block_kind, value)
                VALUES (%s, 'fact_key', %s) ON CONFLICT DO NOTHING
                """,
                (actor_key, k),
            )
        for p in sorted(phrases):
            if not p:
                continue
            cur.execute(
                """
                INSERT INTO stg.persona_topic_blocks (actor_key, block_kind, value)
                VALUES (%s, 'phrase', %s) ON CONFLICT DO NOTHING
                """,
                (actor_key, p),
            )


def add_topic_block(
    conn: psycopg.Connection, actor_key: str, block_kind: str, value: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_topic_blocks (actor_key, block_kind, value)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
            """,
            (actor_key, block_kind, value),
        )


def remove_topic_block(
    conn: psycopg.Connection, actor_key: str, block_kind: str, value: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM stg.persona_topic_blocks
            WHERE actor_key = %s AND block_kind = %s AND value = %s
            """,
            (actor_key, block_kind, value),
        )


def list_topic_candidates(conn: psycopg.Connection, actor_key: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM stg.persona_topic_candidates
            WHERE actor_key = %s
            ORDER BY score DESC NULLS LAST, hit_count DESC NULLS LAST, fact_key
            """,
            (actor_key,),
        )
        return list(cur.fetchall())


def get_topic_extract_run(conn: psycopg.Connection, actor_key: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT generated_at, meta FROM stg.persona_topic_extract_runs
            WHERE actor_key = %s
            """,
            (actor_key,),
        )
        row = cur.fetchone()
    if not row:
        return {}
    meta = dict(row.get("meta") or {})
    meta["generated_at"] = row.get("generated_at").isoformat() if row.get("generated_at") else None
    return meta


def replace_topic_candidates(
    conn: psycopg.Connection,
    actor_key: str,
    candidates: list[dict[str, Any]],
    *,
    run_meta: dict[str, Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM stg.persona_topic_candidates WHERE actor_key = %s", (actor_key,)
        )
        for c in candidates:
            cur.execute(
                """
                INSERT INTO stg.persona_topic_candidates (
                    actor_key, fact_key, status, kind, phrase, match_any, prefer_any,
                    min_len, max_len, hit_count, score, latest_epoch,
                    example_message_ids, example_excerpts, source, enabled, meta, updated_at
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()
                )
                """,
                (
                    actor_key,
                    str(c.get("fact_key") or ""),
                    str(c.get("status") or "pending"),
                    str(c.get("kind") or "preference"),
                    c.get("phrase"),
                    list(c.get("match_any") or []),
                    list(c.get("prefer_any") or []),
                    int(c.get("min_len") or 4),
                    int(c.get("max_len") or 80),
                    c.get("hit_count"),
                    c.get("score"),
                    c.get("latest_epoch"),
                    [int(x) for x in (c.get("example_message_ids") or [])],
                    [str(x) for x in (c.get("example_excerpts") or [])],
                    str(c.get("source") or "topic_mine_v1"),
                    bool(c.get("enabled", True)),
                    Jsonb(c.get("meta") or {}),
                ),
            )
        if run_meta is not None:
            cur.execute(
                """
                INSERT INTO stg.persona_topic_extract_runs (actor_key, generated_at, meta, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (actor_key) DO UPDATE SET
                    generated_at = EXCLUDED.generated_at,
                    meta = EXCLUDED.meta,
                    updated_at = now()
                """,
                (actor_key, _now(), Jsonb(run_meta)),
            )


def delete_topic_candidate(
    conn: psycopg.Connection, actor_key: str, fact_key: str
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM stg.persona_topic_candidates
            WHERE actor_key = %s AND fact_key = %s
            """,
            (actor_key, fact_key),
        )
        return cur.rowcount > 0


def upsert_topic_candidate(conn: psycopg.Connection, actor_key: str, c: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_topic_candidates (
                actor_key, fact_key, status, kind, phrase, match_any, prefer_any,
                min_len, max_len, hit_count, score, latest_epoch,
                example_message_ids, example_excerpts, source, enabled, meta, updated_at
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()
            )
            ON CONFLICT (actor_key, fact_key) DO UPDATE SET
                status = EXCLUDED.status,
                kind = EXCLUDED.kind,
                phrase = EXCLUDED.phrase,
                match_any = EXCLUDED.match_any,
                prefer_any = EXCLUDED.prefer_any,
                min_len = EXCLUDED.min_len,
                max_len = EXCLUDED.max_len,
                hit_count = EXCLUDED.hit_count,
                score = EXCLUDED.score,
                latest_epoch = EXCLUDED.latest_epoch,
                example_message_ids = EXCLUDED.example_message_ids,
                example_excerpts = EXCLUDED.example_excerpts,
                source = EXCLUDED.source,
                enabled = EXCLUDED.enabled,
                meta = EXCLUDED.meta,
                updated_at = now()
            """,
            (
                actor_key,
                str(c.get("fact_key") or ""),
                str(c.get("status") or "pending"),
                str(c.get("kind") or "preference"),
                c.get("phrase"),
                list(c.get("match_any") or []),
                list(c.get("prefer_any") or []),
                int(c.get("min_len") or 4),
                int(c.get("max_len") or 80),
                c.get("hit_count"),
                c.get("score"),
                c.get("latest_epoch"),
                [int(x) for x in (c.get("example_message_ids") or [])],
                [str(x) for x in (c.get("example_excerpts") or [])],
                str(c.get("source") or "topic_mine_v1"),
                bool(c.get("enabled", True)),
                Jsonb(c.get("meta") or {}),
            ),
        )


def topic_candidate_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{row.get('actor_key')}-{row.get('fact_key')}",
        "op": "upsert_topic",
        "fact_key": row.get("fact_key"),
        "kind": row.get("kind"),
        "phrase": row.get("phrase"),
        "match_any": list(row.get("match_any") or []),
        "prefer_any": list(row.get("prefer_any") or []),
        "min_len": row.get("min_len"),
        "max_len": row.get("max_len"),
        "hit_count": row.get("hit_count"),
        "score": row.get("score"),
        "latest_epoch": row.get("latest_epoch"),
        "example_message_ids": list(row.get("example_message_ids") or []),
        "example_excerpts": list(row.get("example_excerpts") or []),
        "status": row.get("status"),
        "source": row.get("source"),
        "enabled": row.get("enabled", True),
    }


def load_topics_pending_view(conn: psycopg.Connection, actor_key: str) -> dict[str, Any]:
    blocked_keys, blocked_phrases = list_topic_blocks(conn, actor_key)
    cands = [topic_candidate_to_dict(r) for r in list_topic_candidates(conn, actor_key)]
    meta = get_topic_extract_run(conn, actor_key)
    return {
        "version": 1,
        "actor_key": actor_key,
        "generated_at": meta.get("generated_at"),
        "messages_scanned": meta.get("messages_scanned"),
        "skipped_existing": meta.get("skipped_existing"),
        "skipped_rejected": meta.get("skipped_rejected"),
        "skipped_low_score": meta.get("skipped_low_score"),
        "official_topic_keys": meta.get("official_topic_keys") or [],
        "blocked_fact_keys": sorted(blocked_keys),
        "blocked_phrases": sorted(blocked_phrases),
        "candidates": cands,
        "storage": "postgres",
    }


# ----- fact candidates / blocks -----


def list_fact_blocks(conn: psycopg.Connection, actor_key: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fact_key FROM stg.persona_fact_blocks WHERE actor_key = %s",
            (actor_key,),
        )
        return {str(r["fact_key"]) for r in cur.fetchall()}


def set_fact_blocks(conn: psycopg.Connection, actor_key: str, keys: set[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM stg.persona_fact_blocks WHERE actor_key = %s", (actor_key,))
        for k in sorted(keys):
            cur.execute(
                """
                INSERT INTO stg.persona_fact_blocks (actor_key, fact_key)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
                """,
                (actor_key, k),
            )


def add_fact_block(conn: psycopg.Connection, actor_key: str, fact_key: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_fact_blocks (actor_key, fact_key)
            VALUES (%s, %s) ON CONFLICT DO NOTHING
            """,
            (actor_key, fact_key),
        )


def remove_fact_block(conn: psycopg.Connection, actor_key: str, fact_key: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM stg.persona_fact_blocks
            WHERE actor_key = %s AND fact_key = %s
            """,
            (actor_key, fact_key),
        )


def list_fact_candidates(conn: psycopg.Connection, actor_key: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM stg.persona_fact_candidates
            WHERE actor_key = %s
            ORDER BY score DESC NULLS LAST, fact_key
            """,
            (actor_key,),
        )
        return list(cur.fetchall())


def get_fact_extract_run(conn: psycopg.Connection, actor_key: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT generated_at, meta FROM stg.persona_fact_extract_runs
            WHERE actor_key = %s
            """,
            (actor_key,),
        )
        row = cur.fetchone()
    if not row:
        return {}
    meta = dict(row.get("meta") or {})
    meta["generated_at"] = row.get("generated_at").isoformat() if row.get("generated_at") else None
    return meta


def replace_fact_candidates(
    conn: psycopg.Connection,
    actor_key: str,
    candidates: list[dict[str, Any]],
    *,
    run_meta: dict[str, Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM stg.persona_fact_candidates WHERE actor_key = %s", (actor_key,)
        )
        for c in candidates:
            cur.execute(
                """
                INSERT INTO stg.persona_fact_candidates (
                    actor_key, fact_key, status, statement, evidence_message_ids,
                    evidence_count_scanned, create_time_epoch, score, score_breakdown,
                    source, op, meta, updated_at
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()
                )
                """,
                (
                    actor_key,
                    str(c.get("fact_key") or ""),
                    str(c.get("status") or "pending"),
                    str(c.get("statement") or ""),
                    [int(x) for x in (c.get("evidence_message_ids") or [])],
                    c.get("evidence_count_scanned"),
                    c.get("create_time_epoch"),
                    c.get("score"),
                    Jsonb(c.get("score_breakdown") or {}),
                    c.get("source"),
                    str(c.get("op") or "upsert"),
                    Jsonb({k: v for k, v in c.items() if k not in {
                        "fact_key", "status", "statement", "evidence_message_ids",
                        "evidence_count_scanned", "create_time_epoch", "score",
                        "score_breakdown", "source", "op", "id",
                    }}),
                ),
            )
        if run_meta is not None:
            cur.execute(
                """
                INSERT INTO stg.persona_fact_extract_runs (actor_key, generated_at, meta, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (actor_key) DO UPDATE SET
                    generated_at = EXCLUDED.generated_at,
                    meta = EXCLUDED.meta,
                    updated_at = now()
                """,
                (actor_key, _now(), Jsonb(run_meta)),
            )


def upsert_fact_candidate(conn: psycopg.Connection, actor_key: str, c: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_fact_candidates (
                actor_key, fact_key, status, statement, evidence_message_ids,
                evidence_count_scanned, create_time_epoch, score, score_breakdown,
                source, op, meta, updated_at
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()
            )
            ON CONFLICT (actor_key, fact_key) DO UPDATE SET
                status = EXCLUDED.status,
                statement = EXCLUDED.statement,
                evidence_message_ids = EXCLUDED.evidence_message_ids,
                evidence_count_scanned = EXCLUDED.evidence_count_scanned,
                create_time_epoch = EXCLUDED.create_time_epoch,
                score = EXCLUDED.score,
                score_breakdown = EXCLUDED.score_breakdown,
                source = EXCLUDED.source,
                op = EXCLUDED.op,
                meta = EXCLUDED.meta,
                updated_at = now()
            """,
            (
                actor_key,
                str(c.get("fact_key") or ""),
                str(c.get("status") or "pending"),
                str(c.get("statement") or ""),
                [int(x) for x in (c.get("evidence_message_ids") or [])],
                c.get("evidence_count_scanned"),
                c.get("create_time_epoch"),
                c.get("score"),
                Jsonb(c.get("score_breakdown") or {}),
                c.get("source"),
                str(c.get("op") or "upsert"),
                Jsonb(c.get("meta") or {}),
            ),
        )


def fact_candidate_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "id": f"{row.get('actor_key')}-{row.get('fact_key')}",
        "op": row.get("op") or "upsert",
        "fact_key": row.get("fact_key"),
        "statement": row.get("statement"),
        "evidence_message_ids": list(row.get("evidence_message_ids") or []),
        "evidence_count_scanned": row.get("evidence_count_scanned"),
        "create_time_epoch": row.get("create_time_epoch"),
        "score": row.get("score"),
        "score_breakdown": dict(row.get("score_breakdown") or {}),
        "status": row.get("status"),
        "source": row.get("source"),
    }
    meta = row.get("meta") or {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            out.setdefault(k, v)
    return out


def load_facts_pending_view(conn: psycopg.Connection, actor_key: str) -> dict[str, Any]:
    blocked = list_fact_blocks(conn, actor_key)
    cands = [fact_candidate_to_dict(r) for r in list_fact_candidates(conn, actor_key)]
    meta = get_fact_extract_run(conn, actor_key)
    return {
        "version": 2,
        "actor_key": actor_key,
        "generated_at": meta.get("generated_at"),
        "limit": meta.get("limit"),
        "since_days": meta.get("since_days"),
        "min_score": meta.get("min_score"),
        "diversify": meta.get("diversify"),
        "skipped_active_keys": meta.get("skipped_active_keys"),
        "skipped_rejected_keys": meta.get("skipped_rejected_keys"),
        "skipped_low_score": meta.get("skipped_low_score"),
        "active_fact_keys": meta.get("active_fact_keys") or [],
        "blocked_fact_keys": sorted(blocked),
        "candidates": cands,
        "storage": "postgres",
    }


# ----- persona cards -----


def load_persona_card_db(
    conn: psycopg.Connection, actor_key: str
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM stg.persona_cards WHERE actor_key = %s",
            (actor_key,),
        )
        row = cur.fetchone()
    if not row:
        return None
    card = dict(row.get("card") or {})
    # Prefer normalized columns when present.
    if row.get("display_name") is not None:
        card["display_name"] = row["display_name"]
    if row.get("relationship") is not None:
        card["relationship"] = row["relationship"]
    if row.get("voice_notes") is not None:
        card["voice_notes"] = row["voice_notes"]
    if row.get("known_facts") is not None:
        card["known_facts"] = row["known_facts"]
    if row.get("boundaries") is not None:
        card["boundaries"] = row["boundaries"]
    if row.get("extra_rules") is not None:
        card["extra_rules"] = row["extra_rules"]
    return card


def upsert_persona_card(
    conn: psycopg.Connection, actor_key: str, card: dict[str, Any]
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stg.persona_cards (
                actor_key, display_name, relationship, voice_notes, known_facts,
                boundaries, extra_rules, card, updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
            ON CONFLICT (actor_key) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                relationship = EXCLUDED.relationship,
                voice_notes = EXCLUDED.voice_notes,
                known_facts = EXCLUDED.known_facts,
                boundaries = EXCLUDED.boundaries,
                extra_rules = EXCLUDED.extra_rules,
                card = EXCLUDED.card,
                updated_at = now()
            """,
            (
                actor_key,
                card.get("display_name"),
                card.get("relationship"),
                Jsonb(card.get("voice_notes") or []),
                Jsonb(card.get("known_facts") or []),
                Jsonb(card.get("boundaries") or []),
                Jsonb(card.get("extra_rules") or []),
                Jsonb(card),
            ),
        )


def list_persona_card_actors(conn: psycopg.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT actor_key FROM stg.persona_cards ORDER BY actor_key")
        return [str(r["actor_key"]) for r in cur.fetchall()]


# ----- import from legacy JSON -----


def import_topic_specs_json(conn: psycopg.Connection, path: Path, actor_key: str) -> int:
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    topics = data.get("topics") if isinstance(data, dict) else data
    n = 0
    for spec in topics or []:
        if not isinstance(spec, dict):
            continue
        spec = dict(spec)
        spec.setdefault("source", "json_import")
        upsert_topic_spec(conn, actor_key, spec)
        n += 1
    return n


def import_topics_pending_json(conn: psycopg.Connection, path: Path, actor_key: str) -> int:
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return 0
    blocked_keys = set(str(x) for x in (data.get("blocked_fact_keys") or []))
    blocked_phrases = set(str(x) for x in (data.get("blocked_phrases") or []))
    set_topic_blocks(conn, actor_key, fact_keys=blocked_keys, phrases=blocked_phrases)
    cands = list(data.get("candidates") or [])
    meta = {
        "messages_scanned": data.get("messages_scanned"),
        "skipped_existing": data.get("skipped_existing"),
        "skipped_rejected": data.get("skipped_rejected"),
        "skipped_low_score": data.get("skipped_low_score"),
        "official_topic_keys": data.get("official_topic_keys") or [],
        "imported_from": str(path),
    }
    replace_topic_candidates(conn, actor_key, cands, run_meta=meta)
    return len(cands)


def import_facts_pending_json(conn: psycopg.Connection, path: Path, actor_key: str) -> int:
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return 0
    set_fact_blocks(conn, actor_key, set(str(x) for x in (data.get("blocked_fact_keys") or [])))
    cands = list(data.get("candidates") or [])
    meta = {
        "limit": data.get("limit"),
        "since_days": data.get("since_days"),
        "min_score": data.get("min_score"),
        "diversify": data.get("diversify"),
        "skipped_active_keys": data.get("skipped_active_keys"),
        "skipped_rejected_keys": data.get("skipped_rejected_keys"),
        "skipped_low_score": data.get("skipped_low_score"),
        "active_fact_keys": data.get("active_fact_keys") or [],
        "imported_from": str(path),
    }
    replace_fact_candidates(conn, actor_key, cands, run_meta=meta)
    return len(cands)


def import_persona_card_json(conn: psycopg.Connection, path: Path, actor_key: str) -> bool:
    if not path.is_file():
        return False
    card = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(card, dict):
        return False
    # Skip legacy extract dumps mistakenly stored under personas/
    if path.name.endswith("_facts.extracted.json"):
        return False
    upsert_persona_card(conn, actor_key, card)
    return True


def import_private_tree(conn: psycopg.Connection, repo_root: Path) -> dict[str, int]:
    private = repo_root / "data" / "private"
    stats = {
        "topic_specs": 0,
        "topic_candidates": 0,
        "fact_candidates": 0,
        "persona_cards": 0,
    }
    for path in sorted((private / "fact_topics").glob("*.json")) if (private / "fact_topics").is_dir() else []:
        stats["topic_specs"] += import_topic_specs_json(conn, path, path.stem)
    for path in sorted((private / "topics_pending").glob("*.json")) if (private / "topics_pending").is_dir() else []:
        stats["topic_candidates"] += import_topics_pending_json(conn, path, path.stem)
    for path in sorted((private / "facts_pending").glob("*.json")) if (private / "facts_pending").is_dir() else []:
        stats["fact_candidates"] += import_facts_pending_json(conn, path, path.stem)
    for path in sorted((private / "personas").glob("*.json")) if (private / "personas").is_dir() else []:
        if import_persona_card_json(conn, path, path.stem.replace("_facts.extracted", "")):
            if not path.name.endswith("_facts.extracted.json"):
                stats["persona_cards"] += 1
    conn.commit()
    return stats

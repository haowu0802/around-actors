#!/usr/bin/env python3
"""Topic-slot governance: LLM propose-one, review, apply to Postgres.

Primary workflow:
  llm-propose -> stg.persona_topic_candidates (pending)
  approve/apply -> stg.persona_topic_specs

Legacy n-gram --mode extract remains for debugging only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from chat_persona import (  # noqa: E402
    chat_completion,
    normalize_kobold_base_url,
    resolve_model,
)
from persona_store import (  # noqa: E402
    delete_topic_candidate,
    ensure_governance_schema,
    list_topic_blocks,
    list_topic_candidates,
    list_topic_specs,
    load_persona_card_db,
    load_topics_pending_view,
    replace_topic_candidates,
    set_topic_blocks,
    topic_candidate_to_dict,
    upsert_topic_candidate,
    upsert_topic_extract_run,
    upsert_topic_spec,
)

WS_RE = re.compile(r"\s+")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
LAUGH_ONLY_RE = re.compile(r"^[哈嘿呵嗯啊哦呃喔唉]+$")
PUNCT_HEAVY_RE = re.compile(r"^[\W_0-9]+$", re.UNICODE)


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
        description="Topic slots: LLM propose-one / legacy extract / apply (Postgres)"
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--actor-key", required=True)
    p.add_argument(
        "--mode",
        choices=("llm-propose", "extract", "apply", "list", "set-status"),
        default="llm-propose",
    )
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--since-days", type=int, default=0)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument(
        "--max-df-ratio",
        type=float,
        default=0.004,
        help="Legacy extract: drop high-DF habit glue",
    )
    p.add_argument(
        "--max-hits",
        type=int,
        default=40,
        help="Legacy extract: drop high hit-count habit glue",
    )
    p.add_argument("--include-existing", action="store_true")
    p.add_argument("--include-rejected", action="store_true")
    p.add_argument("--stopwords-file", default=None)
    p.add_argument(
        "--keys",
        default="",
        help="Comma-separated fact_key list for set-status, or optional apply subset",
    )
    p.add_argument(
        "--status",
        default="",
        choices=("", "pending", "approved", "rejected"),
    )
    p.add_argument(
        "--default-kind",
        default="preference",
        choices=("profile", "preference", "episode"),
    )
    p.add_argument(
        "--promote-pending-to-stops",
        action="store_true",
        help="Legacy: move pending phrases into stop blocks and clear queue",
    )
    p.add_argument(
        "--kobold-url",
        default=os.environ.get("KOBOLD_URL", "http://127.0.0.1:5001/v1"),
    )
    p.add_argument("--model", default=os.environ.get("KOBOLD_MODEL", ""))
    p.add_argument("--gap-sec", type=int, default=3 * 3600, help="Session gap for windows")
    p.add_argument("--max-window-msgs", type=int, default=50)
    p.add_argument("--min-window-msgs", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument(
        "--redo-fact-key",
        default="",
        help="Replace this pending candidate after a redo propose",
    )
    p.add_argument(
        "--window-ids",
        default="",
        help="Comma-separated message ids to reuse as the propose window (redo)",
    )
    p.add_argument(
        "--feedback",
        default="",
        help="Human feedback for redo (why previous proposal was wrong)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_stopwords(repo_root: Path, explicit: str | None) -> set[str]:
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))
    paths.append(repo_root / "data" / "private" / "topic_mine" / "stopwords.json")
    locale_dir = repo_root / "data" / "private" / "locales"
    if locale_dir.is_dir():
        paths.extend(sorted(locale_dir.glob("*.json")))
    paths.append(repo_root / "data" / "samples" / "topic_mine_stopwords.example.json")
    out: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        raw = load_json(path)
        if isinstance(raw, list):
            out.update(str(x).strip() for x in raw if str(x).strip())
        elif isinstance(raw, dict):
            for key in ("stopwords", "stop_tokens", "phrases"):
                vals = raw.get(key) or []
                if isinstance(vals, list):
                    out.update(str(x).strip() for x in vals if str(x).strip())
    return out


def draft_fact_key(phrase: str) -> str:
    digest = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:10]
    return f"draft_{digest}"


def normalize_phrase(text: str) -> str:
    return WS_RE.sub("", (text or "").strip())


def is_noise_phrase(phrase: str, stopwords: set[str]) -> bool:
    p = normalize_phrase(phrase)
    if len(p) < 2 or len(p) > 12:
        return True
    if p in stopwords:
        return True
    if LAUGH_ONLY_RE.match(p):
        return True
    if PUNCT_HEAVY_RE.match(p):
        return True
    if len(set(p)) == 1:
        return True
    return False


def iter_phrases(text: str) -> set[str]:
    found: set[str] = set()
    for run in CJK_RUN_RE.findall(text or ""):
        n = len(run)
        for size in (2, 3, 4):
            if n < size:
                continue
            for i in range(0, n - size + 1):
                found.add(run[i : i + size])
    for tok in LATIN_TOKEN_RE.findall(text or ""):
        low = tok.lower()
        if low in {"http", "https", "www", "com", "http://", "https://"}:
            continue
        if low.startswith("http"):
            continue
        found.add(low)
    return found


def fetch_actor_messages(
    conn: psycopg.Connection,
    actor_key: str,
    *,
    min_epoch: int | None,
    limit_rows: int = 20000,
) -> list[dict]:
    time_sql = ""
    params: list[Any] = [actor_key]
    if min_epoch is not None:
        time_sql = " AND create_time_epoch >= %s "
        params.append(min_epoch)
    params.append(limit_rows)
    sql = f"""
        SELECT id, text_content, create_time_epoch
        FROM stg.messages
        WHERE actor_key = %s
          AND speaker_role = 'actor'
          AND has_semantic_text = TRUE
          AND char_length(text_content) BETWEEN 2 AND 120
          {time_sql}
        ORDER BY create_time_epoch DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def covered_phrases(official: list[dict]) -> set[str]:
    out: set[str] = set()
    for spec in official:
        for field in ("match_any", "prefer_any"):
            for term in spec.get(field) or []:
                t = normalize_phrase(str(term))
                if t:
                    out.add(t)
        fk = normalize_phrase(str(spec.get("fact_key") or ""))
        if fk:
            out.add(fk)
    return out


def suppress_substrings(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    phrases = [normalize_phrase(c["phrase"]) for c in scored]
    for i, c in enumerate(scored):
        p = phrases[i]
        dominated = False
        for j, other in enumerate(scored):
            if i == j:
                continue
            q = phrases[j]
            if len(q) <= len(p):
                continue
            if p in q and int(other["hit_count"]) >= int(c["hit_count"]) * 0.75:
                dominated = True
                break
        if not dominated:
            kept.append(c)
    return kept


def mine_candidates(
    rows: list[dict],
    *,
    stopwords: set[str],
    min_hits: int,
    now_epoch: int,
    max_df_ratio: float = 0.004,
    max_hits: int = 40,
    min_phrase_len: int = 3,
) -> tuple[list[dict[str, Any]], int]:
    hits: dict[str, set[int]] = defaultdict(set)
    latest: dict[str, int] = {}
    examples: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for r in rows:
        mid = int(r["id"])
        text = (r.get("text_content") or "").strip().replace("\n", " ")
        if not text:
            continue
        ts = int(r["create_time_epoch"] or 0)
        for phrase in iter_phrases(text):
            if len(phrase) < min_phrase_len:
                continue
            if is_noise_phrase(phrase, stopwords):
                continue
            hits[phrase].add(mid)
            if ts >= latest.get(phrase, 0):
                latest[phrase] = ts
            bucket = examples[phrase]
            if len(bucket) < 3 and all(mid != x[0] for x in bucket):
                bucket.append((mid, text[:80]))

    msg_n = max(1, len(rows))
    scored: list[dict[str, Any]] = []
    skipped_high_df = 0
    for phrase, id_set in hits.items():
        hit_count = len(id_set)
        if hit_count < min_hits:
            continue
        df_ratio = hit_count / msg_n
        if df_ratio >= max_df_ratio or hit_count >= max_hits:
            skipped_high_df += 1
            continue
        age_days = 9999.0
        ts = latest.get(phrase)
        if ts:
            age_days = max(0.0, (now_epoch - ts) / 86400.0)
        recency = (
            20.0
            if age_days <= 30
            else 12.0
            if age_days <= 180
            else 6.0
            if age_days <= 365
            else 2.0
        )
        length_bonus = 12.0 if len(phrase) >= 3 else 4.0
        volume = min(35.0, 9.0 * (hit_count**0.5))
        ubiquity_penalty = 0.0
        if df_ratio >= max_df_ratio * 0.66 or hit_count >= int(max_hits * 0.66):
            ubiquity_penalty = -12.0
        score = min(
            100.0, max(0.0, 15.0 + volume + recency + length_bonus + ubiquity_penalty)
        )
        scored.append(
            {
                "phrase": phrase,
                "hit_count": hit_count,
                "latest_epoch": ts,
                "example_message_ids": [x[0] for x in examples[phrase]],
                "example_excerpts": [x[1] for x in examples[phrase]],
                "score": round(score, 2),
                "df_ratio": round(df_ratio, 4),
            }
        )

    scored.sort(key=lambda c: (-float(c["score"]), -int(c["hit_count"]), c["phrase"]))
    kept = suppress_substrings(scored)
    return kept, skipped_high_df


def cmd_extract(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    stopwords: set[str],
    limit: int,
    since_days: int,
    min_hits: int,
    min_score: float,
    max_df_ratio: float,
    max_hits: int,
    include_existing: bool,
    include_rejected: bool,
    default_kind: str,
    dry_run: bool,
) -> int:
    prior_rows = list_topic_candidates(conn, actor_key)
    prior_status = {
        str(r.get("fact_key") or ""): str(r.get("status") or "pending")
        for r in prior_rows
        if r.get("status") in {"approved", "rejected"}
    }
    blocked_keys, blocked_phrases = list_topic_blocks(conn, actor_key)
    for r in prior_rows:
        if r.get("status") == "rejected":
            fk = str(r.get("fact_key") or "")
            ph = normalize_phrase(str(r.get("phrase") or ""))
            if fk:
                blocked_keys.add(fk)
            if ph:
                blocked_phrases.add(ph)

    # Stops are also mine stopwords.
    stopwords = set(stopwords) | blocked_phrases

    official = list_topic_specs(conn, actor_key, enabled_only=False)
    existing_keys = {str(s.get("fact_key") or "") for s in official if s.get("fact_key")}
    existing_phrases = covered_phrases(official)

    now_epoch = int(time.time())
    min_epoch = now_epoch - since_days * 86400 if since_days > 0 else None
    rows = fetch_actor_messages(conn, actor_key, min_epoch=min_epoch)
    mined, skipped_high_df = mine_candidates(
        rows,
        stopwords=stopwords,
        min_hits=min_hits,
        now_epoch=now_epoch,
        max_df_ratio=max_df_ratio,
        max_hits=max_hits,
    )

    skipped_existing = 0
    skipped_rejected = 0
    skipped_low = 0
    candidates: list[dict[str, Any]] = []

    for m in mined:
        phrase = normalize_phrase(m["phrase"])
        fact_key = draft_fact_key(phrase)
        if not include_existing and (
            phrase in existing_phrases
            or any(phrase in ep or ep in phrase for ep in existing_phrases if len(ep) >= 2)
        ):
            skipped_existing += 1
            continue
        if not include_existing and fact_key in existing_keys:
            skipped_existing += 1
            continue
        if not include_rejected and (fact_key in blocked_keys or phrase in blocked_phrases):
            skipped_rejected += 1
            continue
        if float(m["score"]) < min_score:
            skipped_low += 1
            continue
        status = prior_status.get(fact_key, "pending")
        if status == "rejected" and not include_rejected:
            skipped_rejected += 1
            continue
        candidates.append(
            {
                "fact_key": fact_key,
                "kind": default_kind,
                "phrase": phrase,
                "match_any": [phrase],
                "prefer_any": [phrase],
                "min_len": 4,
                "max_len": 80,
                "hit_count": m["hit_count"],
                "latest_epoch": m.get("latest_epoch"),
                "example_message_ids": m.get("example_message_ids") or [],
                "example_excerpts": m.get("example_excerpts") or [],
                "score": m["score"],
                "status": status if status != "rejected" else "pending",
                "source": "topic_mine_v1",
                "enabled": True,
                "meta": {"df_ratio": m.get("df_ratio")},
            }
        )

    candidates.sort(key=lambda c: (-float(c["score"]), -int(c["hit_count"]), c["fact_key"]))
    if limit > 0:
        candidates = candidates[:limit]
    for item in candidates:
        print(
            f"CANDIDATE {item['fact_key']} kind={item['kind']} score={item['score']:.1f} "
            f"hits={item['hit_count']} phrase={item['phrase']!r}"
        )

    run_meta = {
        "limit": limit,
        "since_days": since_days,
        "min_hits": min_hits,
        "min_score": min_score,
        "max_df_ratio": max_df_ratio,
        "max_hits": max_hits,
        "default_kind": default_kind,
        "messages_scanned": len(rows),
        "skipped_existing": skipped_existing,
        "skipped_rejected": skipped_rejected,
        "skipped_low_score": skipped_low,
        "skipped_high_df": skipped_high_df,
        "official_topic_keys": sorted(existing_keys),
    }

    if not candidates:
        if not dry_run:
            set_topic_blocks(conn, actor_key, fact_keys=blocked_keys, phrases=blocked_phrases)
            conn.commit()
        print(
            f"extract_done actor_key={actor_key} candidates=0 "
            f"skipped_existing={skipped_existing} skipped_rejected={skipped_rejected} "
            f"skipped_low={skipped_low} skipped_high_df={skipped_high_df} "
            f"messages_scanned={len(rows)} storage=postgres"
        )
        return 0

    if dry_run:
        print(json.dumps({"candidates": candidates, "meta": run_meta}, ensure_ascii=False, indent=2))
    else:
        set_topic_blocks(conn, actor_key, fact_keys=blocked_keys, phrases=blocked_phrases)
        replace_topic_candidates(conn, actor_key, candidates, run_meta=run_meta)
        conn.commit()
        print("storage=postgres table=stg.persona_topic_candidates")
    print(
        f"extract_done actor_key={actor_key} candidates={len(candidates)} "
        f"skipped_existing={skipped_existing} skipped_rejected={skipped_rejected} "
        f"skipped_low={skipped_low} skipped_high_df={skipped_high_df} "
        f"messages_scanned={len(rows)}"
    )
    return 0


def cmd_promote_pending_to_stops(
    conn: psycopg.Connection, actor_key: str, *, dry_run: bool
) -> int:
    """Treat current pending phrases as habit stops and clear the topic queue."""
    rows = list_topic_candidates(conn, actor_key)
    blocked_keys, blocked_phrases = list_topic_blocks(conn, actor_key)
    n = 0
    for r in rows:
        ph = normalize_phrase(str(r.get("phrase") or ""))
        fk = str(r.get("fact_key") or "")
        if ph:
            blocked_phrases.add(ph)
            n += 1
        if fk:
            blocked_keys.add(fk)
        print(f"STOP phrase={ph!r} key={fk}")
    if dry_run:
        print(f"dry_run promote_pending_to_stops n={n}")
        return 0
    set_topic_blocks(conn, actor_key, fact_keys=blocked_keys, phrases=blocked_phrases)
    replace_topic_candidates(conn, actor_key, [], run_meta={"cleared_to_stops": n})
    conn.commit()
    print(f"promote_pending_to_stops done n={n} pending_cleared=1")
    return 0


def cmd_list(conn: psycopg.Connection, actor_key: str) -> int:
    data = load_topics_pending_view(conn, actor_key)
    cands = data.get("candidates") or []
    if not cands:
        print(f"no topic candidates for actor_key={actor_key}")
        return 1
    print(f"storage=postgres actor_key={actor_key} candidates={len(cands)}")
    for c in cands:
        print(
            f"  [{c.get('status')}] {c.get('fact_key')} kind={c.get('kind')} "
            f"score={c.get('score')} hits={c.get('hit_count')} phrase={c.get('phrase')!r}"
        )
    return 0


def cmd_set_status(
    conn: psycopg.Connection, actor_key: str, keys: list[str], status: str
) -> int:
    keyset = set(keys)
    n = 0
    for row in list_topic_candidates(conn, actor_key):
        fk = str(row.get("fact_key") or "")
        if fk not in keyset:
            continue
        c = topic_candidate_to_dict(row)
        c["status"] = status
        upsert_topic_candidate(conn, actor_key, c)
        n += 1
    conn.commit()
    print(f"updated={n} status={status}")
    return 0 if n else 1


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

LLM_TOPIC_SYSTEM = """You propose at most ONE durable memory topic slot about the actor from a chat window.
A topic is a reusable schema slot for later fact mining (identity, stable preference, health limit, important relationship role).
Do NOT propose discourse glue, greetings, one-off plans, temporary moods, or style habits.
The slot title may be an inferred label even if that exact phrase never appears.
match_any must be concrete search clues that could appear in chat (synonyms / related words).
In why/title text, refer to the actor only by the provided display name (or say "the actor").
Never invent a Chinese name from the actor_key spelling.
Return ONLY JSON, no markdown:
{"topic": null}
or
{"topic": {"title_zh":"...","kind":"preference|profile|episode","match_any":["..."],"prefer_any":["..."],"why":"...","evidence_ids":[123],"confidence":0.0}}
Use evidence_ids only from message ids present in the window. Prefer kind=preference or profile; use episode only if clearly recurring.
If nothing durable is present, return {"topic": null}.
"""


def fetch_dialogue_messages(
    conn: psycopg.Connection, actor_key: str
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, speaker_role, text_content, create_time_epoch
            FROM stg.messages
            WHERE actor_key = %s
              AND has_semantic_text = TRUE
              AND speaker_role IN ('self', 'actor')
              AND text_content IS NOT NULL
              AND text_content <> ''
              AND create_time_epoch IS NOT NULL
            ORDER BY create_time_epoch ASC, id ASC
            """,
            (actor_key,),
        )
        return list(cur.fetchall())


def fetch_messages_by_ids(
    conn: psycopg.Connection, actor_key: str, ids: list[int]
) -> list[dict[str, Any]]:
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, speaker_role, text_content, create_time_epoch
            FROM stg.messages
            WHERE actor_key = %s AND id = ANY(%s)
            ORDER BY create_time_epoch ASC, id ASC
            """,
            (actor_key, ids),
        )
        return list(cur.fetchall())


def split_sessions(
    rows: list[dict[str, Any]], *, gap_sec: int
) -> list[list[dict[str, Any]]]:
    sessions: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    prev_ts: int | None = None
    for r in rows:
        ts = int(r["create_time_epoch"] or 0)
        if cur and prev_ts is not None and ts - prev_ts > gap_sec:
            sessions.append(cur)
            cur = []
        cur.append(r)
        prev_ts = ts
    if cur:
        sessions.append(cur)
    return sessions


def month_bucket(session: list[dict[str, Any]]) -> str:
    ts = int(session[0]["create_time_epoch"] or 0)
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def trim_window(
    session: list[dict[str, Any]], max_msgs: int
) -> list[dict[str, Any]]:
    if len(session) <= max_msgs:
        return session
    # Prefer the end of the session (usually denser replies).
    return session[-max_msgs:]


def sample_random_window(
    sessions: list[list[dict[str, Any]]],
    *,
    min_msgs: int,
    max_msgs: int,
) -> tuple[list[dict[str, Any]], str] | None:
    usable = [s for s in sessions if len(s) >= min_msgs]
    if not usable:
        return None
    buckets: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for s in usable:
        buckets[month_bucket(s)].append(s)
    bucket_keys = [k for k, v in buckets.items() if v]
    if not bucket_keys:
        return None
    key = random.choice(bucket_keys)
    session = random.choice(buckets[key])
    return trim_window(session, max_msgs), key


def format_window_for_prompt(window: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for r in window:
        mid = int(r["id"])
        role = str(r.get("speaker_role") or "?")
        text = (r.get("text_content") or "").strip().replace("\n", " ")
        if len(text) > 240:
            text = text[:240] + "…"
        lines.append(f"[id={mid}][{role}] {text}")
    return "\n".join(lines)


def _token_set(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    for v in values or []:
        p = normalize_phrase(str(v))
        if p:
            out.add(p)
    return out


def topic_overlaps_existing(
    match_any: list[str],
    title: str,
    *,
    official: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    skip_fact_key: str | None = None,
) -> str | None:
    proposed = _token_set(match_any)
    title_n = normalize_phrase(title)
    if title_n:
        proposed.add(title_n)
    for spec in official:
        existing = _token_set(list(spec.get("match_any") or []))
        existing |= _token_set(list(spec.get("prefer_any") or []))
        fk = normalize_phrase(str(spec.get("fact_key") or ""))
        if fk:
            existing.add(fk)
        if proposed & existing:
            return f"overlaps official {spec.get('fact_key')}"
    for c in candidates:
        fk = str(c.get("fact_key") or "")
        if skip_fact_key and fk == skip_fact_key:
            continue
        st = str(c.get("status") or "")
        if st not in {"pending", "approved", "applied"}:
            continue
        existing = _token_set(list(c.get("match_any") or []))
        existing |= _token_set([str(c.get("phrase") or "")])
        meta = c.get("meta") or {}
        if isinstance(meta, dict):
            existing |= _token_set([str(meta.get("title_zh") or "")])
        if proposed & existing:
            return f"overlaps candidate {fk}"
    return None


def reject_memory_lines(candidates: list[dict[str, Any]], limit: int = 20) -> list[str]:
    lines: list[str] = []
    for c in candidates:
        if str(c.get("status") or "") != "rejected":
            continue
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        title = str((meta or {}).get("title_zh") or c.get("phrase") or c.get("fact_key") or "")
        match = ", ".join(str(x) for x in (c.get("match_any") or [])[:6])
        lines.append(f"- {title} (match: {match})")
        if len(lines) >= limit:
            break
    return lines


def official_memory_lines(official: list[dict[str, Any]], limit: int = 30) -> list[str]:
    lines: list[str] = []
    for spec in official:
        match = ", ".join(str(x) for x in (spec.get("match_any") or [])[:6])
        lines.append(f"- {spec.get('fact_key')} kind={spec.get('kind')} match=[{match}]")
        if len(lines) >= limit:
            break
    return lines


def parse_topic_payload(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = JSON_BLOCK_RE.search(text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def normalize_llm_topic(
    payload: dict[str, Any], window_ids: set[int]
) -> dict[str, Any] | None:
    topic = payload.get("topic")
    if topic is None:
        return None
    if not isinstance(topic, dict):
        return None
    title = str(topic.get("title_zh") or topic.get("title") or "").strip()
    match_any = [str(x).strip() for x in (topic.get("match_any") or []) if str(x).strip()]
    prefer_any = [str(x).strip() for x in (topic.get("prefer_any") or []) if str(x).strip()]
    if not match_any and title:
        match_any = [title]
    if not match_any:
        return None
    if not prefer_any:
        prefer_any = list(match_any)
    kind = str(topic.get("kind") or "preference").strip()
    if kind not in {"profile", "preference", "episode"}:
        kind = "preference"
    evid_raw = topic.get("evidence_ids") or []
    evid: list[int] = []
    for x in evid_raw:
        try:
            mid = int(x)
        except (TypeError, ValueError):
            continue
        if mid in window_ids:
            evid.append(mid)
    try:
        conf = float(topic.get("confidence") or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    why = str(topic.get("why") or "").strip()
    return {
        "title_zh": title or match_any[0],
        "kind": kind,
        "match_any": match_any,
        "prefer_any": prefer_any,
        "evidence_ids": evid,
        "confidence": conf,
        "why": why,
    }


def cmd_llm_propose_one(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    kobold_url: str,
    model: str,
    gap_sec: int,
    max_window_msgs: int,
    min_window_msgs: int,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    redo_fact_key: str = "",
    window_ids: list[int] | None = None,
    feedback: str = "",
) -> tuple[int, dict[str, Any]]:
    """Propose one topic into pending. Returns (code, info).

    codes: 0 ok, 1 no topic, 2 error, 3 duplicate
    """
    base = normalize_kobold_base_url(kobold_url)
    try:
        model_id = resolve_model(base, model)
    except Exception as e:
        print(f"kobold model resolve failed: {e}", file=sys.stderr)
        return 2, {"error": str(e)}

    official = list_topic_specs(conn, actor_key, enabled_only=False)
    candidates = [topic_candidate_to_dict(r) for r in list_topic_candidates(conn, actor_key)]
    bucket = ""

    if window_ids:
        window = fetch_messages_by_ids(conn, actor_key, window_ids)
        if len(window) < 2:
            print("redo window_ids produced too few messages", file=sys.stderr)
            return 2, {"error": "empty_window"}
    else:
        rows = fetch_dialogue_messages(conn, actor_key)
        sessions = split_sessions(rows, gap_sec=gap_sec)
        sampled = sample_random_window(
            sessions, min_msgs=min_window_msgs, max_msgs=max_window_msgs
        )
        if not sampled:
            print("no eligible dialogue windows", file=sys.stderr)
            return 2, {"error": "no_windows"}
        window, bucket = sampled

    window_id_list = [int(r["id"]) for r in window]
    window_id_set = set(window_id_list)
    excerpts = [
        (r.get("text_content") or "").strip().replace("\n", " ")[:80] for r in window[:5]
    ]

    avoid_official = "\n".join(official_memory_lines(official)) or "(none)"
    avoid_rejected = "\n".join(reject_memory_lines(candidates)) or "(none)"
    card = load_persona_card_db(conn, actor_key) or {}
    display_name = str(card.get("display_name") or "").strip() or actor_key
    user_parts = [
        f"Actor key (machine id only, not a personal name): {actor_key}",
        f"Actor display name (use this if you need a name): {display_name}",
        "Already official topics (do not duplicate):",
        avoid_official,
        "Previously rejected proposals (do not repeat):",
        avoid_rejected,
        "Chat window:",
        format_window_for_prompt(window),
    ]
    if feedback.strip():
        user_parts.append(f"Human feedback for redo: {feedback.strip()}")
    if redo_fact_key.strip():
        user_parts.append(
            "This is a REDO of a previous proposal on the SAME window. "
            "Do not repeat the same title/match_any. Propose a different durable slot, "
            "or return {\"topic\": null} if none remains."
        )
    user_parts.append("Propose zero or one new topic JSON now.")

    messages = [
        {"role": "system", "content": LLM_TOPIC_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    try:
        raw = chat_completion(
            base,
            model_id,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
    except Exception as e:
        print(f"kobold propose failed: {e}", file=sys.stderr)
        return 2, {"error": str(e)}

    payload = parse_topic_payload(raw)
    if payload is None:
        print(f"parse_failed raw={raw[:300]!r}", file=sys.stderr)
        return 2, {"error": "parse_failed", "raw": raw[:500]}

    topic = normalize_llm_topic(payload, window_id_set)
    run_meta = {
        "mode": "llm_propose_one",
        "model": model_id,
        "bucket": bucket,
        "window_ids": window_id_list,
        "window_n": len(window_id_list),
        "raw_preview": (raw or "")[:400],
    }
    if topic is None:
        print("llm_propose_one topic=null")
        if not dry_run:
            upsert_topic_extract_run(conn, actor_key, {**run_meta, "last_result": "null"})
            conn.commit()
        return 1, {"result": "null", "window_ids": window_id_list, "bucket": bucket}

    skip_key = redo_fact_key.strip() or None
    overlap = topic_overlaps_existing(
        topic["match_any"],
        topic["title_zh"],
        official=official,
        candidates=candidates,
        skip_fact_key=skip_key,
    )
    if overlap:
        print(f"llm_propose_one duplicate: {overlap}")
        if not dry_run:
            upsert_topic_extract_run(
                conn, actor_key, {**run_meta, "last_result": "duplicate", "detail": overlap}
            )
            conn.commit()
        return 3, {
            "result": "duplicate",
            "detail": overlap,
            "window_ids": window_id_list,
            "bucket": bucket,
        }

    fact_key = draft_fact_key(topic["title_zh"] + "|" + "|".join(topic["match_any"]))
    candidate = {
        "fact_key": fact_key,
        "status": "pending",
        "kind": topic["kind"],
        "phrase": topic["title_zh"],
        "match_any": topic["match_any"],
        "prefer_any": topic["prefer_any"],
        "min_len": 4,
        "max_len": 80,
        "hit_count": len(topic["evidence_ids"]) or len(window_id_list),
        "score": round(100.0 * float(topic["confidence"]), 2),
        "latest_epoch": int(window[-1]["create_time_epoch"] or 0),
        "example_message_ids": topic["evidence_ids"] or window_id_list[:5],
        "example_excerpts": excerpts,
        "source": "llm_window_v1",
        "enabled": True,
        "meta": {
            "title_zh": topic["title_zh"],
            "why": topic["why"],
            "confidence": topic["confidence"],
            "window_ids": window_id_list,
            "bucket": bucket,
            "model": model_id,
            "feedback": feedback.strip() or None,
        },
    }
    print(
        f"PROPOSE {fact_key} kind={topic['kind']} title={topic['title_zh']!r} "
        f"match={topic['match_any']} bucket={bucket}"
    )
    if dry_run:
        return 0, {"result": "dry_run", "candidate": candidate}

    upsert_topic_candidate(conn, actor_key, candidate)
    if skip_key and skip_key != fact_key:
        delete_topic_candidate(conn, actor_key, skip_key)
    upsert_topic_extract_run(
        conn,
        actor_key,
        {**run_meta, "last_result": "pending", "last_fact_key": fact_key},
    )
    conn.commit()
    return 0, {"result": "pending", "fact_key": fact_key, "candidate": candidate}


def cmd_apply(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    dry_run: bool,
    fact_keys: list[str] | None = None,
) -> int:
    """Apply approved topic candidates into official specs.

    If fact_keys is set, only those approved keys are applied (single or subset).
    """
    key_filter = {k.strip() for k in (fact_keys or []) if str(k).strip()} or None
    approved = [
        topic_candidate_to_dict(r)
        for r in list_topic_candidates(conn, actor_key)
        if r.get("status") == "approved"
    ]
    if key_filter is not None:
        approved = [
            c for c in approved if str(c.get("fact_key") or "").strip() in key_filter
        ]
    if not approved:
        if key_filter is not None:
            print(
                "nothing to apply (no status=approved matching --keys)",
                file=sys.stderr,
            )
        else:
            print("nothing to apply (no status=approved)")
        return 1

    applied = 0
    for c in approved:
        fk = str(c.get("fact_key") or "").strip()
        if not fk:
            continue
        match_any = [str(x) for x in (c.get("match_any") or []) if str(x).strip()]
        prefer_any = [str(x) for x in (c.get("prefer_any") or []) if str(x).strip()]
        if not match_any:
            phrase = normalize_phrase(str(c.get("phrase") or ""))
            if phrase:
                match_any = [phrase]
                prefer_any = prefer_any or [phrase]
        spec = {
            "fact_key": fk,
            "kind": c.get("kind") or "preference",
            "match_any": match_any,
            "prefer_any": prefer_any or match_any,
            "min_len": int(c.get("min_len") or 4),
            "max_len": int(c.get("max_len") or 80),
            "enabled": bool(c.get("enabled", True)),
            "source": "topics_pending_apply",
        }
        print(f"APPLY {fk} kind={spec['kind']} match_any={match_any}")
        if not dry_run:
            upsert_topic_spec(conn, actor_key, spec)
            c["status"] = "applied"
            upsert_topic_candidate(conn, actor_key, c)
        applied += 1

    if not dry_run:
        conn.commit()
    print(f"apply_done applied={applied} storage=postgres table=stg.persona_topic_specs")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    args = parse_args()
    if not args.database_url:
        print("DATABASE_URL required", file=sys.stderr)
        return 2

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        ensure_governance_schema(conn, repo_root)
        if args.promote_pending_to_stops:
            return cmd_promote_pending_to_stops(
                conn, args.actor_key, dry_run=args.dry_run
            )
        if args.mode == "list":
            return cmd_list(conn, args.actor_key)
        if args.mode == "set-status":
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
            if not keys or not args.status:
                print("--keys and --status required for set-status", file=sys.stderr)
                return 2
            return cmd_set_status(conn, args.actor_key, keys, args.status)
        if args.mode == "apply":
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
            return cmd_apply(
                conn,
                actor_key=args.actor_key,
                dry_run=args.dry_run,
                fact_keys=keys or None,
            )
        if args.mode == "llm-propose":
            window_ids = [
                int(x.strip())
                for x in str(args.window_ids or "").split(",")
                if x.strip().isdigit()
            ]
            code, info = cmd_llm_propose_one(
                conn,
                actor_key=args.actor_key,
                kobold_url=args.kobold_url,
                model=args.model,
                gap_sec=int(args.gap_sec),
                max_window_msgs=int(args.max_window_msgs),
                min_window_msgs=int(args.min_window_msgs),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                dry_run=args.dry_run,
                redo_fact_key=str(args.redo_fact_key or ""),
                window_ids=window_ids or None,
                feedback=str(args.feedback or ""),
            )
            if code == 0:
                print(f"ok {info.get('fact_key') or info.get('result')}")
            return code

        stopwords = load_stopwords(repo_root, args.stopwords_file)
        return cmd_extract(
            conn,
            actor_key=args.actor_key,
            stopwords=stopwords,
            limit=args.limit,
            since_days=args.since_days,
            min_hits=args.min_hits,
            min_score=args.min_score,
            max_df_ratio=float(args.max_df_ratio),
            max_hits=int(args.max_hits),
            include_existing=args.include_existing,
            include_rejected=args.include_rejected,
            default_kind=args.default_kind,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    raise SystemExit(main())

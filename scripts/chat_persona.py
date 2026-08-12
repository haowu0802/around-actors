#!/usr/bin/env python3
"""Minimal persona chat CLI with lightweight RAG over stg.messages + Kobold.

Persona card (optional) + style samples + per-turn keyword retrieval.
Private cards stay under data/private/ (gitignored). No private paths hardcoded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")

# Process-wide locale loaded in main(); scripts may also pass locale explicitly.
_LOCALE: dict[str, Any] = {}


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
    p = argparse.ArgumentParser(description="Chat as an actor using stg.messages + Kobold (+ light RAG)")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--actor-key", default=os.environ.get("ACTOR_KEY", "xi"))
    p.add_argument(
        "--display-name",
        default=os.environ.get("ACTOR_DISPLAY_NAME"),
        help="Name used in the system prompt (default: actor-key)",
    )
    p.add_argument(
        "--kobold-url",
        default=os.environ.get("KOBOLD_URL", "http://127.0.0.1:5001/v1"),
        help="OpenAI-compatible base URL (env KOBOLD_URL). LAN example: http://192.168.x.x:5001/v1",
    )
    p.add_argument("--model", default=os.environ.get("KOBOLD_MODEL", ""), help="Optional model id override")
    p.add_argument("--sample-size", type=int, default=12, help="Number of style lines to include")
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rag", dest="rag", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--facts", dest="facts", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--rag-k", type=int, default=8, help="Top-k retrieved memory lines")
    p.add_argument("--fact-k", type=int, default=6, help="Top-k structured facts to inject")
    p.add_argument("--rag-candidate-limit", type=int, default=800, help="Max candidates scanned per turn")
    p.add_argument("--show-rag", action="store_true", help="Print retrieved lines / facts each turn")
    p.add_argument(
        "--persona-card",
        default=os.environ.get("PERSONA_CARD"),
        help="Path to persona JSON (default: data/private/personas/<actor-key>.json if present)",
    )
    p.add_argument("--no-persona-card", action="store_true", help="Ignore persona card even if found")
    p.add_argument(
        "--locale",
        default=os.environ.get("CHAT_LOCALE"),
        help="Chat locale JSON (default: data/private/locales/zh_wechat.json if present)",
    )
    p.add_argument(
        "--message",
        default=None,
        help="Single-turn user message (if omitted, enter interactive mode)",
    )
    p.add_argument("--show-prompt", action="store_true", help="Print the system prompt then exit")
    return p.parse_args()


def resolve_locale_path(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Locale not found: {path}")
        return path
    private = repo_root / "data" / "private" / "locales" / "zh_wechat.json"
    if private.is_file():
        return private
    example = repo_root / "data" / "samples" / "chat_locale.example.json"
    if example.is_file():
        return example
    raise FileNotFoundError(
        "No chat locale found. Expected data/private/locales/zh_wechat.json "
        f"or {example}"
    )


def load_locale(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Locale must be a JSON object: {path}")
    if not data.get("system_prompt_template"):
        raise ValueError(f"Locale missing system_prompt_template: {path}")
    return data


def set_locale(locale: dict[str, Any]) -> None:
    global _LOCALE
    _LOCALE = locale


def get_locale() -> dict[str, Any]:
    if not _LOCALE:
        raise RuntimeError("Chat locale not loaded; call set_locale() first")
    return _LOCALE


def resolve_persona_card_path(repo_root: Path, actor_key: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    default = repo_root / "data" / "private" / "personas" / f"{actor_key}.json"
    return default if default.is_file() else None


def load_persona_card(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Persona card must be a JSON object: {path}")
    return data


def load_persona_card_for_actor(
    conn: psycopg.Connection,
    repo_root: Path,
    actor_key: str,
    explicit: str | None,
) -> tuple[dict[str, Any] | None, str]:
    """Load persona card. Explicit --persona-card path wins; else Postgres; else default JSON."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return load_persona_card(path), str(path)
    try:
        from persona_store import ensure_governance_schema, load_persona_card_db

        ensure_governance_schema(conn, repo_root)
        card = load_persona_card_db(conn, actor_key)
        if card:
            return card, "postgres"
    except Exception:
        pass
    path = resolve_persona_card_path(repo_root, actor_key, None)
    if path is None:
        return None, "none"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return load_persona_card(path), str(path)


def _bullet_block(items: list[Any] | None, empty: str = "") -> str:
    lines = []
    for item in items or []:
        t = str(item).strip()
        if t:
            lines.append(f"- {t}")
    return "\n".join(lines) if lines else empty


def fetch_style_lines(
    conn: psycopg.Connection,
    actor_key: str,
    sample_size: int,
    min_len: int,
    max_len: int,
) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text_content
            FROM stg.messages
            WHERE actor_key = %s
              AND speaker_role = 'actor'
              AND has_semantic_text = TRUE
              AND char_length(text_content) BETWEEN %s AND %s
              AND text_content !~ '[\\[\\]<>]'
            ORDER BY random()
            LIMIT %s
            """,
            (actor_key, min_len, max_len, sample_size),
        )
        rows = cur.fetchall()
    lines = []
    seen = set()
    for r in rows:
        t = (r["text_content"] or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        lines.append(t)
    return lines


def extract_terms(text: str, stop_tokens: set[str] | None = None) -> list[str]:
    if stop_tokens is None:
        stop_tokens = set(get_locale().get("stop_tokens") or []) if _LOCALE else set()
    terms: list[str] = []
    for m in CJK_RE.findall(text):
        # Prefer whole span, plus overlapping bigrams for short queries.
        if len(m) >= 2 and m not in stop_tokens:
            terms.append(m)
        if len(m) >= 4:
            for i in range(len(m) - 1):
                bg = m[i : i + 2]
                if bg not in stop_tokens:
                    terms.append(bg)
    for m in LATIN_RE.findall(text):
        terms.append(m.lower())
    # de-dupe preserve order
    out = []
    seen = set()
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:24]


def retrieve_memories(
    conn: psycopg.Connection,
    actor_key: str,
    query: str,
    top_k: int,
    candidate_limit: int,
) -> list[dict]:
    terms = extract_terms(query)
    if not terms:
        return []

    # Broad SQL filter with OR ILIKE, then score in Python.
    clauses = []
    params: list = [actor_key]
    for t in terms[:12]:
        clauses.append("text_content ILIKE %s")
        params.append(f"%{t}%")
    params.append(candidate_limit)

    sql = f"""
        SELECT
            id,
            speaker_role,
            sender_display,
            text_content,
            create_time_epoch,
            msg_kind
        FROM stg.messages
        WHERE actor_key = %s
          AND has_semantic_text = TRUE
          AND text_content IS NOT NULL
          AND text_content <> ''
          AND ({' OR '.join(clauses)})
        ORDER BY create_time_epoch DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = list(cur.fetchall())

    scored = []
    for r in rows:
        text = (r["text_content"] or "").strip()
        if not text:
            continue
        score = 0.0
        lower = text.lower()
        for t in terms:
            if t.isascii():
                if t in lower:
                    score += 2.0
            else:
                if t in text:
                    score += 1.5 if len(t) >= 3 else 1.0
        if r["speaker_role"] == "actor":
            score += 0.75
        # Mild preference for medium-length chat lines.
        n = len(text)
        if 6 <= n <= 120:
            score += 0.25
        if score <= 0:
            continue
        scored.append((score, r))

    scored.sort(key=lambda x: (-x[0], -(x[1].get("create_time_epoch") or 0)))
    out = []
    seen = set()
    for score, r in scored:
        t = (r["text_content"] or "").strip()
        if t in seen:
            continue
        seen.add(t)
        item = dict(r)
        item["score"] = score
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def build_system_prompt(
    display_name: str,
    style_lines: list[str],
    card: dict[str, Any] | None = None,
    locale: dict[str, Any] | None = None,
) -> str:
    loc = locale or get_locale()
    empty_ex = loc.get("no_style_examples") or "- (no style samples)"
    examples = "\n".join(f"- {line}" for line in style_lines) if style_lines else empty_ex
    card = card or {}
    name = (card.get("display_name") or display_name).strip() or display_name
    relationship = (card.get("relationship") or "").strip()
    voice = _bullet_block(card.get("voice_notes"))
    boundaries = _bullet_block(card.get("boundaries") or loc.get("default_boundaries") or [])
    extra = _bullet_block(card.get("extra_rules"))

    rel_line = f"{loc.get('label_relationship', 'Relationship: ')}{relationship}\n" if relationship else ""
    voice_block = f"\n{loc.get('label_voice', 'Voice notes:')}\n{voice}\n" if voice else ""
    extra_block = f"\n{loc.get('label_extra', 'Extra rules:')}\n{extra}\n" if extra else ""

    template = loc["system_prompt_template"]
    return template.format(
        name=name,
        rel_line=rel_line,
        voice_block=voice_block,
        boundaries=boundaries,
        extra_block=extra_block,
        examples=examples,
    )


def fetch_persona_facts(
    conn: psycopg.Connection,
    actor_key: str,
    query: str,
    top_k: int,
) -> list[dict]:
    """Retrieve active structured facts; prefer term overlap, always allow abstain if none."""
    terms = extract_terms(query)
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT id, fact_key, statement, evidence_message_ids, confidence
                FROM stg.persona_facts
                WHERE actor_key = %s AND status = 'active'
                """,
                (actor_key,),
            )
            rows = list(cur.fetchall())
        except Exception:
            # Table may not exist yet; clear aborted transaction for later queries.
            try:
                conn.rollback()
            except Exception:
                pass
            return []

    if not rows:
        return []

    scored: list[tuple[float, dict]] = []
    for r in rows:
        text = f"{r.get('fact_key') or ''} {r.get('statement') or ''}"
        score = float(r.get("confidence") or 0.5)
        if terms:
            hits = 0
            for t in terms:
                if t in text:
                    hits += 1
                    score += 1.5 if len(t) >= 3 else 1.0
            if hits == 0:
                # Keep a small always-on prior for very small fact tables, but demote hard.
                score *= 0.15
        scored.append((score, r))

    scored.sort(key=lambda x: (-x[0], str(x[1].get("fact_key") or "")))
    out = []
    for score, r in scored[:top_k]:
        # Drop near-zero relevance when the user query had terms.
        if terms and score < 0.5:
            continue
        item = dict(r)
        item["score"] = score
        out.append(item)
    # If everything was demoted away, return empty → model should abstain on bio facts.
    return out


def format_facts_block(facts: list[dict], locale: dict[str, Any] | None = None) -> str:
    loc = locale or get_locale()
    if not facts:
        return loc.get("facts_empty") or "Structured facts: none highly relevant this turn."
    lines = []
    for f in facts:
        key = f.get("fact_key") or "fact"
        stmt = (f.get("statement") or "").strip().replace("\n", " ")
        evid = f.get("evidence_message_ids") or []
        lines.append(f"- [{key}] {stmt} (evidence_n={len(evid)})")
    header = loc.get("facts_header") or "Structured facts:\n"
    return header + "\n".join(lines)


def format_memory_block(
    display_name: str,
    memories: list[dict],
    locale: dict[str, Any] | None = None,
) -> str:
    if not memories:
        return ""
    loc = locale or get_locale()
    self_label = loc.get("self_speaker_label") or "me"
    other_label = loc.get("other_speaker_label") or "other"
    lines = []
    for m in memories:
        role = m.get("speaker_role") or "unknown"
        who = (
            display_name
            if role == "actor"
            else (self_label if role == "self" else (m.get("sender_display") or other_label))
        )
        text = (m.get("text_content") or "").strip().replace("\n", " ")
        if len(text) > 160:
            text = text[:160] + "…"
        lines.append(f"- {who}: {text}")
    joined = "\n".join(lines)
    header = loc.get("memory_header") or "Retrieved memory lines:\n"
    return header + joined


def normalize_kobold_base_url(url: str) -> str:
    """Normalize OpenAI-compatible base, e.g. http://host:5001/v1."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return "http://127.0.0.1:5001/v1"
    if not u.endswith("/v1"):
        # Accept http://host:5001 or http://host:5001/api
        if u.endswith("/v1/"):
            u = u.rstrip("/")
        elif "/v1" not in u.split("://", 1)[-1]:
            u = u + "/v1"
    return u


def resolve_model(base_url: str, override: str) -> str:
    if override:
        return override
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = data.get("data") or []
    if not items:
        return "koboldcpp"
    return items[0].get("id") or "koboldcpp"


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Kobold request failed: {e}") from e

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected Kobold response: {data!r}")
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    if not args.database_url:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    try:
        locale_path = resolve_locale_path(repo_root, args.locale)
        locale = load_locale(locale_path)
        set_locale(locale)
    except Exception as e:
        print(f"Failed to load chat locale: {e}", file=sys.stderr)
        return 2

    args.kobold_url = normalize_kobold_base_url(args.kobold_url)

    display_name = args.display_name or args.actor_key
    user_label = str(locale.get("cli_user_label") or "you")
    card: dict[str, Any] | None = None
    card_src = "none"

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        if not args.no_persona_card:
            try:
                card, card_src = load_persona_card_for_actor(
                    conn, repo_root, args.actor_key, args.persona_card
                )
            except Exception as e:
                print(f"Failed to load persona card: {e}", file=sys.stderr)
                return 2
            if card and card.get("display_name") and not args.display_name:
                display_name = str(card["display_name"]).strip() or display_name

        style_lines = fetch_style_lines(
            conn,
            args.actor_key,
            args.sample_size,
            args.min_len,
            args.max_len,
        )
        system_prompt = build_system_prompt(display_name, style_lines, card, locale)

        if args.show_prompt:
            print(system_prompt)
            print(
                f"\n# style_lines={len(style_lines)} actor_key={args.actor_key} "
                f"rag={args.rag} facts={args.facts} persona_card={card_src} "
                f"locale={locale_path}"
            )
            return 0

        try:
            model = resolve_model(args.kobold_url, args.model)
        except Exception as e:
            print(f"Cannot reach Kobold at {args.kobold_url}: {e}", file=sys.stderr)
            return 2

        history: list[dict] = [{"role": "system", "content": system_prompt}]

        def ask(user_text: str) -> None:
            memories: list[dict] = []
            facts: list[dict] = []
            if args.facts:
                facts = fetch_persona_facts(conn, args.actor_key, user_text, args.fact_k)
            if args.rag:
                memories = retrieve_memories(
                    conn,
                    args.actor_key,
                    user_text,
                    args.rag_k,
                    args.rag_candidate_limit,
                )
            if args.show_rag:
                print(f"[terms] {extract_terms(user_text)}")
                print(f"[facts] hits={len(facts)}")
                for f in facts:
                    print(f"  ({f.get('score', 0):.2f}) [{f.get('fact_key')}] {f.get('statement')}")
                print(f"[rag] hits={len(memories)}")
                for m in memories:
                    print(f"  ({m['score']:.2f}) {m['speaker_role']}: {m['text_content'][:80]}")

            blocks: list[str] = []
            if args.facts:
                blocks.append(format_facts_block(facts))
            memory_block = format_memory_block(display_name, memories) if args.rag else ""
            if memory_block:
                blocks.append(memory_block)

            history.append({"role": "user", "content": user_text})
            call_messages = list(history)
            if blocks:
                call_messages = history[:-1] + [
                    {"role": "system", "content": "\n\n".join(blocks)},
                    history[-1],
                ]

            reply = chat_completion(
                args.kobold_url,
                model,
                call_messages,
                args.max_tokens,
                args.temperature,
                args.top_p,
            )
            history.append({"role": "assistant", "content": reply})
            print(f"{display_name}: {reply}")

        print(
            f"actor_key={args.actor_key} display_name={display_name} "
            f"style_lines={len(style_lines)} rag={args.rag} facts={args.facts} "
            f"rag_k={args.rag_k} fact_k={args.fact_k} persona_card={card_src} "
            f"locale={locale_path.name}"
        )
        print(f"kobold={args.kobold_url} model={model}")
        print("Type /exit to quit.\n")

        if args.message is not None:
            print(f"{user_label}: {args.message}")
            ask(args.message)
            return 0

        while True:
            try:
                user_text = input(f"{user_label}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            try:
                ask(user_text)
            except Exception as e:
                print(f"[error] {e}", file=sys.stderr)
                if history and history[-1].get("role") == "user":
                    history.pop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

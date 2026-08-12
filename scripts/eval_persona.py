#!/usr/bin/env python3
"""Rule-judge eval for persona chat: run cases against Kobold and score replies.

Private Chinese suites live under data/private/eval/. Reports go to eval/reports/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

# Allow `python scripts/eval_persona.py` without installing a package.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from chat_persona import (  # noqa: E402
    build_system_prompt,
    chat_completion,
    fetch_persona_facts,
    format_facts_block,
    format_memory_block,
    get_locale,
    load_dotenv,
    load_locale,
    load_persona_card,
    normalize_kobold_base_url,
    resolve_locale_path,
    resolve_model,
    resolve_persona_card_path,
    retrieve_memories,
    set_locale,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-judge eval for persona chat")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--suite",
        default=None,
        help="Path to cases JSON (default: data/private/eval/<actor-key>_rules.json)",
    )
    p.add_argument("--actor-key", default=os.environ.get("ACTOR_KEY", "xi"))
    p.add_argument("--display-name", default=os.environ.get("ACTOR_DISPLAY_NAME"))
    p.add_argument(
        "--kobold-url",
        default=os.environ.get("KOBOLD_URL", "http://127.0.0.1:5001/v1"),
        help="OpenAI-compatible base URL (env KOBOLD_URL)",
    )
    p.add_argument("--model", default=os.environ.get("KOBOLD_MODEL", ""))
    p.add_argument("--sample-size", type=int, default=12)
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.4, help="Lower = more stable scores")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rag-k", type=int, default=8)
    p.add_argument("--fact-k", type=int, default=6)
    p.add_argument("--rag-candidate-limit", type=int, default=800)
    p.add_argument(
        "--modes",
        default="grounded,baseline",
        help="Comma list: baseline | rag | card | card+rag | grounded",
    )
    p.add_argument("--persona-card", default=os.environ.get("PERSONA_CARD"))
    p.add_argument(
        "--locale",
        default=os.environ.get("CHAT_LOCALE"),
        help="Chat locale JSON (same as chat_persona.py)",
    )
    p.add_argument("--case-id", default=None, help="Run a single case id")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per case (majority-friendly)")
    p.add_argument(
        "--report-dir",
        default=None,
        help="Directory for JSON/MD reports (default: eval/reports)",
    )
    p.add_argument("--dry-run", action="store_true", help="Load suite and print plan only")
    return p.parse_args()


def fetch_style_lines_stable(
    conn: psycopg.Connection,
    actor_key: str,
    sample_size: int,
    min_len: int,
    max_len: int,
) -> list[str]:
    """Deterministic style sample so mode A/B compares fairly."""
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
            ORDER BY id
            LIMIT %s
            """,
            (actor_key, min_len, max_len, sample_size),
        )
        rows = cur.fetchall()
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        t = (r["text_content"] or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"Suite must be an object with cases[]: {path}")
    return data


def resolve_suite_path(repo_root: Path, actor_key: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    private = repo_root / "data" / "private" / "eval" / f"{actor_key}_rules.json"
    if private.is_file():
        return private
    example = repo_root / "eval" / "cases" / "example.json"
    return example


@dataclass
class ModeConfig:
    name: str
    use_card: bool
    use_rag: bool
    use_facts: bool


def parse_modes(spec: str) -> list[ModeConfig]:
    out: list[ModeConfig] = []
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name == "baseline":
            out.append(ModeConfig(name, False, False, False))
        elif name == "rag":
            out.append(ModeConfig(name, False, True, False))
        elif name == "card":
            out.append(ModeConfig(name, True, False, False))
        elif name in {"card+rag", "card_rag"}:
            out.append(ModeConfig("card+rag", True, True, False))
        elif name in {"grounded", "facts+rag", "full"}:
            # Industry-shaped path: style card + structured facts + episodic RAG.
            out.append(ModeConfig("grounded", True, True, True))
        else:
            raise SystemExit(
                f"Unknown mode: {raw!r} (use baseline,rag,card,card+rag,grounded)"
            )
    if not out:
        raise SystemExit("No modes selected")
    return out


def _contains_any(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    return any(n and n in text for n in needles)


def _contains_none(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    return all(not n or n not in text for n in needles)


def _reply_overlaps_facts(reply: str, fact_statements_joined: str) -> bool:
    for stmt in (fact_statements_joined or "").split("\n"):
        stmt = stmt.strip()
        if len(stmt) >= 4 and stmt[:4] in reply:
            return True
        if len(stmt) >= 2:
            for i in range(len(stmt) - 1):
                bg = stmt[i : i + 2]
                if bg.strip() and bg in reply:
                    return True
    return False


def score_checks(
    checks: dict[str, Any],
    final_reply: str,
    rag_joined: str,
    fact_keys: list[str],
    fact_statements_joined: str,
    *,
    facts_enabled: bool,
    default_abstain_any: list[str] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    ok = True

    def add(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        if not passed:
            ok = False
        details.append({"check": name, "pass": passed, "detail": detail})

    must_any = checks.get("final_reply_must_any") or []
    if must_any:
        passed = _contains_any(final_reply, must_any)
        add("final_reply_must_any", passed, f"need any of {must_any!r}")

    must_not = checks.get("final_reply_must_not_any") or []
    if must_not:
        passed = _contains_none(final_reply, must_not)
        hit = [n for n in must_not if n and n in final_reply]
        add("final_reply_must_not_any", passed, f"banned hits={hit!r}")

    bad_patterns = checks.get("ungrounded_cause_patterns") or []
    bad_hit = [n for n in bad_patterns if n and n in final_reply] if bad_patterns else []
    if bad_patterns:
        add("ungrounded_cause_patterns", len(bad_hit) == 0, f"hits={bad_hit!r}")

    fact_need = checks.get("fact_keys_must_any") or []
    if fact_need and facts_enabled:
        # Retrieval metric — only meaningful when the mode injects facts.
        passed = any(k in fact_keys for k in fact_need)
        add("fact_keys_must_any", passed, f"need any of {fact_need!r}, got {fact_keys!r}")

    rag_any = checks.get("rag_joined_must_any") or []
    if rag_any:
        passed = _contains_any(rag_joined, rag_any)
        add("rag_joined_must_any", passed, f"need any of {rag_any!r} in retrieval")

    if checks.get("final_reply_abstain_or_grounded"):
        abstain_any = checks.get("abstain_any")
        if abstain_any is None:
            abstain_any = list(default_abstain_any or [])
        abstained = _contains_any(final_reply, abstain_any) if abstain_any else False
        # Always allow fact-statement overlap as grounding evidence when facts exist.
        grounded = _reply_overlaps_facts(final_reply, fact_statements_joined)
        if checks.get("grounded_any_from_facts") is False:
            grounded = False
        if not grounded and must_any:
            grounded = _contains_any(final_reply, must_any)
        # Abstain plus invented cause still fails via bad_hit.
        passed = (abstained or grounded) and len(bad_hit) == 0
        add(
            "final_reply_abstain_or_grounded",
            passed,
            f"abstained={abstained} grounded={grounded} bad_hit={bad_hit!r}",
        )

    max_chars = checks.get("max_assistant_chars")
    if max_chars is not None:
        passed = len(final_reply) <= int(max_chars)
        add("max_assistant_chars", passed, f"len={len(final_reply)} max={max_chars}")

    return ok, details


def run_case(
    conn: psycopg.Connection,
    *,
    actor_key: str,
    display_name: str,
    style_lines: list[str],
    card: dict[str, Any] | None,
    mode: ModeConfig,
    case: dict[str, Any],
    kobold_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    rag_k: int,
    fact_k: int,
    rag_candidate_limit: int,
) -> dict[str, Any]:
    use_card = mode.use_card and card is not None
    system = build_system_prompt(display_name, style_lines, card if use_card else None)
    history: list[dict[str, str]] = [{"role": "system", "content": system}]
    transcript: list[dict[str, Any]] = []
    last_rag_joined = ""
    last_fact_keys: list[str] = []
    last_fact_statements = ""

    turns = case.get("turns") or []
    if not turns:
        return {"id": case.get("id"), "pass": False, "error": "case has no turns"}

    for turn in turns:
        if (turn.get("role") or "user") != "user":
            continue
        user_text = (turn.get("content") or "").strip()
        if not user_text:
            continue

        memories: list[dict] = []
        facts: list[dict] = []
        blocks: list[str] = []
        if mode.use_facts:
            facts = fetch_persona_facts(conn, actor_key, user_text, fact_k)
            blocks.append(format_facts_block(facts))
            last_fact_keys = [str(f.get("fact_key") or "") for f in facts]
            last_fact_statements = "\n".join(str(f.get("statement") or "") for f in facts)
        if mode.use_rag:
            memories = retrieve_memories(
                conn, actor_key, user_text, rag_k, rag_candidate_limit
            )
            memory_block = format_memory_block(display_name, memories)
            if memory_block:
                blocks.append(memory_block)
            last_rag_joined = "\n".join((m.get("text_content") or "") for m in memories)

        history.append({"role": "user", "content": user_text})
        call_messages = list(history)
        if blocks:
            call_messages = history[:-1] + [
                {"role": "system", "content": "\n\n".join(blocks)},
                history[-1],
            ]

        reply = chat_completion(
            kobold_url, model, call_messages, max_tokens, temperature, top_p
        )
        history.append({"role": "assistant", "content": reply})
        transcript.append(
            {
                "user": user_text,
                "assistant": reply,
                "fact_keys": [f.get("fact_key") for f in facts],
                "rag_hits": len(memories),
            }
        )

    final_reply = transcript[-1]["assistant"] if transcript else ""
    passed, details = score_checks(
        case.get("checks") or {},
        final_reply,
        last_rag_joined,
        last_fact_keys,
        last_fact_statements,
        facts_enabled=mode.use_facts,
        default_abstain_any=list(get_locale().get("default_abstain_any") or []),
    )
    return {
        "id": case.get("id"),
        "description": case.get("description"),
        "mode": mode.name,
        "pass": passed,
        "checks": details,
        "final_reply": final_reply,
        "transcript": transcript,
        "card_used": use_card,
        "rag_used": mode.use_rag,
        "facts_used": mode.use_facts,
        "fact_keys": last_fact_keys,
    }


def write_reports(report_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"persona_eval_{stamp}.json"
    md_path = report_dir / f"persona_eval_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Persona eval {stamp}",
        "",
        f"- actor_key: `{payload['actor_key']}`",
        f"- suite: `{payload['suite']}`",
        f"- model: `{payload['model']}`",
        f"- temperature: `{payload['temperature']}`",
        f"- repeats: `{payload['repeats']}`",
        "",
        "## Mode summary",
        "",
        "| mode | pass | total | rate |",
        "|---|---:|---:|---:|",
    ]
    for mode, stats in payload["summary_by_mode"].items():
        rate = (stats["pass"] / stats["total"]) if stats["total"] else 0.0
        lines.append(f"| {mode} | {stats['pass']} | {stats['total']} | {rate:.0%} |")

    lines.extend(["", "## Failures", ""])
    fails = [r for r in payload["results"] if not r.get("pass")]
    if not fails:
        lines.append("None.")
    else:
        for r in fails:
            lines.append(f"### `{r.get('id')}` · `{r.get('mode')}`")
            lines.append("")
            lines.append(f"- final: {r.get('final_reply')!r}")
            for c in r.get("checks") or []:
                if not c.get("pass"):
                    lines.append(f"- FAIL `{c['check']}`: {c.get('detail')}")
            lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    try:
        locale_path = resolve_locale_path(repo_root, args.locale)
        set_locale(load_locale(locale_path))
    except Exception as e:
        print(f"Failed to load chat locale: {e}", file=sys.stderr)
        return 2

    args.kobold_url = normalize_kobold_base_url(args.kobold_url)

    if not args.database_url:
        print("Missing --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    suite_path = resolve_suite_path(repo_root, args.actor_key, args.suite)
    if not suite_path.is_file():
        print(f"Suite not found: {suite_path}", file=sys.stderr)
        return 2

    suite = load_suite(suite_path)
    actor_key = suite.get("actor_key") or args.actor_key
    cases = list(suite.get("cases") or [])
    if args.case_id:
        cases = [c for c in cases if c.get("id") == args.case_id]
        if not cases:
            print(f"No case id={args.case_id!r} in suite", file=sys.stderr)
            return 2

    modes = parse_modes(args.modes)
    card_path = resolve_persona_card_path(repo_root, actor_key, args.persona_card)
    card = load_persona_card(card_path) if card_path and card_path.is_file() else None
    display_name = args.display_name or (card or {}).get("display_name") or actor_key

    report_dir = Path(args.report_dir) if args.report_dir else repo_root / "eval" / "reports"

    print(f"suite={suite_path}")
    print(f"actor_key={actor_key} cases={len(cases)} modes={[m.name for m in modes]}")
    print(f"persona_card={card_path if card else 'none'}")
    print(f"report_dir={report_dir}")

    if args.dry_run:
        for c in cases:
            print(f"  - {c.get('id')}: {c.get('description')}")
        return 0

    if not args.database_url:
        return 2

    try:
        model = resolve_model(args.kobold_url, args.model)
    except Exception as e:
        print(f"Cannot reach Kobold at {args.kobold_url}: {e}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    started = time.time()

    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        style_lines = fetch_style_lines_stable(
            conn, actor_key, args.sample_size, args.min_len, args.max_len
        )
        print(f"style_lines={len(style_lines)} model={model} temperature={args.temperature}")

        for mode in modes:
            if mode.use_card and card is None:
                print(f"[skip] mode={mode.name}: no persona card", file=sys.stderr)
                continue
            for case in cases:
                for rep in range(args.repeats):
                    print(f"RUN mode={mode.name} case={case.get('id')} rep={rep+1}/{args.repeats}")
                    try:
                        row = run_case(
                            conn,
                            actor_key=actor_key,
                            display_name=str(display_name),
                            style_lines=style_lines,
                            card=card,
                            mode=mode,
                            case=case,
                            kobold_url=args.kobold_url,
                            model=model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            rag_k=args.rag_k,
                            fact_k=args.fact_k,
                            rag_candidate_limit=args.rag_candidate_limit,
                        )
                    except Exception as e:
                        row = {
                            "id": case.get("id"),
                            "mode": mode.name,
                            "pass": False,
                            "error": str(e),
                            "checks": [],
                            "final_reply": "",
                            "transcript": [],
                        }
                    row["repeat"] = rep + 1
                    results.append(row)
                    status = "PASS" if row.get("pass") else "FAIL"
                    print(f"  {status}: {(row.get('final_reply') or '')[:80]!r}")

    summary: dict[str, dict[str, int]] = {}
    for r in results:
        mode = r.get("mode") or "?"
        summary.setdefault(mode, {"pass": 0, "total": 0})
        summary[mode]["total"] += 1
        if r.get("pass"):
            summary[mode]["pass"] += 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - started, 2),
        "actor_key": actor_key,
        "suite": str(suite_path),
        "persona_card": str(card_path) if card else None,
        "model": model,
        "kobold_url": args.kobold_url,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "modes": [m.name for m in modes],
        "summary_by_mode": summary,
        "results": results,
    }
    json_path, md_path = write_reports(report_dir, payload)
    print("\nSummary:")
    for mode, stats in summary.items():
        rate = (stats["pass"] / stats["total"]) if stats["total"] else 0.0
        print(f"  {mode}: {stats['pass']}/{stats['total']} ({rate:.0%})")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    # Fail if any mode that actually ran has failures.
    any_ran = False
    any_failed = False
    for mode_name, stats in summary.items():
        if stats["total"] <= 0:
            continue
        any_ran = True
        if stats["pass"] < stats["total"]:
            any_failed = True
            print(f"[fail] mode={mode_name} {stats['pass']}/{stats['total']}")
    if not any_ran:
        print("No eval modes produced results", file=sys.stderr)
        return 2
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

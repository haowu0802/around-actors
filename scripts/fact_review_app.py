#!/usr/bin/env python3
"""Local Streamlit UI for topic + fact review (Postgres-backed governance).

English UI labels only. Run:
  streamlit run scripts/fact_review_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import psycopg
import streamlit as st
from psycopg.rows import dict_row

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_persona_facts import (  # noqa: E402
    cmd_apply as facts_cmd_apply,
    cmd_extract as facts_cmd_extract,
    delete_fact,
    ensure_schema,
    fetch_facts,
    load_dotenv,
    resolve_topics,
    set_fact_status,
)
from build_persona_topics import (  # noqa: E402
    cmd_apply as topics_cmd_apply,
    cmd_llm_propose_one,
)
from persona_store import (  # noqa: E402
    add_fact_block,
    delete_topic_spec,
    ensure_governance_schema,
    fact_candidate_to_dict,
    list_persona_card_actors,
    list_topic_specs,
    load_facts_pending_view,
    load_persona_card_db,
    load_topics_pending_view,
    remove_fact_block,
    set_topic_spec_enabled,
    upsert_fact_candidate,
    upsert_persona_card,
    upsert_topic_candidate,
)


def _discover_actor_keys(conn) -> list[str]:
    keys: set[str] = set()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT actor_key FROM stg.messages
                UNION
                SELECT actor_key FROM stg.persona_topic_specs
                UNION
                SELECT actor_key FROM stg.persona_cards
                UNION
                SELECT actor_key FROM stg.persona_facts
                """
            )
            keys.update(str(r["actor_key"]) for r in cur.fetchall())
    except Exception:
        pass
    keys.update(list_persona_card_actors(conn))
    preferred = ["guodahong", "xi"]
    ordered = [k for k in preferred if k in keys]
    ordered.extend(sorted(k for k in keys if k not in preferred))
    return ordered or preferred


def _connect():
    load_dotenv(_REPO / ".env")
    url = st.session_state.get("database_url") or __import__("os").environ.get("DATABASE_URL")
    if not url:
        st.error("DATABASE_URL missing. Set it in .env or the sidebar.")
        return None
    return psycopg.connect(url, row_factory=dict_row)


def _load_evidence_map(conn, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, speaker_role, left(text_content, 240) AS t
            FROM stg.messages
            WHERE id = ANY(%s)
            """,
            (ids,),
        )
        out: dict[int, str] = {}
        for r in cur.fetchall():
            role = str(r.get("speaker_role") or "?")
            out[int(r["id"])] = f"[{role}] {r['t'] or ''}"
        return out


def _render_fact_candidates(data: dict[str, Any], conn, actor_key: str) -> None:
    cands = list(data.get("candidates") or [])
    if not cands:
        st.info("No fact candidates in Postgres queue.")
        return

    all_ids: list[int] = []
    for c in cands:
        all_ids.extend(int(x) for x in (c.get("evidence_message_ids") or []))
    evid = _load_evidence_map(conn, sorted(set(all_ids)))

    st.caption(
        f"Generated: {data.get('generated_at', '?')} · storage={data.get('storage')} · "
        f"skipped_active={data.get('skipped_active_keys', '?')} · "
        f"skipped_rejected={data.get('skipped_rejected_keys', '?')} · "
        f"candidates={len(cands)}"
    )

    for i, c in enumerate(cands):
        key = str(c.get("fact_key") or f"row-{i}")
        status = str(c.get("status") or "pending")
        with st.expander(
            f"[{status}] score={c.get('score')} · {key} · {str(c.get('statement') or '')[:48]}",
            expanded=(status == "pending"),
        ):
            st.write(c.get("statement") or "")
            st.json(c.get("score_breakdown") or {})
            for mid in c.get("evidence_message_ids") or []:
                mid_i = int(mid)
                st.markdown(f"- `{mid_i}`: {evid.get(mid_i, '(missing)')}")

            cols = st.columns(3)
            if cols[0].button("Approve", key=f"fact-apr-{key}-{i}"):
                c["status"] = "approved"
                upsert_fact_candidate(conn, actor_key, c)
                remove_fact_block(conn, actor_key, key)
                conn.commit()
                st.rerun()
            if cols[1].button("Reject", key=f"fact-rej-{key}-{i}"):
                c["status"] = "rejected"
                upsert_fact_candidate(conn, actor_key, c)
                add_fact_block(conn, actor_key, key)
                conn.commit()
                st.rerun()
            if cols[2].button("Reset pending", key=f"fact-rst-{key}-{i}"):
                c["status"] = "pending"
                upsert_fact_candidate(conn, actor_key, c)
                remove_fact_block(conn, actor_key, key)
                conn.commit()
                st.rerun()


def _render_db_facts(conn, actor_key: str) -> None:
    st.subheader("Database facts (`stg.persona_facts`)")
    confirm_delete = st.checkbox(
        "Enable hard Delete buttons (destructive)",
        value=False,
        key="confirm_hard_delete",
    )
    rows = fetch_facts(conn, actor_key, status=None)
    if not rows:
        st.info("No rows in stg.persona_facts for this actor.")
        return

    for kind, subset in (
        ("a", [r for r in rows if r.get("status") == "active"]),
        ("i", [r for r in rows if r.get("status") != "active"]),
    ):
        st.markdown("#### Active" if kind == "a" else "#### Inactive / other")
        if not subset:
            st.write("(none)")
            continue
        for r in subset:
            fk = str(r.get("fact_key") or "")
            status = str(r.get("status") or "")
            with st.expander(f"[{status}] {fk} · {str(r.get('statement') or '')[:48]}", expanded=False):
                st.write(r.get("statement") or "")
                cols = st.columns(3)
                if status == "active":
                    if cols[0].button("Disable", key=f"dis-{kind}-{fk}"):
                        set_fact_status(conn, actor_key, fk, "inactive")
                        st.rerun()
                else:
                    if cols[0].button("Enable", key=f"en-{kind}-{fk}"):
                        set_fact_status(conn, actor_key, fk, "active")
                        st.rerun()
                if confirm_delete and cols[1].button("Delete forever", key=f"del-{kind}-{fk}"):
                    delete_fact(conn, actor_key, fk)
                    st.rerun()


def _render_topic_candidates(data: dict[str, Any], conn, actor_key: str) -> None:
    cands = list(data.get("candidates") or [])
    if not cands:
        st.info("No topic candidates yet. Click Propose one topic.")
        return

    all_ids: list[int] = []
    for c in cands:
        all_ids.extend(int(x) for x in (c.get("example_message_ids") or []))
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        # Load full window texts for review (not only the first 20).
        all_ids.extend(int(x) for x in (meta.get("window_ids") or []))
    evid = _load_evidence_map(conn, sorted(set(all_ids)))

    pending = [c for c in cands if str(c.get("status") or "") == "pending"]
    other = [c for c in cands if str(c.get("status") or "") != "pending"]

    st.caption(
        f"storage={data.get('storage')} · pending={len(pending)} · other={len(other)} · "
        f"last_mode={(data.get('mode') or data.get('last_result') or '?')}"
    )

    def _render_one(c: dict[str, Any], i: int, *, expanded: bool) -> None:
        key = str(c.get("fact_key") or f"topic-{i}")
        status = str(c.get("status") or "pending")
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        title = str(meta.get("title_zh") or c.get("phrase") or key)
        with st.expander(
            f"[{status}] {title} · kind={c.get('kind')} · score={c.get('score')}",
            expanded=expanded,
        ):
            st.json(
                {
                    "fact_key": key,
                    "kind": c.get("kind"),
                    "match_any": c.get("match_any"),
                    "prefer_any": c.get("prefer_any"),
                    "why": meta.get("why"),
                    "confidence": meta.get("confidence"),
                    "bucket": meta.get("bucket"),
                    "source": c.get("source"),
                }
            )
            st.caption("Evidence message ids (LLM-picked)")
            evid_ids = [int(x) for x in (c.get("example_message_ids") or [])]
            if evid_ids:
                for mid_i in evid_ids:
                    st.markdown(f"- `{mid_i}`: {evid.get(mid_i, '(missing)')}")
            else:
                st.write("(none)")

            window_ids = [int(x) for x in (meta.get("window_ids") or [])]
            st.caption(
                f"Full chat window sent to LLM (n={len(window_ids)}, bucket={meta.get('bucket') or '?'})"
            )
            if window_ids:
                # Show the whole window (role + text); scroll via expander if long.
                with st.expander("Window transcript", expanded=False):
                    for mid_i in window_ids:
                        st.markdown(f"- `{mid_i}`: {evid.get(mid_i, '(missing)')}")
            else:
                st.warning("No window_ids stored on this candidate; Redo may sample a new window.")

            if status != "pending":
                return

            st.text_input(
                "Optional redo feedback",
                key=f"topic-fb-{key}-{i}",
                placeholder="Why this proposal is wrong (sent on Redo)",
            )
            cols = st.columns(3)
            if cols[0].button("Approve", type="primary", key=f"topic-apr-{key}-{i}"):
                out = dict(c)
                out["status"] = "approved"
                upsert_topic_candidate(conn, actor_key, out)
                conn.commit()
                code = topics_cmd_apply(
                    conn,
                    actor_key=actor_key,
                    dry_run=False,
                    fact_keys=[key],
                )
                if code != 0:
                    st.warning(f"Approve apply failed: {code}")
                st.rerun()
            if cols[1].button("Redo with LLM", key=f"topic-redo-{key}-{i}"):
                # Same window when stored; otherwise falls back to a new random window.
                redo_ids = [int(x) for x in (meta.get("window_ids") or []) if x]
                with st.spinner("Redoing topic with LLM..."):
                    code, info = cmd_llm_propose_one(
                        conn,
                        actor_key=actor_key,
                        kobold_url=st.session_state.get(
                            "t_kobold",
                            os.environ.get("KOBOLD_URL", "http://127.0.0.1:5001/v1"),
                        ),
                        model=st.session_state.get("t_model", ""),
                        gap_sec=int(st.session_state.get("t_gap", 3 * 3600)),
                        max_window_msgs=int(st.session_state.get("t_maxwin", 50)),
                        min_window_msgs=int(st.session_state.get("t_minwin", 6)),
                        max_tokens=int(st.session_state.get("t_maxtok", 400)),
                        temperature=max(
                            0.5, float(st.session_state.get("t_temp", 0.3))
                        ),
                        dry_run=False,
                        redo_fact_key=key,
                        window_ids=redo_ids or None,
                        feedback=str(
                            st.session_state.get(f"topic-fb-{key}-{i}", "") or ""
                        ),
                    )
                st.session_state["t_last_propose"] = {"code": code, "info": info}
                st.rerun()
            if cols[2].button("Reject", key=f"topic-rej-{key}-{i}"):
                out = dict(c)
                out["status"] = "rejected"
                upsert_topic_candidate(conn, actor_key, out)
                conn.commit()
                st.rerun()

    for i, c in enumerate(pending):
        _render_one(c, i, expanded=True)
    if other:
        with st.expander(f"Resolved candidates ({len(other)})", expanded=False):
            for j, c in enumerate(other):
                _render_one(c, 1000 + j, expanded=False)


def _render_official_topics(conn, actor_key: str) -> None:
    st.subheader("Official topics (`stg.persona_topic_specs`)")
    topics = list_topic_specs(conn, actor_key, enabled_only=False)
    if not topics:
        st.info("No official topics yet. Propose → Approve.")
        return
    for i, spec in enumerate(topics):
        fk = str(spec.get("fact_key") or f"t-{i}")
        enabled = bool(spec.get("enabled", True))
        title_bits = ", ".join(str(x) for x in (spec.get("match_any") or [])[:3])
        with st.expander(
            f"[{'on' if enabled else 'off'}] {fk} · kind={spec.get('kind', '?')} · {title_bits}",
            expanded=False,
        ):
            st.json(
                {
                    k: spec.get(k)
                    for k in (
                        "fact_key",
                        "kind",
                        "match_any",
                        "prefer_any",
                        "min_len",
                        "max_len",
                        "enabled",
                        "source",
                    )
                }
            )
            c1, c2, c3 = st.columns(3)
            if enabled:
                if c1.button("Disable", key=f"toff-{fk}-{i}"):
                    set_topic_spec_enabled(conn, actor_key, fk, False)
                    conn.commit()
                    st.rerun()
            else:
                if c1.button("Enable", key=f"ton-{fk}-{i}"):
                    set_topic_spec_enabled(conn, actor_key, fk, True)
                    conn.commit()
                    st.rerun()
            if c2.button("Delete", key=f"tdel-{fk}-{i}"):
                delete_topic_spec(conn, actor_key, fk)
                conn.commit()
                st.rerun()


def _tab_topics(conn, actor_key: str) -> None:
    st.write(
        "LLM proposes one topic slot from a random time-bucketed chat window. "
        "Human gate: Approve (writes official), Redo, or Reject. "
        "Official topics: Disable / Delete only. No manual slot editing."
    )
    kobold = st.text_input(
        "Kobold URL",
        value=os.environ.get("KOBOLD_URL", "http://127.0.0.1:5001/v1"),
        key="t_kobold",
    )
    model = st.text_input(
        "Model override (empty = first /models id)",
        value=os.environ.get("KOBOLD_MODEL", ""),
        key="t_model",
    )
    c_a, c_b, c_c, c_d = st.columns(4)
    gap_sec = c_a.number_input("session gap_sec", min_value=600, max_value=86400, value=10800, key="t_gap")
    min_win = c_b.number_input("min window msgs", min_value=3, max_value=40, value=6, key="t_minwin")
    max_win = c_c.number_input("max window msgs", min_value=10, max_value=120, value=50, key="t_maxwin")
    temperature = c_d.number_input(
        "temperature", min_value=0.0, max_value=1.5, value=0.3, step=0.05, key="t_temp"
    )
    max_tokens = st.number_input("max_tokens", min_value=64, max_value=1200, value=400, key="t_maxtok")

    b1, b2 = st.columns(2)
    propose_clicked = b1.button("Propose one topic", type="primary", key="t_propose")
    reload_clicked = b2.button("Reload", key="t_reload")

    if propose_clicked:
        with st.spinner("Sampling window + calling Kobold..."):
            code, info = cmd_llm_propose_one(
                conn,
                actor_key=actor_key,
                kobold_url=kobold,
                model=model,
                gap_sec=int(gap_sec),
                max_window_msgs=int(max_win),
                min_window_msgs=int(min_win),
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                dry_run=False,
            )
        st.session_state["t_last_propose"] = {"code": code, "info": info}
        if code == 0:
            st.success(f"Pending topic `{info.get('fact_key')}`")
        elif code == 1:
            st.info("LLM found no durable topic in that window. Click again.")
        elif code == 3:
            st.warning(f"Duplicate of existing slot: {info.get('detail')}. Click again.")
        else:
            st.error(f"Propose failed: {info}")
        st.rerun()

    last = st.session_state.get("t_last_propose")
    if last and not propose_clicked:
        st.caption(f"Last propose code={last.get('code')} info_keys={list((last.get('info') or {}).keys())}")

    _render_official_topics(conn, actor_key)
    data = load_topics_pending_view(conn, actor_key)
    st.subheader("Pending topic candidates")
    if reload_clicked:
        st.caption("Reloaded from Postgres.")
    _render_topic_candidates(data, conn, actor_key)


def _compute_voice_metrics(conn, actor_key: str, limit: int = 12000) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text_content
            FROM stg.messages
            WHERE actor_key = %s
              AND speaker_role = 'actor'
              AND has_semantic_text = TRUE
              AND char_length(text_content) BETWEEN 1 AND 120
            ORDER BY create_time_epoch DESC NULLS LAST
            LIMIT %s
            """,
            (actor_key, limit),
        )
        texts = [(r["text_content"] or "").strip() for r in cur.fetchall()]
    n = max(1, len(texts))
    lens = [len(t) for t in texts]
    sticker_n = sum(1 for t in texts if "[" in t and "]" in t)
    laugh_n = sum(1 for t in texts if "哈" in t)
    return {
        "messages_scanned": len(texts),
        "avg_len": round(sum(lens) / n, 2),
        "short_le_8_ratio": round(sum(1 for L in lens if L <= 8) / n, 4),
        "long_ge_40_ratio": round(sum(1 for L in lens if L >= 40) / n, 4),
        "sticker_msg_ratio": round(sticker_n / n, 4),
        "laugh_msg_ratio": round(laugh_n / n, 4),
    }


def _draft_voice_notes(metrics: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    avg = float(metrics.get("avg_len") or 0)
    short = float(metrics.get("short_le_8_ratio") or 0)
    sticker = float(metrics.get("sticker_msg_ratio") or 0)
    laugh = float(metrics.get("laugh_msg_ratio") or 0)
    if short >= 0.6:
        notes.append("Prefers very short chat bubbles")
    elif avg <= 12:
        notes.append("Keeps messages short; not essay-like")
    else:
        notes.append("Comfortable with mid-length chat lines")
    if sticker >= 0.05:
        notes.append("Often uses WeChat stickers/emoji brackets in text")
    else:
        notes.append("Rarely uses sticker/emoji bracket tokens")
    if laugh >= 0.04:
        notes.append("Laugh particles appear often in casual turns")
    notes.append("Tone from style samples only; do not invent biography")
    return notes


def _tab_voice(conn, actor_key: str) -> None:
    st.write(
        "Voice/style channel (not topics/facts). Metrics are automatic; voice_notes "
        "persist on `stg.persona_cards` and feed chat tone."
    )
    metrics = _compute_voice_metrics(conn, actor_key)
    st.subheader("Metrics")
    st.json(metrics)

    card = load_persona_card_db(conn, actor_key) or {
        "display_name": actor_key,
        "voice_notes": [],
        "boundaries": [],
        "extra_rules": [],
        "known_facts": [],
    }
    draft = _draft_voice_notes(metrics)
    st.subheader("Voice notes")
    st.caption("Edit and save. Draft from metrics is a starting point only.")
    if st.button("Fill from metrics draft", key="v_fill"):
        st.session_state["voice_notes_text"] = "\n".join(draft)
    existing = card.get("voice_notes") or []
    default_text = st.session_state.get(
        "voice_notes_text",
        "\n".join(str(x) for x in existing) if existing else "\n".join(draft),
    )
    notes_text = st.text_area("One note per line", value=default_text, height=180, key="v_notes")
    if st.button("Save voice notes to persona card", type="primary", key="v_save"):
        notes = [ln.strip() for ln in notes_text.splitlines() if ln.strip()]
        card = dict(card)
        card["voice_notes"] = notes
        if not card.get("display_name"):
            card["display_name"] = actor_key
        upsert_persona_card(conn, actor_key, card)
        conn.commit()
        st.session_state["voice_notes_text"] = "\n".join(notes)
        st.success("Saved to stg.persona_cards")

    st.subheader("Style sample peek")
    st.caption("Random actor lines used by chat (tone only). Approve flow can deepen later.")
    from chat_persona import fetch_style_lines

    samples = fetch_style_lines(conn, actor_key, 8, 4, 40)
    for s in samples:
        st.write(f"- {s}")


def _tab_facts(conn, actor_key: str) -> None:
    st.write(
        "Extract fact candidates using official Postgres topics, then apply to "
        "`stg.persona_facts`."
    )
    limit = st.number_input("Fact extract limit", min_value=1, max_value=50, value=10, key="f_limit")
    since_days = st.number_input("since_days (0=all)", min_value=0, max_value=3650, value=0, key="f_since")
    min_score = st.number_input("min_score", min_value=0.0, max_value=100.0, value=0.0, key="f_score")
    include_active = st.checkbox("Include already-active fact_keys", value=False, key="f_active")
    include_rejected = st.checkbox("Include previously rejected fact_keys", value=False, key="f_rej")
    diversify = st.checkbox("Diversify statement pick", value=False, key="f_div")
    ensure = st.checkbox("Ensure schema on extract/apply", value=True, key="f_ensure")

    c1, c2, c3 = st.columns(3)
    extract_clicked = c1.button("Extract facts ~N", type="primary", key="f_extract")
    reload_clicked = c2.button("Reload fact pending", key="f_reload")
    apply_clicked = c3.button("Apply approved facts", key="f_apply")

    if extract_clicked:
        try:
            topics = resolve_topics(conn, actor_key)
        except SystemExit as e:
            st.error(str(e))
            return
        if ensure:
            ensure_schema(conn, _REPO)
        with st.spinner("Extracting facts..."):
            code = facts_cmd_extract(
                conn,
                actor_key=actor_key,
                topics=topics,
                evidence_limit=5,
                limit=int(limit),
                since_days=int(since_days),
                include_active_keys=bool(include_active),
                include_rejected_keys=bool(include_rejected),
                min_score=float(min_score),
                diversify=bool(diversify),
                seed=0,
                dry_run=False,
            )
        st.success("Fact extract finished.") if code == 0 else st.warning(f"code={code}")

    if apply_clicked:
        if ensure:
            ensure_schema(conn, _REPO)
        with st.spinner("Applying approved facts..."):
            code = facts_cmd_apply(
                conn,
                actor_key=actor_key,
                replace_active=False,
                dry_run=False,
            )
        if code == 0:
            st.success("Fact apply finished.")
        elif code == 1:
            st.warning("Nothing to apply (no status=approved).")
        else:
            st.error(f"Fact apply failed with code {code}")

    _render_db_facts(conn, actor_key)
    data = load_facts_pending_view(conn, actor_key)
    st.subheader("Pending fact candidates")
    if reload_clicked:
        st.caption("Reloaded from Postgres.")
    _render_fact_candidates(data, conn, actor_key)


def main() -> None:
    st.set_page_config(page_title="Persona Review", layout="wide")
    st.title("Persona topics & facts review")
    st.write(
        "Governance data lives in Postgres (`stg.persona_topic_*`, "
        "`stg.persona_fact_candidates`, `stg.persona_cards`). "
        "Locale / LoRA / eval fixtures stay on disk."
    )

    with st.sidebar:
        st.header("Settings")
        st.session_state["database_url"] = st.text_input(
            "DATABASE_URL (optional override)",
            value=__import__("os").environ.get("DATABASE_URL", ""),
            type="password",
        )

    conn = _connect()
    if conn is None:
        return

    try:
        ensure_governance_schema(conn, _REPO)
        env_actor = __import__("os").environ.get("ACTOR_KEY", "guodahong")
        actors = _discover_actor_keys(conn)
        if env_actor and env_actor not in actors:
            actors = [env_actor] + actors
        default_ix = actors.index(env_actor) if env_actor in actors else 0
        with st.sidebar:
            actor_key = st.selectbox("actor_key", options=actors, index=default_ix)
            st.session_state["actor_key"] = actor_key

        tab_topics, tab_facts, tab_voice = st.tabs(["Topics", "Facts", "Voice"])
        with tab_topics:
            _tab_topics(conn, actor_key)
        with tab_facts:
            _tab_facts(conn, actor_key)
        with tab_voice:
            _tab_voice(conn, actor_key)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

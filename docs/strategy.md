# Development strategy

Living plan for persona simulation on top of cleaned chat history.
English-only; private corpora and Chinese match strings stay under `data/private/`.

## Goal

Roleplay a real person with **grounded** memory and a separable **voice/style** channel.
Prefer abstaining over inventing biography. Style must not outrun factual gates.
LoRA remains optional after grounding is trustworthy.

## Architecture (agreed)

Three **channels** (do not mix):

| Channel | What | Store | Human gate |
|---------|------|-------|------------|
| **Topics** | Schema slots to mine facts into | `stg.persona_topic_specs` (+ candidates) | LLM propose → Approve/Redo/Reject; official Disable/Delete |
| **Facts** | Normalized propositions | `stg.persona_facts` (+ fact candidates/blocks) | Facts tab / CLI |
| **Voice / style** | How they speak (not what is true) | `stg.persona_cards` + style samples | Voice tab |
| **Episodic** | Chat snippets on demand | `stg.messages` via RAG | None (retrieval) |

```text
stg.messages
    │
    ├─► Topics LLM propose-one (time-bucket window) ──Approve──► persona_topic_specs
    │         │
    │         └─► Facts extract ──approve──► persona_facts
    │
    ├─► Style / voice markers ──► Voice channel only (never topics/facts)
    │
    ├─► Style samples + voice_notes ──► chat prompt (tone)
    │
    └─► Keyword RAG (→ hybrid later)
```

**Topics populate (agreed):** Local Kobold reads one random session window (month-bucketed). Returns 0–1 pending slot. Human Approve / Redo / Reject only — no manual field edits, no n-gram stop mine as the discovery path. Rejected rows stay as reject-memory for later prompts. Official: Disable / Delete.

**Intake Q&A** (planned): fill chat-absent profile slots; writes the **same** facts table with `source=human_intake|qa_llm_draft` and empty evidence allowed.

Governance runtime data lives in Postgres (`sql/005_persona_governance.sql`).
**Stay on disk by design:** locales, LoRA exports, eval suites/reports, samples, optional stopword config files.

### Habit phrases vs topics (agreed)

- High-DF discourse glue is **not a topic**.
- With LLM topic propose, glue is filtered by the model + human Reject (reject-memory), not by a stop-phrase mine.
- Relative style (bubble length, sticker density, laugh rate) belongs in **Voice**, never in `fact_key` rows.

### Stop / block

Legacy `persona_topic_blocks` / n-gram extract may remain in code for debugging but are **retired from the Topics UX**. Do not reintroduce stop-mining as the primary topic path.

### Voice / style (planned implementation)

New Streamlit **Voice** tab (alongside Topics / Facts) — products are **not** facts:

1. **Metrics (auto, read-only):** length, short-bubble rate, laugh/sticker density → draft `voice_notes`.
2. **Style samples (semi-auto):** approve/reject short real lines for `chat_persona` tone examples.
3. **Voice notes (editable):** persist on `stg.persona_cards`.

Chat continues to load cards from Postgres (JSON file fallback only for legacy).

### Provenance rules

| Source | Evidence | Notes |
|--------|----------|--------|
| Chat extract | Prefer non-empty `evidence_message_ids` | Statement must be a normalized claim, not a raw bubble |
| Human / Q&A intake | Empty evidence OK | Must set `source` so chat never pretends it was said in history |
| Future “live” interest sim | Own source + optional external refs | Out of band from chat evidence |

Reject handling: rejected keys/phrases stay blocked by default; support reconsider/unblock.

## Principles

- Schema-first MVP with human gates (studio / high-trust).
- Deterministic extract by default; optional diversify only when exploring.
- Ban-list prompt stacking is an antipattern; prefer grounded generation + structured facts.
- Separate **style channel** from **memory channel**.
- Agents do not commit/push; private data stays gitignored.
- Agree before large code changes.

## Response to known gaps

| Gap | Stance |
|-----|--------|
| Topics/facts only cover chat | Accept for now. Later Phase L: “living” curiosity with non-chat `source`. |
| No auto memory at session end | **Deferred (Phase S).** Human-reviewed dialogue memory only. |
| Quote-as-fact | Phase F maturation target (propositional rewrite). |
| Topic mine returns glue noise | Tighten DF/stop; route habits to stop/voice — not topics. |

## Next maturation targets

### Near-term coding order (agreed)

```text
A. Topic Reject → Add as stop (+ clear noisy pending)
B. Voice tab: metrics + voice_notes + sample approve → chat
C. Optional Stop mine queue if A is not enough
D. Phase F: propositional facts + kind chat split + intake Q&A
```

### Phase F — Propositional facts

`statement` must be a third-person (or named) proposition, not a WeChat bubble.

```text
evidence → draft_statement (LLM or template) → human approve → DB statement
```

- Propagate `kind` / clearer `source` on pending and DB.
- Chat: `profile` always-on; soft on demand.
- Intake Q&A for empty profile slots (same facts table).

## Phased roadmap

### Done (keep maintaining)

- Raw ingest → voice stage → Whisper ASR → `stg.messages`
- Persona chat: style samples + keyword RAG + facts + locale + **Postgres persona cards**
- Facts pending / apply / Streamlit Facts tab (**Postgres** `persona_fact_candidates`)
- Topics mine / approve / apply / Streamlit Topics tab (**Postgres** `persona_topic_*`)
- Topics LLM propose-one (Kobold) + Approve/Redo/Reject; official Disable/Delete
- Governance DDL + JSON→PG import: `sql/005_persona_governance.sql`, `migrate_private_json_to_pg.py`
- Rejected-key blocking; optional diversify; LoRA pair export (train optional)

### Phase V — Voice & LLM topics (in progress)

- Topics populate: Kobold propose-one from time-bucketed random windows; Approve / Redo / Reject; official Disable / Delete
- n-gram topic mine + stop UX retired from primary path (legacy CLI only)
- Voice tab: metrics + voice_notes on `stg.persona_cards` (+ style sample peek)

### Phase F — Facts maturation

- Propositional rewrite; `kind` inject; intake Q&A

### Phase M — Memory ops

- Reconsider/unblock; soft TTL; extract vs intake conflicts

### Phase S / L / R / C

- Session memory (human-reviewed) → living interest sim → hybrid retrieval → optional LoRA train

## Non-goals

- Messaging-client reverse engineering as product core  
- Silent auto-write of low-quality session text into long-term facts  
- Treating discourse glue as topics or facts  
- Ban-list-only “safety” instead of grounding  
- Committing private chat or `.env`

## Status snapshot

See [README.md](../README.md) for commands. Strategy truth lives here; update when phases complete or priorities change.

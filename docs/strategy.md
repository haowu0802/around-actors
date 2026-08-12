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
| **Topics** | Schema slots to mine facts into | `stg.persona_topic_specs` (+ candidates/blocks) | Topics tab / CLI |
| **Facts** | Normalized propositions | `stg.persona_facts` (+ fact candidates/blocks) | Facts tab / CLI |
| **Voice / style** | How they speak (not what is true) | `stg.persona_cards` + style samples (planned) | Voice tab (planned) |
| **Episodic** | Chat snippets on demand | `stg.messages` via RAG | None (retrieval) |

```text
stg.messages
    │
    ├─► Topics mine ──approve──► persona_topic_specs
    │         │
    │         └─► Facts extract ──approve──► persona_facts
    │
    ├─► Habit / glue phrases ──► STOP (block topic&fact mine)
    │                         └─► Voice markers (style only; never facts)
    │
    ├─► Style samples + voice_notes ──► chat prompt (tone)
    │
    └─► Keyword RAG (→ hybrid later)
```

**Intake Q&A** (planned): fill chat-absent profile slots; writes the **same** facts table with `source=human_intake|qa_llm_draft` and empty evidence allowed.

Governance runtime data lives in Postgres (`sql/005_persona_governance.sql`).
**Stay on disk by design:** locales, LoRA exports, eval suites/reports, samples, optional stopword config files.

### Habit phrases vs topics (agreed)

Industry practice (Letta persona vs archival, style profiling vs RAG facts): 

- High-DF discourse glue (`不是` / `就是` / `我觉得`…) is **not a topic**.
- It is either **stop/block** for topic&fact extract, or a **voice marker** for style.
- Shared Mandarin glue barely distinguishes actors; **relative** style (bubble length, sticker density, laugh rate) does — e.g. xi heavy WeChat stickers + longer bubbles vs guodahong shorter bubbles + almost no stickers.

Never create `fact_key` rows for speech habits.

### Stop / block (planned implementation)

Semi-manual, **lighter** than topics (no English fact_key required; phrase is the key).

1. **MVP:** Topic Reject → one-click **Add as stop** → `persona_topic_blocks` (and/or dedicated stop table) consumed by topic&fact mine.
2. **Later:** optional Stop mine queue (high-DF candidates) with labels: `stop` | `voice_marker` | `discard`.
3. Stops are actor-scoped (and optionally shared locale lists under `data/private/`).

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
- Governance DDL + JSON→PG import: `sql/005_persona_governance.sql`, `migrate_private_json_to_pg.py`
- Rejected-key blocking; optional diversify; LoRA pair export (train optional)

### Phase V — Voice & stops (next product focus)

- Stop/block MVP from Topic Reject; consume in mine
- Voice tab (metrics, samples, voice_notes on `persona_cards`)
- Tighten topic mine (max DF, prefer contentful phrases)

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

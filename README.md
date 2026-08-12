# Around Actors

Turn chat export JSON into Postgres tables and a local LLM setup that roleplays a person using a persona card, structured facts, and retrieved memories.

Stack focus: data pipelines + local LLM serving/eval. Not messaging-client reverse engineering.

## MVP

For cleaned conversations (one or more actors):

1. Persona card + structured facts + episodic retrieval from chat text  
2. Local OpenAI-compatible chat: persona prompt, and persona + RAG (+ facts)  
3. Small hold-out / rule eval comparing modes  

Not in early MVP: silent auto-memory every session, client decryption, heavy production UI, voice/image generation models, chat bots on messaging apps. LoRA is optional after grounding is solid.

## Memory model (short)

- **Topics / pending / persona cards / fact pending**: Postgres (`sql/005_persona_governance.sql`)
- **Facts (applied)**: `stg.persona_facts`
- **Still on disk (by design)**: locales, LoRA exports, eval suites/reports, samples
- **RAG**: episodic chat snippets on demand

Full plan: [docs/strategy.md](docs/strategy.md).

## Status

| Layer | State |
|-------|--------|
| Raw ingest | `scripts/ingest_raw.py` → `raw.messages` |
| Voice + ASR | `stg.voice_messages` + Whisper scripts |
| Unified chat staging | `stg.messages` |
| Persona chat | `scripts/chat_persona.py` (style + keyword RAG + facts + locale) |
| Facts pending / apply | `scripts/build_persona_facts.py` |
| Fact review UI | `scripts/fact_review_app.py` (Streamlit) |
| Eval | `scripts/eval_persona.py` + private suites |
| LoRA pairs | `scripts/build_lora_pairs.py` (export only; train optional) |
| Topics governance | Postgres `stg.persona_topic_specs` + candidates; Streamlit Topics tab |
| Facts pending | Postgres `stg.persona_fact_candidates` |
| Persona cards | Postgres `stg.persona_cards` (JSON fallback) |
| Propositional rewrite | Next after Topics loop is trusted (Phase F) |

## Quick commands

```bash
pip install -r requirements.txt
cp .env.example .env   # set DATABASE_URL, optional KOBOLD_URL

python scripts/ingest_raw.py --ensure-schema --replace \
  --file "/path/to/export.json" \
  --actor-key person_a

python scripts/chat_persona.py --actor-key person_a --show-rag

python scripts/build_persona_topics.py --actor-key person_a --mode llm-propose
python scripts/build_persona_facts.py --actor-key person_a --mode extract --limit 10
streamlit run scripts/fact_review_app.py
```

`EXPORT_FILE`, `ACTOR_KEY`, `DATABASE_URL`, and `KOBOLD_URL` also work as env vars. Do not hard-code private paths in the repo.

DDL highlights: `sql/001_raw_messages.sql`, voice/staging SQL under `sql/`, facts in `sql/004_stg_persona_facts.sql`.

## Privacy

Do not commit real chat logs. Keep corpora local and consented. Ship only anonymized or synthetic samples in git.

Details: [docs/privacy.md](docs/privacy.md).

## Local-only files

`.env`, dumps, indexes, `data/private/**` stay on the machine (see `.gitignore`).

## Next (priority)

1. **Phase V** — Grow official topics via LLM Propose one; Voice notes on cards  
2. **Phase F** — propositional facts + `kind` chat split + intake Q&A  
3. Deferred: session memory; living interest sim; hybrid retrieval; optional LoRA train  

Details: [docs/strategy.md](docs/strategy.md).

## License

TBD.

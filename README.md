# Around Actors

Turn chat export JSON into Postgres tables and, later, a local LLM setup that roleplays a person using a persona card and retrieved memories.

Stack focus: data pipelines + local LLM serving/eval. Not messaging-client reverse engineering.

## MVP

For one cleaned conversation:

1. Persona card + memory index from chat text
2. Local OpenAI-compatible chat with two modes: persona prompt, and persona + RAG
3. Small hold-out comparison of the two modes

Not in MVP: LoRA, client decryption, heavy UI, voice/image models, chat bots on messaging apps.

## Status

Raw ingest: `scripts/ingest_raw.py` → Postgres `raw.messages`. Clean / persona / chat / eval: not built yet.

## Raw ingest

```bash
pip install -r requirements.txt
cp .env.example .env   # set DATABASE_URL

python scripts/ingest_raw.py --ensure-schema --replace \
  --file "/path/to/export.json" \
  --actor-key person_a
```

`EXPORT_FILE`, `ACTOR_KEY`, and `DATABASE_URL` also work as env vars. Do not hard-code private paths in the repo.

DDL: `sql/001_raw_messages.sql` (export fields 1:1 plus `actor_key`, `source_file`, `ingested_at`).

## Privacy

Do not commit real chat logs. Keep corpora local and consented. Ship only anonymized or synthetic samples in git.

Details: [docs/privacy.md](docs/privacy.md).

## Local-only files

`.env`, dumps, and indexes stay on the machine (see `.gitignore`).

## Next

1. Clean layer; optional voice/ASR notes
2. Persona card + embeddings
3. Local chat + short eval writeup

## License

TBD.

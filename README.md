# Around Actors

Local-first toolkit that turns real chat exports into a **persona-conditioned LLM actor**: a small data pipeline plus prompt/RAG serving, so a locally hosted model replies a bit more like a specific person.

This repository is a portfolio-oriented implementation of that idea—focused on **data platform engineering** and **LLM systems**, not on reverse-engineering messaging clients.

## MVP (what this aims to prove)

Given one cleaned conversation export:

1. Build an editable **persona card** and a **memory index** from chat text.
2. Chat with a **local** OpenAI-compatible model (e.g. Ollama) under two strategies:
   - **Persona prompt** (card + few-shot exemplars)
   - **Persona + RAG** (same, plus retrieved memories)
3. Run a **tiny hold-out eval** and show whether RAG helps.

Out of scope for the MVP: model fine-tuning (LoRA), chat-client decryption, heavy UI, multimodal (voice ASR / images), and production messaging bots.

## Status

Early scaffolding. Runtime pipelines are not implemented yet.

## Privacy

**Do not commit real chat logs.** Use only consenting, personal corpora locally. The repo will ship a tiny anonymized or synthetic sample when code lands—never your full export.

See [docs/privacy.md](docs/privacy.md).

## Local data (not in git)

Point the app at your export directory via environment config once tooling exists. Keep dumps, indexes, and `.env` on your machine only (covered by `.gitignore`).

## Roadmap (high level)

1. Ingest & clean exporter JSON → columnar cleaned messages  
2. Persona card + embedding index  
3. Local chat (prompt vs RAG)  
4. Minimal A/B eval report  

## License

TBD.

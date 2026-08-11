# Privacy

## Principles

- **Consent:** Only use chat data you have a right to use (e.g. your own conversations with people who would reasonably accept this use).
- **Local-first:** Prefer running models and indexes on your machine. Do not upload raw chat dumps to third-party training services.
- **No impersonation:** Outputs approximate style and recalled context. They are not the real person and must not be used to deceive others.
- **Minimize publication:** Public artifacts should be anonymized samples, schemas, metrics, and code—not full private histories.

## What belongs in this repository

- Source code, configs that contain no secrets, docs, and small synthetic/anonymized fixtures.
- Evaluation numbers and architecture write-ups.

## What must stay private

- Real chat export JSON/CSV/Parquet dumps
- Persona cards derived from real people (unless fully anonymized and you accept publication risk)
- Vector indexes built from real chats
- API keys, `.env`, and machine-local paths to personal data

These paths are ignored by `.gitignore`. Double-check before every commit.

## Attribution and expectations

Style similarity on WeChat-style short text is limited. The system may still invent facts if retrieval fails; treat answers as assisted roleplay, not ground truth.

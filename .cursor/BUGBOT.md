# Bugbot review rules — around-actors

Project review guidance for Cursor Bugbot. Coding-agent process rules (git commit/push bans, step-gate agreement) are **not** Bugbot concerns; those stay in `.cursor/rules/`.

## Language and secrets

- Flag new Chinese (or other non-English) in committed docs, comments, UI chrome, or sample filenames meant for the public repo. Private corpora under `data/private/` are gitignored and out of scope unless a commit accidentally includes them.
- Flag secrets or local identifiers in the diff: passwords, full `DATABASE_URL` with credentials, wxids, absolute machine paths, real export filenames that identify people, raw chat dumps.

## Architecture / product invariants

- Keep **memory** (topics/facts) separate from **voice/style**. Discourse glue and speech habits must not become `fact_key` / topic slots or `stg.persona_facts` rows.
- Topics populate is LLM propose-one + human Approve/Redo/Reject; do not reintroduce n-gram stop-mining as the primary topic discovery path.
- Runtime governance for topics, pending queues, and persona cards belongs in Postgres (`stg.persona_*`). Do not reintroduce JSON as the source of truth for those without an explicit migration plan.
- Locale templates, LoRA exports, and eval fixtures may stay on disk; do not require them to move into PG in a drive-by change.

## Code quality

- Prefer focused diffs. Flag unrelated refactors bundled with feature work.
- Flag dead CLI flags, ignored overrides (`--topics-file`, `--persona-card`), and orphan DB rows after renames/updates.
- Flag empty `except:` / broad swallow that hides governance schema or load failures without logging.
- Match existing English-only docstrings and bland engineering tone; avoid marketing filler in user-facing strings.

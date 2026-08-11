# Privacy

## Rules of use

- Only process chats you are allowed to use.
- Prefer local models and local storage. Do not upload raw dumps to third-party training APIs.
- Model output is approximate roleplay, not the real person. Do not use it to impersonate anyone.
- Publish code, schemas, metrics, and anonymized samples—not full private histories.

## OK in git

- Source, secret-free config examples, docs, tiny synthetic/anonymized fixtures
- Eval numbers and architecture notes

## Keep private

- Real export files
- Real-person persona cards (unless you fully anonymize and accept the risk)
- Indexes built from real chats
- API keys, `.env`, absolute paths to personal data

Check `.gitignore` before every commit.

## Limits

Short chat text is a weak style signal. Retrieval misses can still produce invented facts. Treat answers as assisted draft replies, not ground truth.

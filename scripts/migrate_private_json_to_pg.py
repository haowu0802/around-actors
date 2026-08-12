#!/usr/bin/env python3
"""One-shot: create governance tables and import legacy private JSON into Postgres."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_persona_facts import load_dotenv  # noqa: E402
from persona_store import ensure_governance_schema, import_private_tree  # noqa: E402


def main() -> int:
    repo = _SCRIPTS.parent
    load_dotenv(repo / ".env")
    p = argparse.ArgumentParser(description="Import private JSON governance data into Postgres")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args()
    if not args.database_url:
        print("DATABASE_URL required", file=sys.stderr)
        return 2
    with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
        ensure_governance_schema(conn, repo)
        stats = import_private_tree(conn, repo)
    print("governance_schema=ok")
    print(f"imported={stats}")
    print(
        "Note: locale / lora / eval stay on disk. "
        "Legacy JSON left in place as backup; runtime now prefers Postgres."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

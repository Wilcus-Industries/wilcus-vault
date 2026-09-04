"""Shared corpus and helpers for the indexer tests."""

import sqlite3
from pathlib import Path

from wilcus_vault.db import db_path, open_db
from wilcus_vault.embed import TokenOverlapEmbedder

embedder = TokenOverlapEmbedder(32)

FIXTURE = {
    "notes/acme.md": "---\ntype: customer\n---\n# Acme Corp\n\n"
    "renewal, see [[globex]] and [[ghost]]\n",
    "notes/globex.md": "# Globex\n\nvendor policy\n",
    "notes/lonely.md": "# Lonely\n\nno links at all\n",
    ".obsidian/skip.md": "# Skipped\n\n[[acme]]\n",
    "notes/not-markdown.txt": "ignored",
}

# A vault where `customers/acme.md` is the only `acme` and two notes link it bare.
COLLISION = {
    "customers/acme.md": "# Acme the customer\n",
    "hub.md": "# Hub\n\nsee [[acme]] for the account\n",
    "notes/deal.md": "# Deal\n\nclosing [[acme|Acme Corp]] this week\n",
}


def open_index(root: Path) -> sqlite3.Connection:
    return open_db(db_path(root))


def one(db: sqlite3.Connection, sql: str, *params: object) -> object:
    """The single value of a one-column, one-row query (None when there is no row)."""
    row = db.execute(sql, params).fetchone()
    return None if row is None else row[0]


def id_of(db: sqlite3.Connection, path: str) -> int:
    value = one(db, "select id from notes where path = ?", path)
    assert isinstance(value, int)
    return value


def read_file(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")

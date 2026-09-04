"""The SQLite index. Every row here is derived from the `.md` files and disposable."""

import re
import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from .embed import Embedder, Vector
from .term import VaultError

_SCHEMA = """
create table if not exists notes (
  id integer primary key,
  path text not null unique,
  slug text not null,
  title text not null,
  type text,
  hash text not null,
  frontmatter text not null,
  superseded_by text,
  mtime integer not null,
  malformed integer not null default 0
);
create index if not exists notes_slug on notes(slug);
-- to_slug is the link target as written: a bare stem or a path. to_id null
-- means the link is broken or ambiguous.
create table if not exists edges (
  from_id integer not null,
  to_slug text not null,
  to_id integer,
  unique(from_id, to_slug)
);
create index if not exists edges_to_id on edges(to_id);
create virtual table if not exists notes_fts using fts5(title, body);
create table if not exists vector_meta (
  note_id integer primary key,
  model text not null,
  dims integer not null
);
"""


def db_path(root: str | Path) -> Path:
    return Path(root) / ".vault" / "index.db"


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating dirs and file) with sqlite-vec loaded and the schema applied."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, autocommit=True)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("pragma journal_mode = wal")  # watcher, CLI and library callers share it
    db.execute("pragma busy_timeout = 5000")
    db.executescript(_SCHEMA)
    return db


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[None]:
    db.execute("begin")
    try:
        yield
    except BaseException:
        db.execute("rollback")
        raise
    db.execute("commit")


def to_blob(v: Vector) -> bytes:
    """A vector as the float32 bytes vec0 stores."""
    return struct.pack(f"{len(v)}f", *v)


def from_blob(blob: bytes) -> Vector:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _check_dims(embedder: Embedder) -> None:
    # dims are interpolated into the vec0 DDL (they cannot be bound), so an
    # injected embedder must not get to write SQL.
    dims = embedder.dims
    if isinstance(dims, bool) or not isinstance(dims, int) or not 1 <= dims <= 8192:
        raise VaultError(f"embedder {embedder.model}: dims must be an integer in 1..8192")


def vectors_stale(db: sqlite3.Connection, embedder: Embedder) -> bool:
    """Do the stored vectors belong to another model or width? Read-only on purpose:
    the caller asks this before embedding, and must still have the old vectors if
    embedding fails."""
    _check_dims(embedder)
    stale_model = (
        db.execute(
            "select 1 from vector_meta where model <> ? or dims <> ? limit 1",
            (embedder.model, embedder.dims),
        ).fetchone()
        is not None
    )
    existing = db.execute("select sql from sqlite_master where name = 'vectors'").fetchone()
    if existing is None:
        return stale_model
    match = re.search(r"float\[(\d+)\]", existing["sql"])
    current_dims = int(match.group(1)) if match else 0
    return stale_model or current_dims != embedder.dims


def reset_vectors(db: sqlite3.Connection, embedder: Embedder, stale: bool) -> None:
    """Bring the vec0 table into line with `embedder`.

    `stale` drops the table and its meta, which is destructive: call it inside
    the transaction that writes the replacement vectors. Otherwise the table is
    only created if missing.
    """
    _check_dims(embedder)
    if stale:
        db.execute("drop table if exists vectors")
        db.execute("delete from vector_meta")
    db.execute(
        "create virtual table if not exists vectors using vec0("
        f"note_id integer primary key, emb float[{embedder.dims}] distance_metric=cosine)"
    )

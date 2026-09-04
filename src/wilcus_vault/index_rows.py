"""Every index row a note owns, written and purged in one place."""

import json
import sqlite3
from dataclasses import dataclass

from .db import to_blob
from .embed import Embedder, l2_normalize
from .note import Note


@dataclass
class Dirty:
    note: Note
    mtime: int
    id: int | None  # None for a note the index has never seen


def write_note(db: sqlite3.Connection, d: Dirty, vector: list[float], embedder: Embedder) -> None:
    note = d.note
    superseded_by = note.frontmatter.get("superseded_by")
    row = db.execute(
        """insert into notes
             (path, slug, title, type, hash, frontmatter, superseded_by, mtime, malformed)
             values (?, ?, ?, ?, ?, ?, ?, ?, ?)
           on conflict(path) do update set
             slug = excluded.slug, title = excluded.title, type = excluded.type,
             hash = excluded.hash, frontmatter = excluded.frontmatter,
             superseded_by = excluded.superseded_by, mtime = excluded.mtime,
             malformed = excluded.malformed
           returning id""",
        (
            note.path,
            note.slug,
            note.title,
            note.type,
            note.hash,
            json.dumps(note.frontmatter),
            superseded_by if isinstance(superseded_by, str) else None,
            d.mtime,
            int(note.malformed_frontmatter),
        ),
    ).fetchone()
    note_id = row["id"]
    db.execute("delete from notes_fts where rowid = ?", (note_id,))
    db.execute(
        "insert into notes_fts (rowid, title, body) values (?, ?, ?)",
        (note_id, note.title, note.body),
    )
    db.execute("delete from edges where from_id = ?", (note_id,))
    for target in note.links:
        db.execute(
            "insert or ignore into edges (from_id, to_slug) values (?, ?)", (note_id, target)
        )
    db.execute("delete from vectors where note_id = ?", (note_id,))
    # An all-zero vector (no tokens the embedder knows) has no direction and
    # would poison KNN with NaN distances. Skip the row; the note stays findable
    # through FTS, and the meta row still records the attempt.
    emb = l2_normalize(vector)
    if any(x != 0 for x in emb):
        db.execute("insert into vectors (note_id, emb) values (?, ?)", (note_id, to_blob(emb)))
    db.execute(
        "insert or replace into vector_meta (note_id, model, dims) values (?, ?, ?)",
        (note_id, embedder.model, embedder.dims),
    )


def purge_note(db: sqlite3.Connection, note_id: int) -> None:
    """Drop every derived row for a note whose file is gone."""
    for sql in (
        "delete from notes where id = ?",
        "delete from notes_fts where rowid = ?",
        "delete from edges where from_id = ?",
        "delete from vectors where note_id = ?",
        "delete from vector_meta where note_id = ?",
    ):
        db.execute(sql, (note_id,))


def resolve_edges(db: sqlite3.Connection) -> None:
    """Resolve every wikilink. A target with a `/` is a path and matches exactly
    one note or none; a bare stem resolves only when exactly one note carries it.
    Recomputed wholesale, because adding or removing any note can flip links in
    notes that did not change."""
    db.execute(
        """update edges set to_id = case
             when instr(to_slug, '/') > 0
               then (select n.id from notes n where n.path = edges.to_slug || '.md')
               else (select min(n.id) from notes n where n.slug = edges.to_slug having count(*) = 1)
           end"""
    )

"""Files are truth: the scan is the input, the database is the output.

Dirtiness is decided by content hash, never by the database's own bookkeeping,
and one function (`index_paths`) writes every index row.
"""

import json
import os
import sqlite3
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .db import reset_vectors, to_blob, transaction, vectors_stale
from .embed import Embedder, l2_normalize
from .note import Note, parse_note
from .qualify import Qualified, detect_collisions, qualify_collisions
from .term import VaultError, printable

# Rewrites per collision before the rest is left to doctor. A generous fuse
# against a pathological vault, not a tuning knob.
QUALIFY_CAP = 500


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    reembedded: bool = False  # the model or dims changed, so every note was re-embedded
    qualified: list[Qualified] = field(default_factory=list)  # stem collisions this pass handled
    # The re-index of rewritten linkers failed; their rows lag the files until
    # the next pass. Reported, not raised, so the stats still say what changed.
    index_error: str | None = None


@dataclass
class _Dirty:
    note: Note
    mtime: int
    id: int | None  # None for a note the index has never seen


def scan_vault(root: str | Path) -> list[str]:
    """Vault-relative paths of every `.md` file, sorted. Dot-directories are
    skipped and symlinks are never followed."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and not (here / d).is_symlink()]
        for name in filenames:
            path = here / name
            if name.startswith(".") or not name.endswith(".md") or path.is_symlink():
                continue
            if path.is_file():
                out.append(path.relative_to(root).as_posix())
    return sorted(out)


def is_note_path(rel: str) -> bool:
    """Would the scan index this path: a `.md` file with no dot-segment in it?"""
    segments = rel.replace("\\", "/").split("/")
    return rel.endswith(".md") and not any(s.startswith(".") for s in segments)


def note_entry(root: str | Path, rel: str) -> os.stat_result | None:
    """The lstat of a path that really holds a note, or None. Only a regular file
    is a note: a path that is gone, a directory, or a symlink holds none."""
    try:
        entry = os.lstat(Path(root) / rel)
    except FileNotFoundError:
        return None
    return entry if stat.S_ISREG(entry.st_mode) else None


def read_raw(root: str | Path, rel: str) -> str | None:
    """The file's text, or None if it is no longer there. Other errors raise."""
    try:
        return (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def read_note(root: str | Path, rel: str) -> Note | None:
    raw = read_raw(root, rel)
    return None if raw is None else parse_note(raw, rel)


async def reindex(db: sqlite3.Connection, root: str | Path, embedder: Embedder) -> IndexStats:
    """Hash-diff the whole vault against the index and write only what changed."""
    # Every path the files know plus every path the index knows: the ones only
    # the index has are deletions, and index_paths purges them.
    indexed = [r["path"] for r in db.execute("select path from notes")]
    stats = await index_paths(db, root, embedder, [*scan_vault(root), *indexed])
    # Unconditional, unlike inside index_paths: an index written by an older
    # version may hold to_id values an older resolution rule produced.
    resolve_edges(db)
    return stats


async def index_paths(
    db: sqlite3.Connection,
    root: str | Path,
    embedder: Embedder,
    rels: Iterable[str],
    qualify_cap: int = QUALIFY_CAP,
) -> IndexStats:
    """Hash-diff exactly these paths and write only what changed; a path whose
    file is gone is purged. When the pass creates a stem collision, bare links
    to the incumbent are rewritten to its qualified form after the commit."""
    root = Path(root)
    # Asked here, acted on inside the write transaction with the new vectors in hand.
    reembedded = vectors_stale(db, embedder)
    rows = db.execute("select id, path, hash, slug from notes").fetchall()
    by_path = {r["path"]: r for r in rows}
    embedded = {r["note_id"] for r in db.execute("select note_id from vector_meta")}

    dirty: list[_Dirty] = []
    gone: list[int] = []
    unchanged = 0
    for rel in dict.fromkeys(rels):
        row = by_path.get(rel)
        # What is at the path is decided before the read: only a regular file
        # is a note. A directory or symlink there is a deletion.
        entry = note_entry(root, rel)
        note = None if entry is None else read_note(root, rel)
        if note is None or entry is None:
            if row is not None:
                gone.append(row["id"])
            continue
        # A row with no vector is half-indexed (interrupted run): redo it.
        if row and row["hash"] == note.hash and not reembedded and row["id"] in embedded:
            unchanged += 1
            continue
        dirty.append(_Dirty(note, int(entry.st_mtime * 1000), row["id"] if row else None))

    new_notes = [d.note for d in dirty if d.id is None]
    collisions = detect_collisions(root, rows, new_notes, set(gone))

    texts = [f"{d.note.title}\n\n{d.note.body}" for d in dirty]
    vectors = await embedder.embed(texts) if texts else []
    if len(vectors) != len(dirty):
        raise VaultError(
            f"embedder {embedder.model} returned {len(vectors)} vectors for {len(dirty)} texts"
        )
    for v in vectors:
        if len(v) != embedder.dims:
            raise VaultError(
                f"embedder {embedder.model} returned a vector of width {len(v)}, "
                f"expected {embedder.dims}"
            )

    with transaction(db):
        reset_vectors(db, embedder, reembedded)
        for d, vector in zip(dirty, vectors, strict=True):
            _write_note(db, d, vector, embedder)
        for note_id in gone:
            purge_note(db, note_id)
        # Resolution depends only on the note set, so an unchanged pass stays writeless.
        if dirty or gone:
            resolve_edges(db)

    new_paths = {n.path for n in new_notes}
    qualified, rewritten, index_error = qualify_collisions(
        db, root, collisions, new_paths, qualify_cap
    )
    # Re-enter on the rewritten paths so the index never lags our own writes.
    # None of them is new, so no further collision is detected.
    if rewritten:
        try:
            await index_paths(db, root, embedder, rewritten)
        except Exception as e:
            index_error = index_error or printable(e)

    updated = {d.note.path for d in dirty if d.id is not None} | set(rewritten)
    return IndexStats(
        added=len(new_notes),
        updated=len(updated),
        removed=len(gone),
        unchanged=unchanged,
        reembedded=reembedded,
        qualified=qualified,
        index_error=index_error,
    )


def _write_note(db: sqlite3.Connection, d: _Dirty, vector: list[float], embedder: Embedder) -> None:
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

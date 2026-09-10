"""The watcher driven by `touch`: debounce, batching, and filtering."""

import hashlib
import sqlite3
from pathlib import Path

from conftest import MakeVault, vec_of, write_note
from watchfiles import Change

from wilcus_vault.db import db_path, open_db
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.indexer import IndexStats, is_note_path, reindex
from wilcus_vault.watch import WatchOptions, _note_events, watch

EMBEDDER = TokenOverlapEmbedder(32)

FIXTURE = {
    "notes/acme.md": "# Acme Corp\n\nrenewal, see [[globex]]\n",
    "notes/globex.md": "# Globex\n\nvendor policy\n",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def note_row(db: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = db.execute(
        "select id, hash from notes where path = ?", (path,)
    ).fetchone()
    return row


def count(db: sqlite3.Connection, sql: str, *params: object) -> int:
    return int(db.execute(sql, params).fetchone()[0])


def test_is_note_path_md_files_only_never_through_a_dot_directory() -> None:
    assert is_note_path("notes/acme.md")
    assert is_note_path("acme.md")
    # the index itself lives in .vault/: the watcher must not feed itself
    assert not is_note_path(".vault/index.db")
    assert not is_note_path(".obsidian/workspace.md")
    assert not is_note_path(".git/COMMIT_EDITMSG.md")
    assert not is_note_path(".hidden.md")
    assert not is_note_path("notes/photo.png")
    assert not is_note_path("notes")


# The edits below land *before* `watch` starts, so the only events in these
# tests are the `touch` calls themselves; a live fs watch on the same root would
# otherwise race its own events into the flush sequence.
async def test_a_burst_on_one_path_is_debounced_into_a_single_index_pass(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        write_note(root, "notes/acme.md", "# Acme Corp\n\nedited five times\n")
        flushes: list[list[str]] = []
        opts = WatchOptions(debounce=0.02, on_change=lambda paths, _s: flushes.append(paths))
        w = watch(db, root, EMBEDDER, opts)
        for _ in range(5):
            w.touch("notes/acme.md")
        await w.idle()
        assert flushes == [["notes/acme.md"]]
        row = note_row(db, "notes/acme.md")
        assert row is not None and row["hash"] == sha256("# Acme Corp\n\nedited five times\n")
        await w.close()
    finally:
        db.close()


async def test_paths_ready_together_are_indexed_in_one_pass(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        write_note(root, "notes/acme.md", "# Acme Corp\n\none\n")
        write_note(root, "notes/globex.md", "# Globex\n\ntwo\n")
        flushes: list[list[str]] = []
        opts = WatchOptions(debounce=0.01, on_change=lambda paths, _s: flushes.append(paths))
        w = watch(db, root, EMBEDDER, opts)
        w.touch("notes/acme.md")
        w.touch("notes/globex.md")
        await w.idle()
        assert flushes == [["notes/acme.md", "notes/globex.md"]]
        await w.close()
    finally:
        db.close()


async def test_touch_ignores_anything_the_scan_would_not_index(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        flushes: list[list[str]] = []
        opts = WatchOptions(debounce=0.001, on_change=lambda paths, _s: flushes.append(paths))
        w = watch(db, root, EMBEDDER, opts)
        # the index's own files live under .vault/: acting on them would feed
        # the watcher its own tail
        w.touch(".vault/index.db")
        w.touch(".vault/index.db-wal")
        w.touch(".obsidian/workspace.md")
        w.touch("notes/photo.png")
        await w.idle()
        assert flushes == []
        await w.close()
    finally:
        db.close()


async def test_the_watcher_indexes_a_create_an_edit_and_a_delete(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        w = watch(db, root, EMBEDDER, WatchOptions(debounce=0.005))

        # create: row, vector and a wikilink edge that resolves
        write_note(root, "notes/new.md", "# New\n\npoints at [[acme]]\n")
        w.touch("notes/new.md")
        await w.idle()
        created = note_row(db, "notes/new.md")
        assert created is not None
        assert len(vec_of(db, created["id"])) == 32
        acme = note_row(db, "notes/acme.md")
        assert acme is not None
        edge = db.execute(
            "select to_id from edges where from_id = ? and to_slug = 'acme'", (created["id"],)
        ).fetchone()
        assert edge["to_id"] == acme["id"]

        # edit: hash and vector both move
        before = vec_of(db, created["id"])
        write_note(root, "notes/new.md", "# New\n\nrewritten with entirely different words\n")
        w.touch("notes/new.md")
        await w.idle()
        edited = note_row(db, "notes/new.md")
        assert edited is not None and edited["hash"] != created["hash"]
        assert vec_of(db, created["id"]) != before

        # delete: every derived row goes with the file
        (Path(root) / "notes" / "new.md").unlink()
        w.touch("notes/new.md")
        await w.idle()
        assert note_row(db, "notes/new.md") is None
        assert count(db, "select count(*) from vectors where note_id = ?", created["id"]) == 0
        assert count(db, "select count(*) from notes_fts where rowid = ?", created["id"]) == 0
        await w.close()
    finally:
        db.close()


async def test_an_unchanged_file_is_reported_but_rewrites_nothing(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        seen: list[IndexStats] = []
        opts = WatchOptions(debounce=0.001, on_change=lambda _p, s: seen.append(s))
        w = watch(db, root, EMBEDDER, opts)
        before = db.total_changes
        w.touch("notes/acme.md")  # editor rewrote identical bytes
        await w.idle()
        assert [(s.added, s.unchanged) for s in seen] == [(0, 1)]
        assert db.total_changes == before
        await w.close()
    finally:
        db.close()


def test_the_index_never_feeds_the_watcher_its_own_tail() -> None:
    """`.vault/` is where the index writes, and in WAL mode it writes constantly.
    Those events are dropped before they reach Python, not after."""
    assert _note_events(Change.modified, "/v/notes/acme.md")
    assert _note_events(Change.added, "/v/deep/nested/note.md")
    assert not _note_events(Change.modified, "/v/.vault/index.db")
    assert not _note_events(Change.modified, "/v/.vault/index.db-wal")
    assert not _note_events(Change.modified, "/v/.vault/index.db-shm")
    assert not _note_events(Change.added, "/v/.discarded.log")

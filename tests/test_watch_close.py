"""Closing the watcher, model swaps under it, and real filesystem events."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from conftest import MakeVault, vec_of, write_note
from fakes import stub_embedder
from test_watch import EMBEDDER, FIXTURE, count, note_row, sha256

from wilcus_vault.db import db_path, open_db
from wilcus_vault.embed import TokenOverlapEmbedder, Vector
from wilcus_vault.indexer import IndexStats, reindex
from wilcus_vault.vault import open as open_vault
from wilcus_vault.watch import WatchOptions, watch


async def until[T](what: str, fn: Callable[[], T | None], timeout: float = 8.0) -> T:
    """Poll until `fn` returns something truthy. Real fs events have no promise
    to await, so the end-to-end test waits for the *effect* with a deadline
    generous enough that a loaded machine is not a failing one."""
    deadline = time.monotonic() + timeout
    while True:
        value = fn()
        if value:
            return value
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


async def test_close_waits_for_the_pass_in_flight_and_starts_no_more(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        held = asyncio.Event()

        async def slow_embed(texts: list[str]) -> list[Vector]:
            await held.wait()
            return await EMBEDDER.embed(texts)

        slow = stub_embedder("token-overlap-v1", 32, slow_embed)
        acme_text = "# Acme Corp\n\nindexed by the pass in flight\n"
        write_note(root, "notes/acme.md", acme_text)
        write_note(root, "notes/globex.md", "# Globex\n\nqueued behind it, never indexed\n")

        w = watch(db, root, slow, WatchOptions(debounce=0.001))
        w.touch("notes/acme.md")
        await asyncio.sleep(0.02)  # the pass is now inside embed()
        w.touch("notes/globex.md")  # queues behind it
        await asyncio.sleep(0.02)

        closing = w.close()
        held.set()
        await closing
        settled = db.total_changes

        # closing resolves only once the database is nobody's: a caller may now
        # close it without racing a write
        await asyncio.sleep(0.05)
        assert db.total_changes == settled
        acme = note_row(db, "notes/acme.md")
        assert acme is not None and acme["hash"] == sha256(acme_text)
        # the queued path never ran: what a close drops, doctor picks up
        globex = note_row(db, "notes/globex.md")
        assert globex is not None and globex["hash"] == sha256(FIXTURE["notes/globex.md"])
    finally:
        db.close()


async def test_a_model_swap_under_the_watcher_reembeds_the_whole_vault(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)  # 32 dims
        reported: list[IndexStats] = []
        wider = TokenOverlapEmbedder(64)
        opts = WatchOptions(debounce=0.001, on_change=lambda _p, s: reported.append(s))
        w = watch(db, root, wider, opts)
        w.touch("notes/acme.md")
        await w.idle()
        assert reported[-1].reembedded is True
        assert count(db, "select count(*) from vectors") == 2
        assert count(db, "select count(*) from vector_meta where dims = 64") == 2
        await w.close()
    finally:
        db.close()


async def test_a_model_swap_does_not_lose_the_qualify_report(make_vault: MakeVault) -> None:
    root = make_vault({"customers/acme.md": "# Acme\n", "hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)  # 32 dims
        # the same pass both creates a stem collision and detects the dims
        # change: the follow-up whole-vault reindex must not replace the
        # qualify report, the only place that says files were rewritten
        reported: list[IndexStats] = []
        wider = TokenOverlapEmbedder(64)
        opts = WatchOptions(debounce=0.001, on_change=lambda _p, s: reported.append(s))
        w = watch(db, root, wider, opts)
        write_note(root, "vendors/acme.md", "# Acme two\n")
        w.touch("vendors/acme.md")
        await w.idle()
        assert reported[-1].reembedded is True
        assert [q.stem for q in reported[-1].qualified] == ["acme"]
        await w.close()
    finally:
        db.close()


async def test_close_stops_the_watcher_later_events_are_dropped(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        await reindex(db, root, EMBEDDER)
        w = watch(db, root, EMBEDDER, WatchOptions(debounce=0.001))
        w.close()  # not awaited: closing takes effect immediately

        write_note(root, "notes/acme.md", "# Acme Corp\n\nafter close\n")
        w.touch("notes/acme.md")
        await asyncio.sleep(0.02)
        await w.idle()
        row = note_row(db, "notes/acme.md")
        assert row is not None and row["hash"] == sha256(FIXTURE["notes/acme.md"])
    finally:
        db.close()


async def test_real_fs_events_an_edit_a_create_and_a_delete_reach_the_index(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    vault = open_vault(root, EMBEDDER)
    await vault.reindex()
    w = vault.watch(WatchOptions(debounce=0.025))
    # a second connection: reading what the watcher's own handle writes
    db = open_db(db_path(root))
    try:
        # The fs watch is set up by a task, so it needs a turn of the loop before
        # it can see anything written below.
        await asyncio.sleep(0.2)
        acme = note_row(db, "notes/acme.md")
        assert acme is not None
        vec_before = vec_of(db, acme["id"])

        write_note(root, "notes/acme.md", "# Acme Corp\n\nrewritten on disk by a human\n")

        def edited() -> bool:
            row = note_row(db, "notes/acme.md")
            return row is not None and row["hash"] != acme["hash"]

        await until("the edit to be indexed", edited)
        assert vec_of(db, acme["id"]) != vec_before

        write_note(root, "notes/fresh.md", "# Fresh\n\nwritten straight into the vault\n")
        fresh = await until("the new note to be indexed", lambda: note_row(db, "notes/fresh.md"))
        assert len(vec_of(db, fresh["id"])) == 32

        (Path(root) / "notes" / "fresh.md").unlink()
        await until("the deletion to be indexed", lambda: note_row(db, "notes/fresh.md") is None)
        assert count(db, "select count(*) from vectors where note_id = ?", fresh["id"]) == 0
    finally:
        w.close()
        await w.idle()
        db.close()
        vault.close()

"""Watcher error handling: a failing embedder, and a failing error reporter."""

from conftest import MakeVault, write_note
from fakes import stub_embedder
from test_watch import EMBEDDER, FIXTURE, note_row, sha256

from wilcus_vault.db import db_path, open_db
from wilcus_vault.embed import Vector
from wilcus_vault.indexer import reindex
from wilcus_vault.watch import WatchOptions, watch


class Flaky:
    """An embedder that fails while `down`, otherwise embeds like EMBEDDER."""

    def __init__(self) -> None:
        self.down = False

    async def embed(self, texts: list[str]) -> list[Vector]:
        if self.down:
            raise RuntimeError("provider down")
        return await EMBEDDER.embed(texts)


async def test_an_embedder_failure_is_reported_and_the_watcher_keeps_working(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        provider = Flaky()
        flaky = stub_embedder("flaky-v1", 32, provider.embed)
        await reindex(db, root, flaky)
        row = note_row(db, "notes/acme.md")
        assert row is not None
        stale = row["hash"]
        errors: list[BaseException] = []
        w = watch(db, root, flaky, WatchOptions(debounce=0.001, on_error=errors.append))
        provider.down = True
        text = "# Acme Corp\n\nedited while the provider is down\n"
        write_note(root, "notes/acme.md", text)
        w.touch("notes/acme.md")
        await w.idle()
        assert "provider down" in str(errors[0])
        # the pass failed, and it failed without taking the watcher down with it
        row = note_row(db, "notes/acme.md")
        assert row is not None and row["hash"] == stale

        provider.down = False
        w.touch("notes/acme.md")
        await w.idle()
        row = note_row(db, "notes/acme.md")
        assert row is not None and row["hash"] == sha256(text)
        await w.close()
    finally:
        db.close()


async def test_a_reporter_that_throws_does_not_wedge_the_watcher(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_db(db_path(root))
    try:
        provider = Flaky()
        flaky = stub_embedder("flaky-v1", 32, provider.embed)
        await reindex(db, root, flaky)
        write_note(root, "notes/acme.md", "# Acme Corp\n\nfails to embed\n")
        reports = 0

        def broken_reporter(_e: BaseException) -> None:
            nonlocal reports
            reports += 1
            raise RuntimeError("the error reporter is broken too")

        w = watch(db, root, flaky, WatchOptions(debounce=0.001, on_error=broken_reporter))
        provider.down = True
        w.touch("notes/acme.md")
        await w.idle()  # must resolve: a throwing reporter must not leave the pass in flight
        assert reports == 1

        # and the next change still gets indexed
        provider.down = False
        write_note(root, "notes/globex.md", "# Globex\n\nindexes fine\n")
        w.touch("notes/globex.md")
        await w.idle()
        row = note_row(db, "notes/globex.md")
        assert row is not None and row["hash"] == sha256("# Globex\n\nindexes fine\n")
        await w.close()
    finally:
        db.close()

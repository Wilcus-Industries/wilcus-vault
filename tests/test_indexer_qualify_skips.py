"""Auto-qualify never guesses: every uncertain or failed linker lands in `skipped`."""

import os

from conftest import MakeVault, write_note
from fakes import stub_embedder
from indexer_common import COLLISION, embedder, open_index, read_file

from wilcus_vault.embed import Vector
from wilcus_vault.indexer import index_paths, reindex
from wilcus_vault.qualify import Qualified


async def test_an_incumbent_path_that_cannot_round_trip_a_wikilink_is_reported(
    make_vault: MakeVault,
) -> None:
    # `[` and `]` are legal filename characters; spliced into `[[...]]` they
    # destroy the link (`[[[archive]/acme]]` parses as no edge at all).
    root = make_vault({"[archive]/acme.md": "# Acme\n", "hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "vendors/acme.md", "# Acme two\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    assert stats.qualified == [Qualified("acme", None, [], ["hub.md"])]
    assert read_file(root, "hub.md") == "# Hub\n\n[[acme]]\n"
    db.close()

    # whitespace is the other thing the parser does to a target (it trims), so
    # a leading-space directory cannot round-trip either: `[[ archive/acme]]`
    # parses back as `archive/acme`, a different (possibly existing) note
    root2 = make_vault({" archive/acme.md": "# Acme\n", "hub.md": "# Hub\n\n[[acme]]\n"})
    db2 = open_index(root2)
    await reindex(db2, root2, embedder)
    write_note(root2, "vendors/acme.md", "# Acme two\n")
    stats2 = await index_paths(db2, root2, embedder, ["vendors/acme.md"])
    assert stats2.qualified == [Qualified("acme", None, [], ["hub.md"])]
    assert read_file(root2, "hub.md") == "# Hub\n\n[[acme]]\n"
    db2.close()


async def test_an_incumbent_whose_file_is_gone_is_no_collision(make_vault: MakeVault) -> None:
    # A move seen by the watcher in two passes: the create arrives first, the
    # delete has not been indexed yet. Rewriting to the stale row's path would
    # break every bare link vault-wide on the next pass.
    root = make_vault({"a/acme.md": "# Acme\n", "hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    (root / "a" / "acme.md").unlink()
    write_note(root, "b/acme.md", "# Acme\n")
    stats = await index_paths(db, root, embedder, ["b/acme.md"])
    assert stats.qualified == []
    assert read_file(root, "hub.md") == "# Hub\n\n[[acme]]\n"
    db.close()


async def test_an_unwritable_linking_note_is_skipped_and_reported(make_vault: MakeVault) -> None:
    root = make_vault({"customers/acme.md": "# Acme\n", "ro/hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "vendors/acme.md", "# Acme two\n")
    os.chmod(root / "ro", 0o555)
    try:
        # one unwritable linker must not throw away the whole pass's stats
        stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
        assert stats.qualified == [Qualified("acme", "customers/acme", [], ["ro/hub.md"])]
    finally:
        os.chmod(root / "ro", 0o755)
    assert read_file(root, "ro/hub.md") == "# Hub\n\n[[acme]]\n"
    db.close()


async def test_vanished_and_new_in_pass_linkers_land_in_skipped(make_vault: MakeVault) -> None:
    root = make_vault(COLLISION)
    db = open_index(root)
    await reindex(db, root, embedder)
    # hub.md vanishes without its deletion being indexed, and a brand-new note
    # links the stem bare in the same pass as the collision
    (root / "hub.md").unlink()
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    write_note(root, "notes/new.md", "# New\n\n[[acme]]\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md", "notes/new.md"])
    assert stats.qualified == [
        Qualified("acme", "customers/acme", ["notes/deal.md"], ["hub.md", "notes/new.md"])
    ]
    # the new note's own bare link has no settled meaning: left for doctor
    assert read_file(root, "notes/new.md") == "# New\n\n[[acme]]\n"
    db.close()


async def test_a_confined_path_failure_on_a_linker_is_skipped_not_raised(
    make_vault: MakeVault,
) -> None:
    # A linker's parent directory is replaced by a symlink after it was indexed
    # (a stale row): confined_path must not throw out of the pass after an
    # earlier linker was already rewritten — it is routed into skipped instead.
    root = make_vault(
        {
            "customers/acme.md": "# Acme the customer\n",
            "sub/hub.md": "# Hub\n\nsee [[acme]] for the account\n",
            "notes/deal.md": "# Deal\n\nclosing [[acme|Acme Corp]] this week\n",
        }
    )
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    sub, real = root / "sub", root / "sub-real"
    sub.rename(real)
    sub.symlink_to(real)
    try:
        stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    finally:
        sub.unlink()
        real.rename(sub)
    assert stats.qualified == [
        Qualified("acme", "customers/acme", ["notes/deal.md"], ["sub/hub.md"])
    ]
    assert stats.index_error is not None and "symlink" in stats.index_error
    assert (
        read_file(root, "notes/deal.md")
        == "# Deal\n\nclosing [[customers/acme|Acme Corp]] this week\n"
    )
    db.close()


async def test_an_incumbent_failing_confinement_is_not_a_collision_not_raised(
    make_vault: MakeVault,
) -> None:
    # The incumbent's own parent directory is replaced by a symlink after it
    # was indexed (a stale row): confined_path must not throw out of
    # detect_collisions — the identical condition is already a silent
    # non-collision one line below (an incumbent whose file is gone). A linker
    # to the stem makes this discriminating: without the confinement check,
    # the symlinked incumbent still passes the is_file()-and-not-symlink()
    # test below it, so it would be treated as a real collision and hub.md
    # would get rewritten.
    root = make_vault(
        {
            "sub/acme.md": "# Acme the customer\n",
            "hub.md": "# Hub\n\nsee [[acme]] for the account\n",
        }
    )
    db = open_index(root)
    await reindex(db, root, embedder)
    sub, real = root / "sub", root / "sub-real"
    sub.rename(real)
    sub.symlink_to(real)
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    try:
        stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    finally:
        sub.unlink()
        real.rename(sub)
    assert stats.qualified == []
    assert stats.index_error is None
    assert read_file(root, "hub.md") == "# Hub\n\nsee [[acme]] for the account\n"
    db.close()


async def test_a_re_entry_failure_after_rewrites_is_reported_not_raised(
    make_vault: MakeVault,
) -> None:
    # The embedder dies exactly when it sees the rewritten text; the files have
    # already changed, so the caller must still get the stats saying which ones.
    async def flaky_embed(texts: list[str]) -> list[Vector]:
        if any("[[customers/acme]]" in t for t in texts):
            raise RuntimeError("embed endpoint down")
        return [[1.0] * 8 for _ in texts]

    flaky = stub_embedder("flaky-v1", 8, flaky_embed)
    root = make_vault(COLLISION)
    db = open_index(root)
    await reindex(db, root, flaky)
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    stats = await index_paths(db, root, flaky, ["vendors/acme.md"])
    assert stats.qualified[0].rewritten == ["hub.md", "notes/deal.md"]
    assert stats.index_error is not None and "embed endpoint down" in stats.index_error
    assert read_file(root, "hub.md") == "# Hub\n\nsee [[customers/acme]] for the account\n"
    db.close()

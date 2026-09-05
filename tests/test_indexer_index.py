"""The scan, the hash-diff and what each pass writes or purges."""

import json
import math
import os
import re

import pytest
from conftest import MakeVault, vec_of, write_note
from fakes import stub_embedder
from indexer_common import FIXTURE, embedder, id_of, one, open_index

from wilcus_vault.doctor import doctor
from wilcus_vault.embed import Vector
from wilcus_vault.indexer import IndexStats, index_paths, read_note, reindex, scan_vault
from wilcus_vault.term import VaultError


def test_scan_md_only_dot_dirs_skipped_symlinks_never_followed(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    other = make_vault({"secret.md": "# Secret\n"})
    os.symlink(other, root / "linked-dir")
    os.symlink(other / "secret.md", root / "linked.md")
    assert scan_vault(root) == ["notes/acme.md", "notes/globex.md", "notes/lonely.md"]


async def test_reindex_writes_notes_fts_vectors_and_edges(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    stats = await reindex(db, root, embedder)
    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (3, 0, 0, 0)

    acme = db.execute("select * from notes where path = 'notes/acme.md'").fetchone()
    assert acme["slug"] == "acme"
    assert acme["title"] == "Acme Corp"
    assert acme["type"] == "customer"
    assert json.loads(acme["frontmatter"]) == {"type": "customer"}
    assert acme["mtime"] > 0
    assert acme["malformed"] == 0

    # FTS row is keyed by note id and searchable
    assert one(db, "select rowid from notes_fts where notes_fts match 'renewal'") == acme["id"]

    # vector: one per note, unit length, meta records model + dims
    assert one(db, "select count(*) from vectors") == 3
    v = vec_of(db, acme["id"])
    assert len(v) == 32
    assert math.hypot(*v) == pytest.approx(1, abs=1e-5)
    meta = db.execute("select model, dims from vector_meta where note_id = ?", (acme["id"],))
    assert tuple(meta.fetchone()) == (embedder.model, 32)

    # edges: unique stem resolves, unknown stem stays null
    edges = db.execute(
        "select to_slug, to_id from edges where from_id = ? order by to_slug", (acme["id"],)
    ).fetchall()
    assert [tuple(e) for e in edges] == [("ghost", None), ("globex", id_of(db, "notes/globex.md"))]
    db.close()


async def test_a_pass_with_nothing_changed_writes_nothing(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    await reindex(db, root, embedder)
    before = db.total_changes
    # the watcher's entry point: an editor saving identical bytes must not
    # dirty the database
    stats = await index_paths(db, root, embedder, scan_vault(root))
    assert stats == IndexStats(unchanged=3)
    assert db.total_changes == before
    db.close()


async def test_edits_update_in_place_deletes_purge_every_table(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    await reindex(db, root, embedder)
    note_id = id_of(db, "notes/acme.md")

    write_note(root, "notes/acme.md", "# Acme Corp\n\nnow points at [[lonely]] only\n")
    stats = await reindex(db, root, embedder)
    assert (stats.added, stats.updated, stats.unchanged) == (0, 1, 2)
    assert id_of(db, "notes/acme.md") == note_id  # stable identity
    slugs = db.execute("select to_slug from edges where from_id = ?", (note_id,)).fetchall()
    assert [r["to_slug"] for r in slugs] == ["lonely"]

    (root / "notes" / "acme.md").unlink()
    stats = await reindex(db, root, embedder)
    assert (stats.removed, stats.unchanged) == (1, 2)
    assert one(db, "select count(*) from notes") == 2
    for table, key in (
        ("edges", "from_id"),
        ("vectors", "note_id"),
        ("vector_meta", "note_id"),
        ("notes_fts", "rowid"),
    ):
        assert one(db, f"select count(*) from {table} where {key} = ?", note_id) == 0
    db.close()


async def test_malformed_frontmatter_still_indexes_and_is_flagged(make_vault: MakeVault) -> None:
    root = make_vault({"bad.md": "---\ntype: [unclosed\n---\n# Bad\n\nbody\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    row = db.execute("select title, malformed from notes where path='bad.md'").fetchone()
    assert tuple(row) == ("Bad", 1)
    db.close()


async def test_a_note_replaced_by_a_symlink_is_purged_never_read_outside(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    outside = make_vault({"secret.md": "# Secret\n\nnot in this vault at all\n"})
    db = open_index(root)
    await reindex(db, root, embedder)

    # the scan skips symlinks, so without a check the row would keep its content
    # from *outside* the vault forever: doctor reports it missing and repair,
    # reading straight through the link, never removes it
    (root / "notes" / "lonely.md").unlink()
    os.symlink(outside / "secret.md", root / "notes" / "lonely.md")

    stats = await reindex(db, root, embedder)
    assert (stats.removed, stats.added, stats.updated) == (1, 0, 0)
    assert one(db, "select count(*) from notes where path='notes/lonely.md'") == 0
    assert one(db, "select rowid from notes_fts where notes_fts match 'secret'") is None
    db.close()


async def test_a_note_replaced_by_a_directory_is_purged(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    await reindex(db, root, embedder)

    (root / "notes" / "lonely.md").unlink()
    (root / "notes" / "lonely.md").mkdir()  # `mv note.md dir/` leaves this behind
    assert (await reindex(db, root, embedder)).removed == 1
    assert one(db, "select count(*) from notes where path='notes/lonely.md'") == 0
    db.close()

    # and doctor agrees rather than failing on EISDIR with no way to repair
    assert (await doctor(root, embedder)).missing == []


def test_read_note_returns_none_when_the_file_vanished(make_vault: MakeVault) -> None:
    assert read_note(make_vault(FIXTURE), "notes/ghost.md") is None


async def test_a_file_deleted_mid_run_does_not_abort_the_reindex(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)

    async def racing(texts: list[str]) -> list[Vector]:
        (root / "notes" / "globex.md").unlink()  # gone after we read it, before we write
        return [[1.0] * 32 for _ in texts]

    racy = stub_embedder("race-v1", 32, racing)
    await reindex(db, root, racy)  # must not raise
    stats = await reindex(db, root, racy)
    assert (stats.removed, stats.unchanged) == (1, 2)
    db.close()


async def test_an_embedder_returning_the_wrong_shape_fails_clearly(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)

    async def eight_wide(texts: list[str]) -> list[Vector]:
        return [[0.0] * 8 for _ in texts]

    async def nothing(texts: list[str]) -> list[Vector]:
        return []

    with pytest.raises(VaultError, match=re.compile(r"narrow-v1.*width 8.*32", re.S)):
        await reindex(db, root, stub_embedder("narrow-v1", 32, eight_wide))
    with pytest.raises(VaultError, match=re.compile(r"short-v1.*0 vectors for 3", re.S)):
        await reindex(db, root, stub_embedder("short-v1", 32, nothing))
    db.close()


async def test_an_empty_vault_indexes_to_an_empty_database(make_vault: MakeVault) -> None:
    root = make_vault({})
    (root / "empty-dir").mkdir()
    db = open_index(root)
    stats = await reindex(db, root, embedder)
    assert (stats.added, stats.removed) == (0, 0)
    assert one(db, "select count(*) from notes") == 0
    db.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
async def test_an_unreadable_directory_raises_rather_than_purging_its_notes(
    make_vault: MakeVault,
) -> None:
    """A directory we cannot read is not a directory whose notes are gone: EACCES
    read as absence would silently drop every note under it from the index."""
    root = make_vault({"locked/secret.md": "# Secret\n\nbody\n", "open.md": "# Open\n"})
    db = open_index(root)
    try:
        assert (await reindex(db, root, embedder)).added == 2
        (root / "locked").chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                await reindex(db, root, embedder)
            rows = [r["path"] for r in db.execute("select path from notes order by path")]
            assert "locked/secret.md" in rows  # not purged behind our back
        finally:
            (root / "locked").chmod(0o755)
    finally:
        db.close()

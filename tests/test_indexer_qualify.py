"""Auto-qualifying bare wikilinks when a pass creates a stem collision."""

from conftest import MakeVault, write_note
from indexer_common import COLLISION, embedder, open_index, read_file

from wilcus_vault.doctor import doctor
from wilcus_vault.indexer import index_paths, reindex, scan_vault
from wilcus_vault.qualify import Qualified


async def test_a_new_stem_collision_auto_qualifies_every_bare_link(make_vault: MakeVault) -> None:
    root = make_vault(COLLISION)
    db = open_index(root)
    # cold first index: every note is new, structurally no incumbent, no rewrite
    cold = await reindex(db, root, embedder)
    assert cold.qualified == []
    assert read_file(root, "hub.md") == COLLISION["hub.md"]

    # the collision: a second acme appears
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    assert stats.qualified == [Qualified("acme", "customers/acme", ["hub.md", "notes/deal.md"], [])]
    # the re-entry's pass is part of this one: the notes it rewrote count as updated
    assert stats.updated == 2
    # textual body edits: link qualified, alias kept, everything else verbatim
    assert read_file(root, "hub.md") == "# Hub\n\nsee [[customers/acme]] for the account\n"
    assert read_file(root, "notes/deal.md") == (
        "# Deal\n\nclosing [[customers/acme|Acme Corp]] this week\n"
    )
    # the rewritten notes were re-entered through index_paths: edges point at the
    # incumbent by path and nothing is ambiguous
    rows = db.execute(
        """select n.path from edges e join notes n on n.id = e.to_id
           where e.to_slug = 'customers/acme' order by n.path"""
    ).fetchall()
    assert [r["path"] for r in rows] == ["customers/acme.md", "customers/acme.md"]
    assert (await doctor(root, embedder)).ambiguous_links == []

    # idempotent: the next pass has nothing to qualify and dirties nothing
    before = db.total_changes
    again = await index_paths(db, root, embedder, scan_vault(root)[0])
    assert again.qualified == []
    assert db.total_changes == before
    db.close()


async def test_both_colliding_notes_new_in_the_same_pass_no_rewrite(make_vault: MakeVault) -> None:
    root = make_vault({"hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "customers/acme.md", "# Acme one\n")
    write_note(root, "vendors/acme.md", "# Acme two\n")
    stats = await index_paths(db, root, embedder, ["customers/acme.md", "vendors/acme.md"])
    # picking a winner would be the "first match" resolution the design forbids
    assert stats.qualified == []
    assert read_file(root, "hub.md") == "# Hub\n\n[[acme]]\n"
    assert len((await doctor(root, embedder)).ambiguous_links) == 1
    db.close()


async def test_a_rename_is_not_a_collision(make_vault: MakeVault) -> None:
    root = make_vault({"a/acme.md": "# Acme\n", "hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    (root / "a" / "acme.md").unlink()
    write_note(root, "b/acme.md", "# Acme\n")
    stats = await index_paths(db, root, embedder, ["a/acme.md", "b/acme.md"])
    assert stats.qualified == []
    assert read_file(root, "hub.md") == "# Hub\n\n[[acme]]\n"
    db.close()


async def test_a_root_incumbent_has_no_qualified_form(make_vault: MakeVault) -> None:
    root = make_vault({"acme.md": "# Acme\n", "hub.md": "# Hub\n\n[[acme]]\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "vendors/acme.md", "# Acme two\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    assert stats.qualified == [Qualified("acme", None, [], ["hub.md"])]
    assert read_file(root, "hub.md") == "# Hub\n\n[[acme]]\n"
    db.close()


async def test_a_linking_note_edited_mid_flight_is_skipped_never_clobbered(
    make_vault: MakeVault,
) -> None:
    root = make_vault(COLLISION)
    db = open_index(root)
    await reindex(db, root, embedder)
    # hub.md changes on disk but this pass does not carry it, so its indexed
    # hash is stale: the rewrite's check-and-write must refuse it
    write_note(root, "hub.md", "# Hub\n\nsee [[acme]], hand-edited\n")
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    assert stats.qualified == [Qualified("acme", "customers/acme", ["notes/deal.md"], ["hub.md"])]
    assert read_file(root, "hub.md") == "# Hub\n\nsee [[acme]], hand-edited\n"
    db.close()


async def test_the_qualify_cap_bounds_rewrites_and_reports_the_remainder(
    make_vault: MakeVault,
) -> None:
    root = make_vault(COLLISION)
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "vendors/acme.md", "# Acme the vendor\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md"], 1)
    assert stats.qualified == [Qualified("acme", "customers/acme", ["hub.md"], ["notes/deal.md"])]
    assert read_file(root, "notes/deal.md") == COLLISION["notes/deal.md"]
    db.close()


async def test_a_collision_with_no_bare_linkers_earns_no_report_entry(
    make_vault: MakeVault,
) -> None:
    root = make_vault({"customers/acme.md": "# Acme\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    write_note(root, "vendors/acme.md", "# Acme two\n")
    stats = await index_paths(db, root, embedder, ["vendors/acme.md"])
    assert stats.qualified == []
    db.close()

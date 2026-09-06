"""Doctor: the link graph report, drift repair, the discard log, and rebuild."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import MakeVault, write_note

from wilcus_vault.db import db_path, open_db
from wilcus_vault.discards import list_discards
from wilcus_vault.doctor import AmbiguousLink, DoctorOptions, LinkProblem, doctor
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.indexer import index_paths, reindex

embedder = TokenOverlapEmbedder(32)

GRAPH = {
    "notes/acme.md": "# Acme\n\n[[globex]], [[ghost]], [[dup]]\n",
    "notes/globex.md": "# Globex\n\nno outgoing links\n",
    "notes/lonely.md": "# Lonely\n\nnothing here\n",
    "one/dup.md": "# Dup one\n",
    "two/dup.md": "# Dup two\n",
}


def snapshot(root: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """notes + edges as comparable, id-free tuples"""
    db = open_db(db_path(root))
    notes = [
        tuple(r)
        for r in db.execute("select path, slug, title, hash, malformed from notes order by path")
    ]
    edges = [
        tuple(r)
        for r in db.execute(
            """select f.path as from_path, e.to_slug, t.path as to_path
               from edges e join notes f on f.id = e.from_id
               left join notes t on t.id = e.to_id
               order by f.path, e.to_slug"""
        )
    ]
    db.close()
    return notes, edges


async def test_separates_broken_links_from_ambiguous_and_names_candidates(
    make_vault: MakeVault,
) -> None:
    root = make_vault(GRAPH)
    report = await doctor(root, embedder)
    # 0 candidates is a typo or a deleted note; 2+ is a link that needs
    # qualifying, and the report says what to qualify it with
    assert report.broken_links == [LinkProblem("notes/acme.md", "ghost")]
    # candidates are written as links, not as filenames: the report's fix is
    # paste-able straight into the note ([[one/dup]])
    assert report.ambiguous_links == [AmbiguousLink("notes/acme.md", "dup", ["one/dup", "two/dup"])]
    assert report.orphans == ["notes/lonely.md", "one/dup.md", "two/dup.md"]
    assert report.malformed == []


async def test_counts_the_discard_log_total_and_recent(make_vault: MakeVault) -> None:
    root = make_vault({"a.md": "# A\n\n[[b]]\n", "b.md": "# B\n\n[[a]]\n"})
    # no log at all is zero, not an error
    assert (await doctor(root, embedder)).discards == {"entries": 0, "recent": 0}

    def line(at: str) -> str:
        return json.dumps({"at": at, "candidate": {"title": "t", "body": "b"}}) + "\n"

    (root / ".discarded.log").write_text(
        line("2020-01-01T00:00:00.000Z") + line(datetime.now(UTC).isoformat())
    )
    assert (await doctor(root, embedder)).discards == {"entries": 2, "recent": 1}


async def test_qualified_link_is_broken_never_ambiguous(make_vault: MakeVault) -> None:
    root = make_vault(
        {
            # namespaced notes may share a stem: that is what namespaces are for. Only
            # a *bare* link to them is a problem, and qualifying it is the fix.
            "notes/acme.md": "# Acme\n\n[[one/dup]] and [[two/dup]]\n",
            "notes/gone.md": "# Gone\n\n[[one/ghost]]\n",
            "one/dup.md": "# Dup one\n",
            "two/dup.md": "# Dup two\n",
        }
    )
    report = await doctor(root, embedder)
    assert report.ambiguous_links == []
    assert report.broken_links == [LinkProblem("notes/gone.md", "one/ghost")]


async def test_reports_stale_rows_and_missing_files_then_repairs(make_vault: MakeVault) -> None:
    root = make_vault(GRAPH)
    db = open_db(db_path(root))
    await reindex(db, root, embedder)
    db.close()

    write_note(root, "notes/globex.md", "# Globex\n\nedited by a human\n")
    write_note(root, "notes/new.md", "# New\n\n[[globex]]\n")
    os.remove(root / "notes" / "lonely.md")

    found = await doctor(root, embedder, DoctorOptions(repair=False))
    assert sorted(found.stale) == ["notes/globex.md", "notes/new.md"]
    assert found.missing == ["notes/lonely.md"]

    repaired = await doctor(root, embedder)
    assert sorted(repaired.stale) == ["notes/globex.md", "notes/new.md"]  # what it fixed
    clean = await doctor(root, embedder, DoctorOptions(repair=False))
    assert (clean.stale, clean.missing) == ([], [])

    after = open_db(db_path(root))
    gone = after.execute("select count(*) from notes where path='notes/lonely.md'").fetchone()
    assert gone[0] == 0
    assert after.execute("select count(*) from notes").fetchone()[0] == 5
    after.close()


async def test_reports_malformed_frontmatter(make_vault: MakeVault) -> None:
    root = make_vault({"bad.md": "---\ntype: [unclosed\n---\n# Bad\n"})
    assert (await doctor(root, embedder)).malformed == ["bad.md"]


async def test_rebuild_reproduces_an_identical_index_after_db_deleted(
    make_vault: MakeVault,
) -> None:
    root = make_vault(GRAPH)
    await doctor(root, embedder)
    before = snapshot(root)

    os.remove(db_path(root))
    report = await doctor(root, embedder, DoctorOptions(rebuild=True))
    assert report.missing == []
    assert snapshot(root) == before

    # rebuilding over a live index leaves no temp files behind
    await doctor(root, embedder, DoctorOptions(rebuild=True))
    assert snapshot(root) == before
    assert os.listdir(root / ".vault") == ["index.db"]


async def test_moves_a_discard_log_left_in_vault_dir_out_once(make_vault: MakeVault) -> None:
    root = make_vault(GRAPH)
    log = root / ".discarded.log"
    old = root / ".vault" / "discarded.log"

    def entry(at: str, title: str) -> str:
        line = {"at": at, "candidate": {"title": title, "body": "b"}}
        return json.dumps(line, separators=(",", ":")) + "\n"

    write_note(root, ".vault/discarded.log", entry("2026-01-01T00:00:00.000Z", "old"))
    write_note(root, ".discarded.log", entry("2026-02-01T00:00:00.000Z", "new"))

    # a report is a report: repair=False must not move anything
    assert (await doctor(root, embedder, DoctorOptions(repair=False))).migrated_discard_log is False
    assert old.exists()

    report = await doctor(root, embedder, DoctorOptions(repair=True))
    assert report.migrated_discard_log is True
    # appended, not replaced: both files are history and neither may be lost
    assert '"title":"old"' in log.read_text()
    assert '"title":"new"' in log.read_text()
    assert not old.exists()

    # once: a second run must not re-append what it already moved
    assert (await doctor(root, embedder, DoctorOptions(repair=True))).migrated_discard_log is False
    assert len(log.read_text().strip().split("\n")) == 2


async def test_rebuild_reembeds_by_definition_and_reports_it(make_vault: MakeVault) -> None:
    root = make_vault(GRAPH)
    assert (await doctor(root, embedder, DoctorOptions(rebuild=True))).reembedded is True


async def test_skips_a_log_entry_whose_candidate_is_not_one(make_vault: MakeVault) -> None:
    """A title-only candidate cannot be restored, so the line is malformed: skipped
    and counted, never an exception out of doctor."""
    root = make_vault(GRAPH)
    (root / ".discarded.log").write_text(
        '{"at":"2026-01-01T00:00:00.000Z","candidate":{"title":"old"}}\n'
        '{"at":"2026-01-01T00:00:00.000Z","candidate":{"title":"t","body":"b","extra":1}}\n'
    )
    assert (await doctor(root, embedder)).discards["entries"] == 1  # the extra key is dropped
    entries, malformed = list_discards(root)
    assert [e.candidate.title for e in entries] == ["t"]
    assert malformed == 1


async def test_migration_never_writes_through_a_symlink_and_gitignores_the_log(
    make_vault: MakeVault,
) -> None:
    """The migration writes whole candidate bodies, so it obeys the same two
    rails as every other write to the log: never through a link, always ignored."""
    root = make_vault(GRAPH)
    outside = make_vault({})
    line = json.dumps({"at": "2026-01-01T00:00:00.000Z", "candidate": {"title": "t", "body": "b"}})
    write_note(root, ".vault/discarded.log", line + "\n")
    os.symlink(outside / "stolen.log", root / ".discarded.log")

    with pytest.raises(OSError):
        await doctor(root, embedder, DoctorOptions(repair=True))
    assert not (outside / "stolen.log").exists()

    (root / ".discarded.log").unlink()
    assert (await doctor(root, embedder, DoctorOptions(repair=True))).migrated_discard_log is True
    assert ".discarded.log*" in (root / ".gitignore").read_text()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
async def test_doctor_reports_directories_it_could_not_read(make_vault: MakeVault) -> None:
    """Doctor is the operator's view of drift, and "we could not look here" is
    drift it cannot repair: the notes under a locked directory are not missing."""
    root = make_vault({"locked/secret.md": "# Secret\n", "open.md": "# Open\n"})
    (root / "locked").chmod(0o000)
    try:
        report = await doctor(root, embedder, DoctorOptions(repair=False))
        assert report.unreadable == ["locked"]
    finally:
        (root / "locked").chmod(0o755)


async def test_rebuild_keeps_the_live_index_file_so_open_handles_are_not_stranded(
    make_vault: MakeVault,
) -> None:
    """`--rebuild` used to rename a fresh index over `index.db`. A process that
    already had the file open kept writing to the replaced inode — into a file
    nothing would ever open again — so its rows were lost. The rebuild happens in
    the live file instead, and there is no second inode to be stranded on."""
    root = make_vault(GRAPH)
    await doctor(root, embedder)
    held = open_db(db_path(root))
    inode = db_path(root).stat().st_ino
    try:
        await doctor(root, embedder, DoctorOptions(rebuild=True))
        assert db_path(root).stat().st_ino == inode
        write_note(root, "notes/after.md", "# After\n")
        await index_paths(held, root, embedder, ["notes/after.md"])
    finally:
        held.close()
    fresh = open_db(db_path(root))
    rows = fresh.execute("select count(*) from notes where path = 'notes/after.md'").fetchone()
    fresh.close()
    assert rows[0] == 1  # the held handle wrote into the index everyone else reads

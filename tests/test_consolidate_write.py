"""What a write run does to the files: the gate's own create and supersede
rails, run over N members."""

from pathlib import Path

import pytest
from conftest import MakeVault, write_note
from consolidate_fixture import CTX, NOTES, PAIRS, fake_merger, fm, open_vault, read

import wilcus_vault.gate_write as gate_write
from wilcus_vault import ConsolidateRun
from wilcus_vault.paths import write_new


async def test_a_write_run_creates_the_merged_note_and_marks_every_member_superseded(
    make_vault: MakeVault,
) -> None:
    merger, seen = fake_merger()
    v = open_vault(make_vault, NOTES, merger)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True, ctx=CTX))

    assert r.dry_run is False
    merge = r.merges[0]
    # Written through the gate's create rail: slug from the title, in the
    # cluster's own namespace.
    assert merge.path == "notes/merged-note.md"
    assert merge.superseded == ["notes/alpha.md", "notes/beta.md"]
    assert merge.unmarked == []
    front = fm(v.root, merge.path)
    assert front["title"] == "Merged note"
    assert front["type"] == "customer"
    assert front["vault_agent"] == "core/librarian"
    assert front["vault_source"] == "task-9"

    for rel in merge.superseded:
        old = read(v.root, rel)
        assert 'superseded_by: "notes/merged-note.md"' in old
        assert "[[notes/merged-note]]" in old  # path-qualified, like the gate's
        assert NOTES[rel].strip() in old  # nothing deleted
    # The merger saw the bodies as they are on disk, not as the index has them.
    assert [n.path for n in seen[0].notes] == ["notes/alpha.md", "notes/beta.md"]
    assert "ALPHA marks the first note" in seen[0].notes[0].body

    # Reindexed by the pass: the merged note is searchable and the members are
    # out of search, and no wikilink was left dangling.
    hits = [h.path for h in await v.search("ALPHA")]
    assert "notes/merged-note.md" in hits
    assert "notes/alpha.md" not in hits
    assert "notes/beta.md" not in hits
    assert (await v.doctor()).broken_links == []
    v.close()


async def test_a_member_is_marked_never_restamped_or_reserialized(make_vault: MakeVault) -> None:
    legacy = """---
type: customer
id: 01234 # legacy account number, must survive a consolidation
rate: 1.0
---
# Alpha

ALPHA marks the first note.
"""
    v = open_vault(make_vault, {**NOTES, "notes/alpha.md": legacy}, fake_merger()[0])
    await v.consolidate(ConsolidateRun(ceiling=0.25, write=True, ctx=CTX))

    old = read(v.root, "notes/alpha.md")
    # A YAML round-trip would drop the comment and turn 01234 into 1234.
    assert "id: 01234 # legacy account number, must survive a consolidation" in old
    assert "rate: 1.0" in old
    # Marking is bookkeeping, not authorship: the merging agent is recorded on
    # the note it wrote, not on the ones it retired.
    front = fm(v.root, "notes/alpha.md")
    assert front["superseded_by"] == "notes/merged-note.md"
    assert "vault_agent" not in front
    assert "vault_source" not in front
    v.close()


async def test_the_cap_counts_clusters_merged_and_reports_the_remainder_untouched(
    make_vault: MakeVault,
) -> None:
    v = open_vault(make_vault, PAIRS, fake_merger()[0])
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, cap=1, write=True))

    assert len(r.merges) == 1
    assert r.merges[0].cluster.members == ["notes/a1.md", "notes/a2.md"]
    assert [c.members for c in r.remaining] == [
        ["notes/d1.md", "notes/d2.md"],
        ["notes/e1.md", "notes/e2.md"],
    ]
    # A run that wants to rewrite half the vault is evidence the ceiling is
    # wrong: the remainder is a report, not a half-finished rewrite.
    for cluster in r.remaining:
        for rel in cluster.members:
            assert "superseded_by" not in read(v.root, rel)
    v.close()


async def test_a_member_edited_mid_flight_is_reported_unmarked_and_the_merged_note_stands(
    make_vault: MakeVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = open_vault(make_vault, NOTES, fake_merger()[0])

    def racing_write(abs_path: Path, text: str) -> bool:
        done = write_new(abs_path, text)
        if done and "merged-note" in abs_path.name:
            # The merged note has landed; now a human saves a member.
            monkeypatch.setattr(gate_write, "write_new", write_new)
            write_note(v.root, "notes/beta.md", "# Beta\n\nhand edit\n")
        return done

    monkeypatch.setattr(gate_write, "write_new", racing_write)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))

    merge = r.merges[0]
    assert merge.path == "notes/merged-note.md"
    assert merge.superseded == ["notes/alpha.md"]
    assert merge.unmarked == ["notes/beta.md"]
    assert read(v.root, "notes/beta.md") == "# Beta\n\nhand edit\n"  # not clobbered
    assert "ALPHA and BETA" in read(v.root, "notes/merged-note.md")
    v.close()

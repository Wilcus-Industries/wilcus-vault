"""What a cluster is (complete linkage under a mandatory ceiling), and what a
dry run reports without touching a file."""

import math
from typing import Any

import pytest
from conftest import MakeVault
from consolidate_fixture import NOTES, embedder, fake_merger, open_vault, read

from wilcus_vault import ConsolidateRun, MergedNote, VaultContext, VaultError, open


async def test_complete_linkage_refuses_a_chain_into_a_note_the_cluster_does_not_resemble(
    make_vault: MakeVault,
) -> None:
    v = open_vault(make_vault, NOTES, fake_merger()[0])
    r = await v.consolidate(ConsolidateRun(ceiling=0.25))

    # beta is under the ceiling from both alpha (0.10) and gamma (0.20), but
    # alpha and gamma are 0.54 apart: single linkage would merge all three.
    assert len(r.merges) == 1
    cluster = r.merges[0].cluster
    assert cluster.members == ["notes/alpha.md", "notes/beta.md"]
    assert cluster.namespace == "notes/"
    assert cluster.distance == pytest.approx(0.1, abs=1e-3)
    assert r.cross_namespace == []
    assert r.remaining == []
    assert "gamma" not in repr(r)

    # A wide enough ceiling does take all three: the rule is the ceiling, not a
    # refusal to cluster more than two notes.
    wide = await v.consolidate(ConsolidateRun(ceiling=0.6))
    assert wide.merges[0].cluster.members == ["notes/alpha.md", "notes/beta.md", "notes/gamma.md"]
    v.close()


async def test_the_ceiling_is_mandatory_and_bounded_and_the_cap_is_a_positive_integer(
    make_vault: MakeVault,
) -> None:
    v = open_vault(make_vault, NOTES, fake_merger()[0])
    bad_ceilings: list[Any] = [None, "0.2", math.nan, math.inf, -0.1, 2.5]
    for ceiling in bad_ceilings:
        with pytest.raises(VaultError, match="ceiling"):
            await v.consolidate(ConsolidateRun(ceiling=ceiling))
    bad_caps: list[Any] = [0, -1, 1.5, math.nan]
    for cap in bad_caps:
        with pytest.raises(VaultError, match="cap"):
            await v.consolidate(ConsolidateRun(ceiling=0.25, cap=cap))
    with pytest.raises(VaultError, match="agent"):
        await v.consolidate(ConsolidateRun(ceiling=0.25, ctx=VaultContext(agent="  ")))
    v.close()

    # And the merger is injected like the decider: without one there is no pass.
    bare = open(make_vault(NOTES), embedder)
    with pytest.raises(VaultError, match="vault: consolidate needs a merger"):
        await bare.consolidate(ConsolidateRun(ceiling=0.25))
    bare.close()


async def test_a_cluster_spanning_namespaces_is_reported_and_never_merged(
    make_vault: MakeVault,
) -> None:
    files = {"one/alpha.md": NOTES["notes/alpha.md"], "two/beta.md": NOTES["notes/beta.md"]}
    merger, seen = fake_merger()
    v = open_vault(make_vault, files, merger)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))

    assert r.merges == []
    assert len(r.cross_namespace) == 1
    assert r.cross_namespace[0].members == ["one/alpha.md", "two/beta.md"]
    assert r.cross_namespace[0].namespace is None
    # Collapsing a namespace boundary is a human call, so not even a model call
    # is spent on it, and this was a *write* run.
    assert seen == []
    for rel, text in files.items():
        assert read(v.root, rel) == text

    # A deeper namespace is still another namespace: notes/ and notes/old/ are
    # reported, not merged into whichever of the two is shallower.
    deep = open_vault(
        make_vault,
        {"notes/alpha.md": NOTES["notes/alpha.md"], "notes/old/beta.md": NOTES["notes/beta.md"]},
        merger,
    )
    assert len((await deep.consolidate(ConsolidateRun(ceiling=0.25))).cross_namespace) == 1
    deep.close()
    v.close()


async def test_a_dry_run_is_the_default_it_reports_the_merge_and_writes_nothing(
    make_vault: MakeVault,
) -> None:
    v = open_vault(make_vault, NOTES, fake_merger()[0])
    r = await v.consolidate(ConsolidateRun(ceiling=0.25))

    assert r.dry_run is True
    merge = r.merges[0]
    assert merge.candidate == MergedNote("Merged note", "ALPHA and BETA in one note.\n", "customer")
    assert merge.path is None
    assert merge.superseded is None
    assert merge.unmarked is None
    assert not (v.root / "notes/merged-note.md").exists()
    for rel, text in NOTES.items():
        assert read(v.root, rel) == text
    v.close()


async def test_a_superseded_note_is_never_a_cluster_member(make_vault: MakeVault) -> None:
    retired = '---\nsuperseded_by: "notes/beta.md"\n---\n# Alpha\n\nALPHA marks the first note.\n'
    v = open_vault(make_vault, {**NOTES, "notes/alpha.md": retired}, fake_merger()[0])
    r = await v.consolidate(ConsolidateRun(ceiling=0.25))

    # alpha is out, so what is left is the 0.20 pair it used to hide
    assert len(r.merges) == 1
    assert r.merges[0].cluster.members == ["notes/beta.md", "notes/gamma.md"]
    assert "alpha" not in repr(r)
    v.close()

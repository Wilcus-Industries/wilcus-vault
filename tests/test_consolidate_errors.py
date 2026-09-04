"""A write run collects per-cluster errors and never loses the report of what landed."""

import os
from pathlib import Path

import pytest
from conftest import MakeVault
from consolidate_fixture import PAIRS, embedder, fake_merger, marker_embed, open_vault, read
from fakes import stub_embedder

import wilcus_vault.consolidate as consolidate_module
import wilcus_vault.gate_write as gate_write
from wilcus_vault import (
    ConsolidateOptions,
    ConsolidateRun,
    MergedNote,
    MergeInput,
    VaultError,
    open,
)
from wilcus_vault.db import db_path, open_db
from wilcus_vault.embed import Vector
from wilcus_vault.indexer import index_paths, read_raw
from wilcus_vault.paths import write_atomic


async def delta_falls_over(input: MergeInput) -> MergedNote:
    """A merger that throws on the DELTA cluster and merges the rest."""
    notes = input.notes
    if any("DELTA" in n.body for n in notes):
        raise Exception("model fell over")
    return MergedNote(notes[0].title + " merged", "".join(n.body for n in notes))


async def test_a_write_run_collects_per_cluster_errors_and_the_index_never_lags(
    make_vault: MakeVault,
) -> None:
    # The ALPHA merge has already landed when DELTA throws, and EPSILON still runs.
    v = open_vault(make_vault, PAIRS, delta_falls_over)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))

    # The landed merge is reported, not discarded behind the exception.
    assert len(r.merges) == 2
    assert [m.cluster.members[0] for m in r.merges] == ["notes/a1.md", "notes/e1.md"]
    assert len(r.errors) == 1
    assert r.errors[0].cluster.members == ["notes/d1.md", "notes/d2.md"]
    assert "model fell over" in r.errors[0].error
    # The erroring cluster's members are untouched.
    for rel in ("notes/d1.md", "notes/d2.md"):
        assert read(v.root, rel) == PAIRS[rel]
    # And the closing reindex ran: the index does not lag the landed writes.
    assert "notes/a1.md" not in [h.path for h in await v.search("ALPHA")]
    assert "notes/e1.md" not in [h.path for h in await v.search("EPSILON")]
    v.close()

    # An errored cluster counts against the cap: it spent its model call.
    capped = open_vault(make_vault, PAIRS, delta_falls_over)
    cr = await capped.consolidate(ConsolidateRun(ceiling=0.25, cap=2, write=True))
    assert len(cr.merges) == 1
    assert len(cr.errors) == 1
    assert [c.members for c in cr.remaining] == [["notes/e1.md", "notes/e2.md"]]
    capped.close()

    # Dry-run behavior is unchanged: a merger throw still propagates.
    dry = open_vault(make_vault, PAIRS, delta_falls_over)
    with pytest.raises(Exception, match="model fell over"):
        await dry.consolidate(ConsolidateRun(ceiling=0.25))
    dry.close()


async def test_a_failing_closing_reindex_is_reported_never_a_throw(make_vault: MakeVault) -> None:
    # The embedder works for the opening reindex and dies on the closing one: a
    # network hiccup after the merge landed must not cost the operator the report.
    calls = 0

    async def flaky_embed(texts: list[str]) -> list[Vector]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise Exception("embedding endpoint fell over")
        return await marker_embed(texts)

    flaky = stub_embedder("marker-v1", 3, flaky_embed)
    files = {k: PAIRS[k] for k in ("notes/a1.md", "notes/a2.md")}
    v = open(make_vault(files), flaky, consolidate=ConsolidateOptions(fake_merger()[0]))
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))

    assert len(r.merges) == 1
    assert r.merges[0].superseded == ["notes/a1.md", "notes/a2.md"]
    assert r.index_error is not None
    assert "embedding endpoint fell over" in r.index_error
    v.close()


async def test_a_member_path_escaping_the_vault_aborts_the_run_loudly(
    make_vault: MakeVault,
) -> None:
    # A healthy in-vault cluster (identical DELTA pair, distance 0) sorts ahead
    # of the escaping one (ALPHA–BETA, 0.10): the abort must land before *any*
    # merge does, or the report and the closing reindex are discarded with it.
    root = make_vault({k: PAIRS[k] for k in ("notes/d1.md", "notes/d2.md")})
    evil = make_vault(
        {"z1.md": "# Z one\n\nALPHA marks it.\n", "z2.md": "# Z two\n\nBETA marks it.\n"}
    )
    v = open(root, embedder, consolidate=ConsolidateOptions(fake_merger()[0]))
    # Plant index rows whose paths point outside the vault: what a tampered or
    # corrupt index looks like to the pass.
    db = open_db(db_path(root))
    rels = [os.path.relpath(evil / f, root).replace("\\", "/") for f in ("z1.md", "z2.md")]
    await index_paths(db, root, embedder, rels)
    db.close()

    with pytest.raises(VaultError, match="resolves outside the vault"):
        await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))
    # Nothing was merged behind the abort, not even the healthy cluster.
    assert not (evil / "merged-note.md").exists()
    assert not (root / "notes/merged-note.md").exists()
    assert "superseded_by" not in read(root, "notes/d1.md")
    v.close()


async def test_an_error_before_the_model_call_does_not_burn_a_cap_slot(
    make_vault: MakeVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {k: PAIRS[k] for k in ("notes/a1.md", "notes/a2.md", "notes/d1.md", "notes/d2.md")}
    v = open_vault(make_vault, files, fake_merger()[0])

    # The ALPHA cluster sorts first and its member turns unreadable between the
    # opening reindex and the cluster loop's own read: no model call.
    def unreadable_a1(root: str | Path, rel: str) -> str | None:
        if rel.endswith("notes/a1.md"):
            raise PermissionError("EACCES: permission denied")
        return read_raw(root, rel)

    monkeypatch.setattr(consolidate_module, "read_raw", unreadable_a1)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, cap=1, write=True))

    # The unreadable cluster is an error, and the DELTA cluster still got the
    # run's one model call instead of being pushed to remaining.
    assert len(r.errors) == 1
    assert r.errors[0].cluster.members == ["notes/a1.md", "notes/a2.md"]
    assert len(r.merges) == 1
    assert r.merges[0].cluster.members == ["notes/d1.md", "notes/d2.md"]
    assert r.remaining == []
    v.close()


async def test_a_reported_error_is_scrubbed_of_control_characters(make_vault: MakeVault) -> None:
    async def merger(_input: MergeInput) -> MergedNote:
        raise Exception("model \x1b[2J fell over")

    v = open_vault(make_vault, {k: PAIRS[k] for k in ("notes/a1.md", "notes/a2.md")}, merger)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))
    # Merger output travels out as report data a consumer will print.
    assert r.errors[0].error == "model ?[2J fell over"
    v.close()


async def test_a_throw_after_create_still_reports_the_path_and_the_reindex_covers_it(
    make_vault: MakeVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = open_vault(
        make_vault, {k: PAIRS[k] for k in ("notes/a1.md", "notes/a2.md")}, fake_merger()[0]
    )

    def disk_full_on_a1(abs_path: Path, text: str) -> None:
        # The merged note lands, then marking the first member hits a full disk.
        if "a1.md" in abs_path.name:
            raise OSError("disk full")
        write_atomic(abs_path, text)

    monkeypatch.setattr(gate_write, "write_atomic", disk_full_on_a1)
    r = await v.consolidate(ConsolidateRun(ceiling=0.25, write=True))

    assert r.merges == []
    assert len(r.errors) == 1
    assert "disk full" in r.errors[0].error
    # The operator learns which file the pass created before it threw,
    assert r.errors[0].path == "notes/merged-note.md"
    # and the closing reindex covered it: the orphan is live in search, not invisible.
    assert "notes/merged-note.md" in [h.path for h in await v.search("ALPHA")]
    v.close()

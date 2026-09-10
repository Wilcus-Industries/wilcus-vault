"""Write-gate evals: create, and the note-authoring corners of the gate."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import MakeVault
from fakes import fixed_decider
from gate_common import CANDIDATE, EMBEDDER, NOTHING_SIMILAR, OLD_NOTE, VAULT, open_gate, read

from wilcus_vault.db import db_path, open_db
from wilcus_vault.decision import DeciderInput, Decision
from wilcus_vault.note import parse_note
from wilcus_vault.paths import slugify
from wilcus_vault.search_sql import Cutoffs
from wilcus_vault.term import VaultError
from wilcus_vault.vault import open

CREATE = fixed_decider(Decision("create"))


async def test_create_writes_a_new_note_we_authored_confined_and_indexed(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    r = await v.propose(CANDIDATE)

    assert (r.action, r.path, r.fell_back) == ("create", "notes/acme-renewal-2026.md", False)
    assert r.path is not None
    raw = read(v.root, r.path)
    assert raw.startswith("---\n")
    assert "title: Acme renewal 2026" in raw
    assert "type: customer" in raw
    assert re.search(r"created: '?20\d\d-", raw)
    assert raw.endswith(CANDIDATE.body)

    # reindexed synchronously by propose: the new note is searchable right away
    hits = [h.path for h in await v.search("acme renewal 2026 pricing")]
    assert "notes/acme-renewal-2026.md" in hits
    v.close()


async def test_create_never_overwrites_an_existing_note_or_reuses_a_stem(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    r = await v.propose(replace(CANDIDATE, title="Acme renewal"))
    assert r.path == "notes/acme-renewal-2.md"
    assert read(v.root, "notes/acme-renewal.md") == OLD_NOTE  # untouched
    v.close()


async def test_no_similar_notes_is_a_valid_outcome_cutoffs_are_passed(
    make_vault: MakeVault,
) -> None:
    seen: list[DeciderInput] = []

    async def decider(input: DeciderInput) -> Decision:
        seen.append(input)
        return Decision("create")

    v = await open_gate(make_vault(VAULT), decider, NOTHING_SIMILAR)
    r = await v.propose(
        replace(
            CANDIDATE,
            title="Zeppelin fuselage torque",
            type=None,
            body="Torque values for the zeppelin fuselage struts.\n",
        )
    )
    assert seen[0].similar == []  # the vault has notes; none of them are similar
    assert (r.action, r.path) == ("create", "notes/zeppelin-fuselage-torque.md")
    v.close()


async def test_a_title_with_no_slug_characters_still_gets_a_note_keyed_by_content(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    r = await v.propose(replace(CANDIDATE, title="日本語のメモ", body="本文です。\n"))
    # slugified to nothing, so the stem comes off the candidate's own hash:
    # losing the note because its title is not Latin is not an option
    assert r.path is not None
    assert re.fullmatch(r"notes/note-[0-9a-f]{8}\.md", r.path)
    assert "日本語のメモ" in read(v.root, r.path)
    assert slugify("日本語") is None
    v.close()


async def test_a_vault_out_of_free_filenames_logs_the_candidate_before_giving_up(
    make_vault: MakeVault,
) -> None:
    files = {
        f"notes/acme-renewal-2026{'' if i == 1 else f'-{i}'}.md": f"# taken {i}\n\nfull.\n"
        for i in range(1, 51)
    }
    v = await open_gate(make_vault(files), CREATE)
    with pytest.raises(VaultError, match="free filename"):
        await v.propose(CANDIDATE)

    # thrown, but not lost: the candidate is recoverable from the discard log
    entry = json.loads(read(v.root, ".discarded.log").strip())
    assert entry["candidate"] == CANDIDATE.to_json()
    assert "free filename" in entry["reason"]
    assert entry["similar"] == []  # every line carries the field
    v.close()


async def test_cutoffs_that_set_no_ceiling_at_all_are_refused(make_vault: MakeVault) -> None:
    # `Cutoffs()` type-checks, which would make the mandate cosmetic: with no
    # ceiling the search returns the least unrelated note and the gate acts on it.
    v = await open_gate(make_vault(VAULT), CREATE, Cutoffs())
    with pytest.raises(VaultError, match="cutoffs"):
        await v.propose(CANDIDATE)
    v.close()


async def test_the_facade_refuses_to_propose_without_a_decider_and_cutoffs(
    make_vault: MakeVault,
) -> None:
    v = open(make_vault(VAULT), EMBEDDER)
    await v.reindex()
    with pytest.raises(VaultError, match="gate"):
        await v.propose(CANDIDATE)
    assert len(await v.search("acme")) > 0  # the rest of the facade works
    assert (await v.doctor()).stale == []
    v.close()


def test_write_new_claims_a_name_and_a_second_writer_loses_it(make_vault: MakeVault) -> None:
    """`os.replace` lets a second writer overwrite the first. A create claims its
    filename with the write itself, so only one of two racing writers wins."""
    from wilcus_vault.paths import write_new

    root = make_vault({})
    target = root / "note.md"
    assert write_new(target, "first\n") is True
    assert write_new(target, "second\n") is False
    assert target.read_text() == "first\n"
    assert [p.name for p in root.iterdir()] == ["note.md"]  # no temp file left behind


async def test_a_filename_taken_after_the_index_check_is_never_overwritten(
    make_vault: MakeVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window between "is this name free?" and the write: another writer
    takes it. The candidate must move to the next slug, never clobber."""
    from wilcus_vault.paths import write_new

    root = make_vault(VAULT)
    taken: list[Path] = []

    def racing_write_new(abs_path: Path, text: str) -> bool:
        if not taken:  # first attempt only: someone else got there first
            taken.append(abs_path)
            abs_path.write_text("not ours\n", encoding="utf-8")
        return write_new(abs_path, text)

    monkeypatch.setattr("wilcus_vault.gate_write.write_new", racing_write_new)
    v = await open_gate(root, CREATE)
    r = await v.propose(CANDIDATE)

    assert r.action == "create"
    assert r.path is not None and r.path.endswith("-2.md")  # moved on, did not clobber
    assert taken[0].read_text() == "not ours\n"  # the other writer's note stands
    v.close()


async def test_the_closing_pass_still_sees_a_note_edited_behind_our_back(
    make_vault: MakeVault,
) -> None:
    """Freshness is why the closing pass is whole: an out-of-band edit has to reach
    the index, or the next call's similarity search is blind to it and duplicates it.

    Asserted on the indexed hash rather than on search: a stale row keeps its old
    title and body, which can still match a query for reasons that have nothing to
    do with the edit.
    """
    v = await open_gate(make_vault(VAULT), CREATE)
    await v.propose(CANDIDATE)

    rel = "notes/support-rota.md"
    edited = Path(v.root) / rel
    edited.write_text("# Support rota\n\nThe pager rota moved to the renewal calendar.\n")
    fresh = parse_note(edited.read_text(), rel).hash

    db = open_db(db_path(v.root))
    try:
        stale = db.execute("select hash from notes where path = ?", (rel,)).fetchone()["hash"]
        assert stale != fresh  # nothing has told the index yet
        await v.propose(replace(CANDIDATE, title="Another note"))  # any write runs the pass
        indexed = db.execute("select hash from notes where path = ?", (rel,)).fetchone()["hash"]
        assert indexed == fresh, "the closing pass did not re-read a note edited on disk"
    finally:
        db.close()
        v.close()

"""Write-gate evals: create, and the note-authoring corners of the gate."""

import json
import re
from dataclasses import replace

import pytest
from conftest import MakeVault
from fakes import fixed_decider
from gate_common import CANDIDATE, EMBEDDER, NOTHING_SIMILAR, OLD_NOTE, VAULT, open_gate, read

from wilcus_vault.decision import DeciderInput, Decision
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

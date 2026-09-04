"""Write-gate evals: a VaultContext is stamped as provenance on notes the gate authors."""

import re
from dataclasses import replace

import pytest
from conftest import MakeVault
from fakes import fixed_decider
from gate_common import CANDIDATE, CTX, VAULT, fm, open_gate, read

from wilcus_vault.decision import Decision
from wilcus_vault.scope import VaultContext
from wilcus_vault.term import VaultError

TARGET = "notes/acme-renewal.md"
CREATE = fixed_decider(Decision("create"))
UPDATE = fixed_decider(Decision("update", TARGET, "# Acme renewal\n\nRenewal closes 2026-03-01.\n"))
LEGACY = "id: 01234 # legacy account number, must survive a gate write"


async def test_a_ctx_stamps_provenance_on_a_note_the_gate_authors(make_vault: MakeVault) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    r = await v.propose(CANDIDATE, CTX)
    assert r.path is not None
    front = fm(v.root, r.path)
    assert (front["vault_agent"], front["vault_source"]) == ("core/scheduler", "task-42")
    v.close()


async def test_updates_provenance_says_exactly_who_made_this_call_set_and_unset(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), UPDATE)
    await v.propose(CANDIDATE, CTX)
    raw = read(v.root, TARGET)
    # patched textually, so the note we did not author is otherwise untouched
    assert LEGACY in raw
    assert "rate: 1.0" in raw
    front = fm(v.root, TARGET)
    assert (front["vault_agent"], front["vault_source"]) == ("core/scheduler", "task-42")

    # vault_agent is "who wrote this note last", not a growing list of writers
    await v.propose(CANDIDATE, VaultContext("core/librarian", "task-43"))
    assert len(re.findall(r"^vault_agent:", read(v.root, TARGET), re.M)) == 1
    front = fm(v.root, TARGET)
    assert (front["vault_agent"], front["vault_source"]) == ("core/librarian", "task-43")

    # a ctx with no source clears the last one: core/archivist beside task-43
    # would be a pairing that never happened
    await v.propose(CANDIDATE, VaultContext("core/archivist"))
    cleared = fm(v.root, TARGET)
    assert cleared["vault_agent"] == "core/archivist"
    assert "vault_source" not in cleared

    # and no ctx stamps nothing, including nothing left over from before
    await v.propose(CANDIDATE)
    bare = fm(v.root, TARGET)
    assert "vault_agent" not in bare
    assert "vault_source" not in bare
    # the human's frontmatter survived all four writes untouched
    assert LEGACY in read(v.root, TARGET)
    v.close()


async def test_a_yaml_hostile_agent_name_stays_one_quoted_line_and_invents_no_keys(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), UPDATE)
    # a value that would close the frontmatter block and open a key of its own
    evil = "x\n---\ninjected: true"
    await v.propose(CANDIDATE, VaultContext(evil, evil))

    raw = read(v.root, TARGET)
    assert len(re.findall(r"^vault_agent:", raw, re.M)) == 1
    assert len(re.findall(r"^vault_source:", raw, re.M)) == 1
    front = fm(v.root, TARGET)
    assert (front["vault_agent"], front["vault_source"], front["type"]) == (evil, evil, "customer")
    assert "injected" not in front  # text, not syntax
    v.close()


@pytest.mark.parametrize("agent", ["", "  \t"])
async def test_a_ctx_that_names_no_agent_is_refused_before_anything_is_written(
    make_vault: MakeVault, agent: str
) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    with pytest.raises(VaultError, match="agent"):
        await v.propose(CANDIDATE, VaultContext(agent))
    assert not (v.root / "notes/acme-renewal-2026.md").exists()
    v.close()


async def test_supersede_stamps_the_successor_and_does_not_restamp_the_retired_note(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), fixed_decider(Decision("supersede", TARGET)))
    r = await v.propose(CANDIDATE, CTX)
    assert r.path is not None
    front = fm(v.root, r.path)
    assert (front["vault_agent"], front["vault_source"]) == ("core/scheduler", "task-42")

    # marking the old note is bookkeeping, not authorship: the superseding
    # agent is already recorded on the successor
    old = fm(v.root, TARGET)
    assert old["superseded_by"] == "notes/acme-renewal-2026.md"
    assert "vault_agent" not in old
    assert "vault_source" not in old
    v.close()


async def test_no_ctx_stamps_nothing_and_a_ctx_without_a_source_stamps_the_agent_alone(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    bare = await v.propose(CANDIDATE)
    assert bare.path is not None
    assert "vault_" not in read(v.root, bare.path)

    agent_only = await v.propose(
        replace(CANDIDATE, title="Acme renewal 2027"), VaultContext("core/scheduler")
    )
    assert agent_only.path is not None
    raw = read(v.root, agent_only.path)
    assert "vault_agent" in raw
    assert "vault_source" not in raw  # absent, not an empty key
    v.close()

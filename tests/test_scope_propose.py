"""Per-agent scopes on the write gate: the namespace check, what the decider
sees, and a decision that targets a note the agent may not write."""

from dataclasses import replace

import pytest
from test_scope_policy import POLICY
from test_scope_vault import CANDIDATE, NOTES, SCHEDULER, OpenVault
from test_scope_vault import open_vault as _open_vault

from wilcus_vault.decision import DeciderInput, Decision, gate_prompt
from wilcus_vault.term import VaultError

open_vault = _open_vault  # the shared fixture, registered in this module for pytest


async def test_propose_refuses_an_unwritable_namespace_before_the_decider_runs(
    open_vault: OpenVault,
) -> None:
    seen: list[DeciderInput] = []

    async def decider(input: DeciderInput) -> Decision:
        seen.append(input)
        return Decision("create")

    v = await open_vault(POLICY, decider)
    # Fail fast: no model spend on a doomed write.
    with pytest.raises(VaultError, match="may not write"):
        await v.propose(replace(CANDIDATE, namespace="ledger"), NOTES)
    with pytest.raises(VaultError, match="may not write"):
        await v.propose(replace(CANDIDATE, namespace=None), NOTES)
    assert seen == []

    r = await v.propose(CANDIDATE, NOTES)
    assert (r.action, r.path) == ("create", "notes/acme-renewal-2026.md")
    assert len(seen) == 1


async def test_propose_canonicalizes_the_namespace_before_checking_it(
    open_vault: OpenVault,
) -> None:
    seen: list[DeciderInput] = []

    async def decider(input: DeciderInput) -> Decision:
        seen.append(input)
        return Decision("create")

    v = await open_vault(POLICY, decider)
    # `notes/../ledger` is inside the vault, so confinement passes; it starts
    # with `notes/`, so a raw check passes, and the file would land in `ledger/`.
    # The check and the write have to be looking at the same string.
    for namespace in ["notes/../ledger", "./ledger", "ledger/", "ledger/../ledger"]:
        with pytest.raises(VaultError, match="may not write"):
            await v.propose(replace(CANDIDATE, namespace=namespace), NOTES)
    # An absolute namespace is not a spelling of anything inside the vault, so
    # it is refused by the older rail before the scope is consulted at all.
    with pytest.raises(VaultError, match="outside the vault"):
        await v.propose(replace(CANDIDATE, namespace="/ledger"), NOTES)
    assert seen == []
    assert v.list("ledger", SCHEDULER) == ["ledger/q3.md"]  # nothing landed


async def test_only_readable_notes_feed_the_decider_and_unwritable_ones_are_read_only(
    open_vault: OpenVault,
) -> None:
    seen: list[DeciderInput] = []

    async def decider(input: DeciderInput) -> Decision:
        seen.append(input)
        return Decision("discard")

    v = await open_vault(POLICY, decider)
    await v.propose(CANDIDATE, NOTES)

    similar = seen[0].similar
    assert len(similar) > 1
    by_path = {s.note.path: s for s in similar}
    # An agent must not have another agent's note bodies quoted back to it.
    assert "secret/plans.md" not in by_path
    assert "ledger/q3.md" in by_path
    assert by_path["ledger/q3.md"].read_only is True
    assert by_path["notes/acme-renewal.md"].read_only is False

    prompt = gate_prompt(seen[0])
    assert "ledger/q3.md (read-only)" in prompt
    assert "notes/acme-renewal.md (read-only)" not in prompt
    assert "nobody else may read" not in prompt


async def test_a_decision_targeting_an_unwritable_note_falls_back_to_create(
    open_vault: OpenVault,
) -> None:
    async def decider(_input: DeciderInput) -> Decision:
        return Decision(
            "update",
            target="ledger/q3.md",
            body="# Q3 ledger\n\nRewritten by an agent that may not write here.\n",
        )

    v = await open_vault(POLICY, decider)
    before = (await v.get("ledger/q3.md", NOTES)).hash  # type: ignore[union-attr]
    # Like a target that failed check-and-write twice: the candidate always lands.
    r = await v.propose(CANDIDATE, NOTES)
    assert (r.action, r.path, r.fell_back) == ("create", "notes/acme-renewal-2026.md", True)
    assert (await v.get("ledger/q3.md", NOTES)).hash == before  # type: ignore[union-attr]

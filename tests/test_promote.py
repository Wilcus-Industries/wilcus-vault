"""Promote: one note through the write gate into a namespace, then removed —
unless it changed while the gate ran. Deterministic embedder, fake deciders."""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from conftest import MakeVault
from gate_common import CUTOFFS, fm

from wilcus_vault.decision import DeciderInput, Decision
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.gate import GateOptions
from wilcus_vault.scope import ScopePolicy, VaultContext
from wilcus_vault.term import VaultError
from wilcus_vault.vault import Vault, open

PROPOSAL = "proposals/peer/acme-2026.md"
FILES = {
    "shared/acme-renewal.md": "# Acme renewal\n\nThe Acme renewal closes in March.\n",
    PROPOSAL: "---\ntype: customer\n---\n# Acme renewal 2026\n\n"
    "The Acme renewal closes in March 2026 at the agreed pricing.\n",
}
# The swarm layout: the orchestrator promotes, a peer writes only its own proposals.
POLICY: ScopePolicy = {
    "orchestrator": [{"prefix": "", "read": True, "write": True}],
    "peer": [
        {"prefix": "shared/", "read": True},
        {"prefix": "proposals/peer/", "read": True, "write": True},
    ],
    # writes shared/ but only reads proposals, so it could never remove one
    "reader": [{"prefix": "", "read": True}, {"prefix": "shared/", "write": True}],
    "outsider": [{"prefix": "shared/", "read": True, "write": True}],
}
ORCHESTRATOR = VaultContext("orchestrator")

Decide = Callable[[Path, DeciderInput], Decision]
Promoting = Callable[[Decide], Awaitable[tuple[Vault, list[DeciderInput]]]]


@pytest.fixture
async def promoting(
    make_vault: MakeVault, embedder: TokenOverlapEmbedder
) -> AsyncIterator[Promoting]:
    opened: list[Vault] = []

    async def make(decide: Decide) -> tuple[Vault, list[DeciderInput]]:
        root = make_vault(FILES)
        seen: list[DeciderInput] = []

        async def decider(input: DeciderInput) -> Decision:
            seen.append(input)
            return decide(root, input)

        v = open(root, embedder, gate=GateOptions(decider, CUTOFFS), scopes=POLICY)
        opened.append(v)
        await v.reindex()  # the proposal is indexed, as it is in a live vault
        return v, seen

    yield make
    for v in opened:
        v.close()


@pytest.mark.parametrize(
    ("decision", "path"),
    [
        (Decision("create"), "shared/acme-renewal-2026.md"),
        (Decision("update", target="shared/acme-renewal.md"), "shared/acme-renewal.md"),
        (Decision("supersede", target="shared/acme-renewal.md"), "shared/acme-renewal-2026.md"),
        (Decision("discard"), None),
    ],
    ids=["create", "update", "supersede", "discard"],
)
async def test_every_gate_action_removes_the_proposal_and_its_index_row(
    promoting: Promoting, decision: Decision, path: str | None
) -> None:
    v, seen = await promoting(lambda _root, _input: decision)
    r = await v.promote(PROPOSAL, "shared", ORCHESTRATOR)
    assert (r.action, r.path, r.removed) == (decision.action, path, True)
    assert not (v.root / PROPOSAL).exists()
    assert v.list("proposals", ORCHESTRATOR) == []  # the row went with the file

    # The candidate is the proposal placed in the namespace, never judged against
    # itself: shown its own text, a decider finds it already written down.
    candidate = seen[0].candidate
    assert (candidate.title, candidate.type, candidate.namespace) == (
        "Acme renewal 2026",
        "customer",
        "shared",
    )
    similar = [s.note.path for s in seen[0].similar]
    assert "shared/acme-renewal.md" in similar
    assert PROPOSAL not in similar
    if path is None:
        assert "closes in March 2026" in (v.root / ".discarded.log").read_text()
    else:
        assert fm(v.root, path)["vault_source"] == PROPOSAL  # where it came from


async def test_a_proposal_edited_while_the_gate_ran_is_kept(promoting: Promoting) -> None:
    edit = "# Acme renewal 2026\n\nA peer's correction: it closes in April.\n"

    def peer_edits(root: Path, _input: DeciderInput) -> Decision:
        (root / PROPOSAL).write_text(edit)
        return Decision("create")

    v, _ = await promoting(peer_edits)
    r = await v.promote(PROPOSAL, "shared", ORCHESTRATOR)
    assert (r.action, r.path, r.removed) == ("create", "shared/acme-renewal-2026.md", False)
    # Nothing is lost: what was read is in shared/, and the peer's edit stays put.
    assert "closes in March 2026" in (v.root / "shared/acme-renewal-2026.md").read_text()
    assert (v.root / PROPOSAL).read_text() == edit
    assert v.list("proposals", ORCHESTRATOR) == [PROPOSAL]


async def test_an_agent_that_cannot_remove_the_proposal_or_write_the_namespace_is_refused_first(
    promoting: Promoting,
) -> None:
    v, seen = await promoting(lambda _root, _input: Decision("create"))
    # The peer may write its own proposal, but not shared/.
    with pytest.raises(VaultError, match='"peer" may not write to shared/'):
        await v.promote(PROPOSAL, "shared", VaultContext("peer"))
    with pytest.raises(VaultError, match=f'"reader" may not write {PROPOSAL}'):
        await v.promote(PROPOSAL, "shared", VaultContext("reader"))
    # A proposal the agent may not read is the same answer as one that is not there.
    for ctx, path in [
        (VaultContext("outsider"), PROPOSAL),
        (ORCHESTRATOR, "proposals/peer/nothing.md"),
    ]:
        with pytest.raises(VaultError, match=f"promote: no note at {path}"):
            await v.promote(path, "shared", ctx)
    assert seen == []  # all refused before any model spend
    assert (v.root / PROPOSAL).exists()


async def test_a_source_the_caller_gives_is_kept(promoting: Promoting) -> None:
    v, _ = await promoting(lambda _root, _input: Decision("create"))
    await v.promote(PROPOSAL, "shared", VaultContext("orchestrator", "task-9"))
    assert fm(v.root, "shared/acme-renewal-2026.md")["vault_source"] == "task-9"

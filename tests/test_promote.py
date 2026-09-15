"""Promote: one note through the write gate into a namespace, then removed —
unless it changed while the gate ran. Deterministic embedder, fake deciders."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from gate_common import fm
from test_scope_vault import OpenVault
from test_scope_vault import open_vault as _open_vault

from wilcus_vault.db import db_path, open_db
from wilcus_vault.decision import Candidate, Decider, DeciderInput, Decision
from wilcus_vault.note import parse_note
from wilcus_vault.scope import ScopePolicy, VaultContext
from wilcus_vault.term import VaultError

open_vault = _open_vault  # the shared fixture, registered in this module for pytest

# Named by its title's slug, as `vault propose` names the notes it writes.
PROPOSAL = "proposals/peer/acme-renewal-2026.md"
TEXT = (
    "---\ntype: customer\n---\n# Acme renewal 2026\n\n"
    "The Acme renewal closes in March 2026 at the agreed pricing.\n"
)
FILES = {
    "shared/acme-renewal.md": "# Acme renewal\n\nThe Acme renewal closes in March.\n",
    PROPOSAL: TEXT,
    # the peer's own copy outside shared/: the closest match there is, and no place to land
    "roles/peer/scratch.md": TEXT,
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
# A body of the decider's own, which can drop what the proposal said.
LOSSY = "# Acme renewal\n\nMerged, and the 2026 date lost on the way.\n"


def deciding(
    decision: Decision, seen: list[DeciderInput], meanwhile: Callable[[], object] = lambda: None
) -> Decider:
    """Records what the decider was shown, runs `meanwhile`, then answers `decision`."""

    async def decide(input: DeciderInput) -> Decision:
        seen.append(input)
        meanwhile()
        return decision

    return decide


@pytest.mark.parametrize(
    ("decision", "path"),
    [
        (Decision("create"), "shared/acme-renewal-2026.md"),
        (Decision("update", target="shared/acme-renewal.md", body=LOSSY), "shared/acme-renewal.md"),
        (Decision("supersede", target="shared/acme-renewal.md"), "shared/acme-renewal-2026.md"),
        (Decision("discard"), None),
    ],
    ids=["create", "update", "supersede", "discard"],
)
async def test_every_gate_action_removes_the_proposal_and_keeps_its_text(
    open_vault: OpenVault, decision: Decision, path: str | None
) -> None:
    seen: list[DeciderInput] = []
    v = await open_vault(POLICY, deciding(decision, seen), FILES)
    r = await v.promote(PROPOSAL, "shared", ORCHESTRATOR)
    # the proposal's own stem does not push the new note to `-2`
    assert (r.action, r.path, r.removed) == (decision.action, path, True)
    assert not (v.root / PROPOSAL).exists()
    assert v.list("proposals", ORCHESTRATOR) == []  # the row went with the file

    # The candidate is the parsed note placed in the namespace, judged against it alone.
    parsed = parse_note(TEXT, PROPOSAL)
    candidate = seen[0].candidate
    assert (candidate.title, candidate.type, candidate.namespace) == (
        "Acme renewal 2026",
        "customer",
        "shared",
    )
    assert candidate.body == parsed.body
    assert [s.note.path for s in seen[0].similar] == ["shared/acme-renewal.md"]

    # Logged whole before the file goes; a discard is already the gate's own entry.
    lines = (v.root / ".discarded.log").read_text().splitlines()
    logged = [json.loads(line) for line in lines]
    reason = None if path is None else "promoted"
    assert [(e.get("reason"), e.get("path"), e["candidate"]["body"]) for e in logged] == [
        (reason, path, parsed.body)
    ]
    if path is not None:
        assert fm(v.root, path)["vault_source"] == PROPOSAL  # where it came from


async def test_a_proposal_edited_while_the_gate_ran_is_kept(open_vault: OpenVault) -> None:
    edit = "# Acme renewal 2026\n\nA peer's correction: it closes in April.\n"
    seen: list[DeciderInput] = []

    def peer_edits() -> None:
        (v.root / PROPOSAL).write_text(edit)

    v = await open_vault(POLICY, deciding(Decision("create"), seen, peer_edits), FILES)
    r = await v.promote(PROPOSAL, "shared", ORCHESTRATOR)
    assert (r.action, r.path, r.removed) == ("create", "shared/acme-renewal-2026.md", False)
    # Nothing is lost: what was read is in shared/, and the peer's edit stays put.
    assert "closes in March 2026" in (v.root / "shared/acme-renewal-2026.md").read_text()
    assert (v.root / PROPOSAL).read_text() == edit
    assert v.list("proposals", ORCHESTRATOR) == [PROPOSAL]
    assert not (v.root / ".discarded.log").exists()  # a kept proposal needs no copy


async def test_a_proposal_that_cannot_be_promoted_is_refused_before_the_decider_and_kept(
    open_vault: OpenVault,
) -> None:
    broken = "proposals/peer/broken.md"
    garbled = "---\ntitle: [unclosed\n---\n# Broken\n\nThe Acme renewal, garbled.\n"
    seen: list[DeciderInput] = []
    v = await open_vault(POLICY, deciding(Decision("create"), seen), {**FILES, broken: garbled})
    # The peer may write its own proposal, but not shared/.
    with pytest.raises(VaultError, match='"peer" may not write to shared/'):
        await v.promote(PROPOSAL, "shared", VaultContext("peer"))
    with pytest.raises(VaultError, match=f'"reader" may not write {PROPOSAL}'):
        await v.promote(PROPOSAL, "shared", VaultContext("reader"))
    # Its broken block would otherwise be copied into the shared note's body.
    with pytest.raises(VaultError, match=f"promote: {broken} has malformed frontmatter"):
        await v.promote(broken, "shared", ORCHESTRATOR)
    # A proposal the agent may not read is the same answer as one that is not there.
    for ctx, path in [
        (VaultContext("outsider"), PROPOSAL),
        (ORCHESTRATOR, "proposals/peer/nothing.md"),
    ]:
        with pytest.raises(VaultError, match=f"promote: no note at {path}"):
            await v.promote(path, "shared", ctx)
    assert seen == []  # all refused before any model spend
    assert (v.root / PROPOSAL).read_text() == TEXT
    assert (v.root / broken).read_text() == garbled


@pytest.mark.parametrize(
    ("scopes", "ctx", "shown"),
    [
        # a deny under the namespace still holds once the call is confined to it
        (
            {
                "orchestrator": [
                    {"prefix": "", "read": True, "write": True},
                    {"prefix": "shared/private/", "read": False, "write": False},
                ]
            },
            ORCHESTRATOR,
            ["shared/acme-renewal.md"],
        ),
        (None, None, ["shared/acme-renewal.md", "shared/private/acme-renewal.md"]),
    ],
    ids=["policy", "no-policy"],
)
async def test_a_note_outside_the_namespace_is_never_shown_or_targeted(
    open_vault: OpenVault, scopes: ScopePolicy | None, ctx: VaultContext | None, shown: list[str]
) -> None:
    seen: list[DeciderInput] = []
    files = {**FILES, "shared/private/acme-renewal.md": TEXT}
    peer_copy = Decision("update", target="roles/peer/scratch.md")
    v = await open_vault(scopes, deciding(peer_copy, seen), files)
    # roles/ is writable here, but a promotion that lands there never reaches shared/.
    with pytest.raises(VaultError, match="not among the similar notes"):
        await v.promote(PROPOSAL, "shared", ctx)
    assert sorted(s.note.path for s in seen[0].similar) == shown
    assert (v.root / "roles/peer/scratch.md").read_text() == TEXT
    assert (v.root / PROPOSAL).read_text() == TEXT


async def test_a_source_the_caller_gives_is_kept(open_vault: OpenVault) -> None:
    v = await open_vault(POLICY, deciding(Decision("create"), []), FILES)
    await v.promote(PROPOSAL, "shared", VaultContext("orchestrator", "task-9"))
    assert fm(v.root, "shared/acme-renewal-2026.md")["vault_source"] == "task-9"


def linked_from(root: Path, rel: str) -> list[str]:
    """The notes `rel`'s links resolve to, as the index holds them."""
    db = open_db(db_path(root))
    try:
        rows = db.execute(
            """select t.path from edges e join notes f on f.id = e.from_id
               join notes t on t.id = e.to_id where f.path = ?""",
            (rel,),
        ).fetchall()
        return [r["path"] for r in rows]
    finally:
        db.close()


@pytest.mark.parametrize(
    ("decision", "freshness"),
    [
        (Decision("create"), 0.0),
        (Decision("supersede", target="shared/acme-renewal.md"), 0.0),
        (Decision("create"), 3600.0),  # inside the window: only the paths written are indexed
    ],
    ids=["create", "supersede", "create-in-window"],
)
async def test_a_bare_link_to_the_proposal_follows_it_into_the_namespace(
    open_vault: OpenVault, decision: Decision, freshness: float
) -> None:
    linker, links = (
        "roles/peer/renewals.md",
        "# Renewals\n\nThe date is in [[acme-renewal-2026]].\n",
    )
    files = {**FILES, linker: links}
    v = await open_vault(POLICY, deciding(decision, []), files, freshness=freshness)
    if freshness:
        rota = Candidate(title="Support rota", body="Who carries the pager.\n", namespace="shared")
        await v.propose(rota, ORCHESTRATOR)  # walks, and opens the window
    r = await v.promote(PROPOSAL, "shared", ORCHESTRATOR)
    # Never qualified to the proposal's path, which the promotion removes: the new
    # note takes the stem over, as a rename would.
    assert (v.root / linker).read_text() == links
    assert linked_from(v.root, linker) == [r.path]


@pytest.mark.parametrize("path", [PROPOSAL, "shared/acme-draft.md"], ids=["outside", "inside"])
async def test_the_decider_sees_at_most_n_notes_and_never_the_one_promoted(
    open_vault: OpenVault, path: str
) -> None:
    seen: list[DeciderInput] = []
    files = {**FILES, "shared/acme-draft.md": TEXT}
    v = await open_vault(POLICY, deciding(Decision("discard"), seen), files, n=1)
    await v.promote(path, "shared", ORCHESTRATOR)
    shown = [s.note.path for s in seen[0].similar]
    assert len(shown) == 1
    assert path not in shown

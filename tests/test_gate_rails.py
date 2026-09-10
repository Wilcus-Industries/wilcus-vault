"""Write-gate evals: path confinement. An LLM-derived string never names a raw path."""

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import MakeVault
from fakes import fixed_decider
from gate_common import CANDIDATE, CUTOFFS, EMBEDDER, VAULT, open_gate, read

from wilcus_vault.db import db_path, open_db
from wilcus_vault.decision import DeciderInput, Decision
from wilcus_vault.gate import GateOptions
from wilcus_vault.note import parse_note
from wilcus_vault.paths import confined_path, slugify
from wilcus_vault.term import VaultError
from wilcus_vault.vault import open

CREATE = fixed_decider(Decision("create"))


async def test_a_traversing_title_is_slugified_not_obeyed(make_vault: MakeVault) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    r = await v.propose(replace(CANDIDATE, title="../../evil"))
    assert r.path == "notes/evil.md"
    assert not (v.root.parent / "evil.md").exists()
    assert not (v.root.parent.parent / "evil.md").exists()
    assert slugify("../../evil") == "evil"
    assert slugify("TLS/SSL — notes..md") == "tls-ssl-notes-md"
    assert slugify("../..") is None
    v.close()


async def test_a_traversing_namespace_or_symlinked_dir_is_refused(make_vault: MakeVault) -> None:
    v = await open_gate(make_vault(VAULT), CREATE)
    with pytest.raises(VaultError, match="outside the vault"):
        await v.propose(replace(CANDIDATE, namespace="../x"))
    with pytest.raises(VaultError, match="outside the vault"):
        await v.propose(replace(CANDIDATE, namespace="/etc"))
    assert not (v.root.parent / "x").exists()
    # the scan skips dot-directories, so a note written there could never index
    with pytest.raises(VaultError, match="hidden"):
        await v.propose(replace(CANDIDATE, namespace=".vault"))

    os.symlink(v.root / "notes", v.root / "linked")
    with pytest.raises(VaultError, match="symlink"):
        await v.propose(replace(CANDIDATE, namespace="linked"))

    # A NUL reaches `lstat` as a raw ValueError at the caller unless the rail
    # refuses it first, like every other path it will not build.
    with pytest.raises(VaultError, match=r"^vault: no\?pe.*NUL byte"):
        await v.propose(replace(CANDIDATE, namespace="no\0pe"))
    with pytest.raises(VaultError, match="NUL byte"):
        confined_path(v.root, "notes/\0.md")
    v.close()


async def test_a_namespace_is_canonicalized_and_refused_before_the_decider(
    make_vault: MakeVault,
) -> None:
    seen: list[DeciderInput] = []

    async def decider(input: DeciderInput) -> Decision:
        seen.append(input)
        return Decision("create")

    v = await open_gate(make_vault(VAULT), decider)
    # Every spelling of one namespace names one directory, and the path the
    # caller gets back is the canonical one: it is an identity they may store,
    # and (with scopes) the string the write check was made against.
    for i, namespace in enumerate(["notes", "notes/", "notes//", "./notes", "other/../notes"]):
        r = await v.propose(replace(CANDIDATE, namespace=namespace, title=f"Canonical {i}"))
        assert r.path == f"notes/canonical-{i}.md"
        assert f"title: Canonical {i}" in read(v.root, r.path)
    assert slugify("Canonical 0") == "canonical-0"

    # A namespace that is not a path at all is refused *before* the decider
    # runs: a doomed write should not cost a model call, and it is refused
    # whatever the decider would have answered, which a `discard` used to slip
    # past because only the create path ever built the filename.
    before = len(seen)
    with pytest.raises(VaultError, match="outside the vault"):
        await v.propose(replace(CANDIDATE, namespace="../x"))
    with pytest.raises(VaultError, match="hidden"):
        await v.propose(replace(CANDIDATE, namespace=".vault"))
    assert len(seen) == before
    v.close()

    discards = await open_gate(make_vault(VAULT), fixed_decider(Decision("discard")))
    with pytest.raises(VaultError, match="outside the vault"):
        await discards.propose(replace(CANDIDATE, namespace="../x"))
    discards.close()


def test_the_vault_root_is_not_walked_so_a_symlinked_root_opens(make_vault: MakeVault) -> None:
    link = make_vault({}) / "vault-link"
    os.symlink(make_vault({}), link)
    # The path *is* the root: there is no segment between it and itself to
    # check, and lstat-ing the root would refuse every vault whose own path is a
    # symlink, which the scan is perfectly happy to walk.
    assert confined_path(link, ".") == Path(os.path.abspath(link))
    assert confined_path(link, "") == Path(os.path.abspath(link))
    # Below the root the rule is unchanged.
    with pytest.raises(VaultError, match="outside the vault"):
        confined_path(link, "../elsewhere.md")


def _indexed_hash(root: Path, rel: str) -> str:
    db = open_db(db_path(root))
    try:
        return str(db.execute("select hash from notes where path = ?", (rel,)).fetchone()["hash"])
    finally:
        db.close()


async def test_a_freshness_window_lets_a_burst_of_writes_share_one_walk(
    make_vault: MakeVault,
) -> None:
    """The closing pass re-reads every note, so a burst of writes pays for the vault
    once per note. A window says how stale the gate's view of the files may be, and
    the writes inside it share one walk."""
    root = make_vault(VAULT)
    v = open(
        root,
        EMBEDDER,
        gate=GateOptions(decider=CREATE, cutoffs=CUTOFFS, freshness=3600),
    )
    await v.reindex()
    try:
        rel = "notes/support-rota.md"
        await v.propose(CANDIDATE)  # first write: due, so it walks
        before = _indexed_hash(root, rel)

        (Path(root) / rel).write_text("# Support rota\n\nThe rota moved to the calendar.\n")
        r = await v.propose(replace(CANDIDATE, title="Second note"))

        # Inside the window an edit made outside the vault API is not looked for...
        assert _indexed_hash(root, rel) == before
        # ...but a note the gate wrote itself is never stale, window or no window.
        assert r.path is not None
        assert r.path in [h.path for h in await v.search("second note")]
    finally:
        v.close()


async def test_a_write_inside_the_window_does_not_push_the_window_along(
    make_vault: MakeVault,
) -> None:
    """Bounded is the entire claim the window makes, and it rests on the clock
    marking walks rather than writes. Mark it on every write and each one inside
    the window shifts the deadline forward: a caller writing faster than its own
    window then never walks again, and the blindness stops being bounded at all.

    Three writes, because two cannot show it — the middle one is what moves a
    clock that should not have moved.
    """
    root = make_vault(VAULT)
    v = open(root, EMBEDDER, gate=GateOptions(decider=CREATE, cutoffs=CUTOFFS, freshness=0.1))
    await v.reindex()
    try:
        rel = "notes/support-rota.md"
        await v.propose(CANDIDATE)  # walks, and starts the window
        (Path(root) / rel).write_text("# Support rota\n\nThe rota moved to the calendar.\n")
        fresh = parse_note((Path(root) / rel).read_text(), rel).hash

        await asyncio.sleep(0.06)
        await v.propose(replace(CANDIDATE, title="Second note"))  # inside: must not walk
        assert _indexed_hash(root, rel) != fresh

        await asyncio.sleep(0.06)  # now past the window measured from the first write
        await v.propose(replace(CANDIDATE, title="Third note"))
        assert _indexed_hash(root, rel) == fresh, "a write inside the window moved the deadline"
    finally:
        v.close()

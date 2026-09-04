"""Write-gate evals: path confinement. An LLM-derived string never names a raw path."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import MakeVault
from fakes import fixed_decider
from gate_common import CANDIDATE, VAULT, open_gate, read

from wilcus_vault.decision import DeciderInput, Decision
from wilcus_vault.paths import confined_path, slugify
from wilcus_vault.term import VaultError

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

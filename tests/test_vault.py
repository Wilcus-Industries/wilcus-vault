"""The facade's two direct read paths: `get` reads the file, `list` reads the
index. Files are truth, so a stale or missing index row never changes what
`get` hands back, and only a regular `.md` file under the root is a note at all."""

import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from conftest import MakeVault, write_note
from fakes import fixed_decider

from wilcus_vault import Decision, GateOptions, Vault, VaultError, open
from wilcus_vault.decision import Candidate
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.search_sql import Cutoffs

EMBEDDER = TokenOverlapEmbedder()
CUTOFFS = Cutoffs(distance_ceiling=0.9, bm25_ceiling=0)
# Only the symlinked-root test writes; every vault gets a gate anyway.
GATE = GateOptions(decider=fixed_decider(Decision("create")), cutoffs=CUTOFFS)

VAULT = {
    "ledger/q3.md": "---\ntitle: Q3\n---\n\nThe Q3 ledger.\n",
    "ledger/q4.md": "# Q4\n\nThe Q4 ledger.\n",
    # The sibling namespace a raw `startswith` prefix would wrongly sweep in.
    "ledger-archive/q3.md": "# Archived Q3\n\nSuperseded numbers.\n",
    "root-note.md": "# Root note\n\nAt the vault root.\n",
}

OpenAt = Callable[[str | Path], Vault]
OpenVault = Callable[..., Awaitable[Vault]]


@pytest.fixture
def open_at() -> Iterator[OpenAt]:
    """Opens vaults and closes every one at teardown, before the fixture dirs go."""
    opened: list[Vault] = []

    def at(root: str | Path) -> Vault:
        v = open(root, EMBEDDER, gate=GATE)
        opened.append(v)
        return v

    yield at
    for v in opened:
        v.close()


@pytest.fixture
def open_vault(make_vault: MakeVault, open_at: OpenAt) -> OpenVault:
    async def opener(files: dict[str, str] = VAULT) -> Vault:
        v = open_at(make_vault(files))
        await v.reindex()
        return v

    return opener


async def test_get_reads_the_file_never_the_index_row(open_vault: OpenVault) -> None:
    v = await open_vault()
    # One note edited behind the index's back, one never indexed at all.
    write_note(v.root, "ledger/q3.md", "---\ntitle: Q3 (corrected)\n---\n\nRevised.\n")
    write_note(v.root, "ledger/q5.md", "# Q5\n\nNot in the index.\n")

    q3 = await v.get("ledger/q3.md")
    assert q3 is not None
    assert (q3.path, q3.slug, q3.title) == ("ledger/q3.md", "q3", "Q3 (corrected)")
    assert "Revised." in q3.body

    assert "ledger/q5.md" not in v.list()  # the index has not caught up
    q5 = await v.get("ledger/q5.md")
    assert q5 is not None and q5.title == "Q5"  # the file is the truth


async def test_get_absent_directory_symlink_and_non_md_are_none(
    open_vault: OpenVault, make_vault: MakeVault
) -> None:
    v = await open_vault()
    assert await v.get("ledger/nope.md") is None
    # Identity includes the extension: `ledger/q3` names no note.
    assert await v.get("ledger/q3") is None

    (v.root / "ledger" / "sub.md").mkdir()
    assert await v.get("ledger/sub.md") is None

    # A symlink is a second name for a file that may not be a note at all: the
    # scan skips them, so `get` must not serve one either. Least of all one
    # pointing clean out of the vault.
    outside = make_vault({"secret.md": "# Secret\n\nNot in this vault.\n"})
    os.symlink(v.root / "ledger" / "q4.md", v.root / "link.md")
    os.symlink(outside / "secret.md", v.root / "ledger" / "leak.md")
    assert await v.get("link.md") is None
    assert await v.get("ledger/leak.md") is None


async def test_get_normalizes_the_path_to_the_notes_identity(
    open_vault: OpenVault,
) -> None:
    v = await open_vault()
    # Every one of these names `ledger/q3.md`, and `path` is the identity a
    # caller stores, so it has to come back canonical whichever form went in.
    for rel in [
        "ledger/q3.md",
        "./ledger/q3.md",
        "ledger//q3.md",
        "ledger/../ledger/q3.md",
        str(v.root / "ledger" / "q3.md"),  # an absolute path inside the vault
    ]:
        note = await v.get(rel)
        assert note is not None and note.path == "ledger/q3.md", rel

    # A NUL is not a byte any filename holds: "no note there", not the raw
    # ValueError `lstat` throws at whoever called us.
    assert await v.get("ledger/q3.md\0") is None
    assert await v.get("led\0ger/q3.md") is None

    # Normalizing is not a way out of the vault.
    with pytest.raises(VaultError, match="outside the vault"):
        await v.get("./../../etc/passwd")
    with pytest.raises(VaultError, match="outside the vault"):
        await v.get("ledger/../../etc/passwd")


async def test_a_vault_opened_through_a_symlinked_root_still_reads_and_writes(
    make_vault: MakeVault, open_at: OpenAt
) -> None:
    real = make_vault(VAULT)
    link = make_vault({}) / "vault-link"
    os.symlink(real, link)

    v = open_at(link)
    await v.reindex()
    # A root-level note has nothing above it but the root, so this is the one
    # read that would lstat the root itself, and the root is the symlink.
    note = await v.get("root-note.md")
    assert note is not None and note.title == "Root note"
    assert "root-note.md" in v.list()

    r = await v.propose(Candidate("Through the link", "Written.\n", namespace="ledger"))
    assert (r.action, r.path) == ("create", "ledger/through-the-link.md")
    assert r.path is not None and (real / r.path).exists()


async def test_get_outside_root_dot_directory_or_symlinked_directory_is_refused(
    open_vault: OpenVault,
) -> None:
    v = await open_vault()
    with pytest.raises(VaultError, match="outside the vault"):
        await v.get("../../etc/passwd")
    with pytest.raises(VaultError, match="outside the vault"):
        await v.get("/etc/passwd")
    # The index lives in `.vault/`, and the scan skips dot-directories: nothing
    # in one is a note, whatever its extension.
    with pytest.raises(VaultError, match="hidden directory"):
        await v.get(".vault/index.db")
    with pytest.raises(VaultError, match="hidden directory"):
        await v.get(".vault/notes.md")

    os.symlink(v.root / "ledger", v.root / "linked")
    with pytest.raises(VaultError, match="symlink"):
        await v.get("linked/q4.md")


async def test_list_every_indexed_note_path_sorted(open_vault: OpenVault) -> None:
    v = await open_vault()
    assert v.list() == ["ledger-archive/q3.md", "ledger/q3.md", "ledger/q4.md", "root-note.md"]
    assert v.list("") == v.list()


async def test_list_prefix_matches_on_segment_boundaries(
    open_vault: OpenVault,
) -> None:
    v = await open_vault()
    ledger = ["ledger/q3.md", "ledger/q4.md"]
    assert v.list("ledger") == ledger
    assert v.list("ledger/") == ledger
    assert v.list("ledger-archive") == ["ledger-archive/q3.md"]
    # A leading or lone slash is the root namespace, not a path: "/" is the whole
    # vault, and no note path starts with one.
    assert v.list("/") == v.list()
    assert v.list("/ledger") == ledger
    # A prefix is a namespace, not a string match: neither half a segment nor a
    # namespace that does not exist brings anything back.
    assert v.list("ledg") == []
    assert v.list("nope") == []


async def test_list_follows_the_files(open_vault: OpenVault) -> None:
    v = await open_vault()
    write_note(v.root, "ledger/q5.md", "# Q5\n\nNew.\n")
    (v.root / "ledger" / "q4.md").unlink()
    await v.reindex()
    assert v.list("ledger") == ["ledger/q3.md", "ledger/q5.md"]


async def test_list_is_the_note_set_superseded_notes_are_listed(
    open_vault: OpenVault,
) -> None:
    v = await open_vault(
        {
            "a.md": "---\nsuperseded_by: b.md\n---\n\nThe old note.\n",
            "b.md": "# B\n\nThe new note.\n",
        }
    )
    assert v.list() == ["a.md", "b.md"]


async def test_get_a_path_that_cannot_hold_a_note_is_none_never_an_oserror(
    open_vault: OpenVault,
) -> None:
    """Only FileNotFoundError used to mean "no note", so every other way the
    filesystem says "nothing here" escaped as a raw OSError."""
    v = await open_vault()
    assert await v.get("ledger/q4.md/nested.md") is None  # ENOTDIR: a file used as a directory
    assert await v.get("ledger/" + "x" * 300 + ".md") is None  # ENAMETOOLONG

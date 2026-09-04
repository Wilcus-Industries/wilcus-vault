"""Write-gate evals: supersede writes the successor first, then marks the old note."""

import os
from pathlib import Path

import pytest
from conftest import MakeVault, write_note
from fakes import fixed_decider
from gate_common import CANDIDATE, VAULT, open_gate, read

from wilcus_vault.decision import DeciderInput, Decision
from wilcus_vault.paths import write_atomic

TARGET = "notes/acme-renewal.md"
SUCCESSOR = "notes/acme-renewal-2026.md"
SUPERSEDE = fixed_decider(Decision("supersede", TARGET))


async def test_supersede_writes_new_marks_old_and_drops_it_from_search(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), SUPERSEDE)
    r = await v.propose(CANDIDATE)
    assert (r.action, r.path, r.superseded, r.fell_back) == ("supersede", SUCCESSOR, TARGET, False)

    old = read(v.root, TARGET)
    assert f'superseded_by: "{SUCCESSOR}"' in old
    assert "id: 01234 # legacy account number, must survive a gate write" in old
    assert "The Acme renewal closes in March. See [[support-rota]]." in old
    # the forward link is path-qualified: the gate knows the exact path, so the
    # link cannot go ambiguous later behind a note that shares the stem
    assert "[[notes/acme-renewal-2026]]" in old
    assert "March 2026" in read(v.root, SUCCESSOR)

    # the supersede chain is excluded from the default search
    hits = [h.path for h in await v.search("acme renewal march")]
    assert SUCCESSOR in hits
    assert TARGET not in hits
    # and the forward wikilink resolves: no broken edge left behind
    assert (await v.doctor()).broken_links == []

    # ...including once another namespace holds a note of the same stem, which
    # is exactly what a bare `[[acme-renewal-2026]]` would not survive
    write_note(v.root, "archive/acme-renewal-2026.md", "# Archived copy\n")
    after = await v.doctor()
    assert after.broken_links == []
    assert after.ambiguous_links == []
    v.close()


async def test_supersede_really_retires_a_note_whose_fenced_frontmatter_is_unparseable(
    make_vault: MakeVault,
) -> None:
    # The block looks like frontmatter but parse_note cannot use it, so a key
    # patched *inside* it would be a key nothing reads: the gate would report
    # success while both notes stayed live in search.
    broken = "---\ntype: [unclosed\n---\n# Acme renewal\n\nAcme renewal closes in March.\n"
    v = await open_gate(make_vault({TARGET: broken}), SUPERSEDE)
    r = await v.propose(CANDIDATE)
    assert r.superseded == TARGET

    old = read(v.root, TARGET)
    assert old.startswith(f'---\nsuperseded_by: "{SUCCESSOR}"\n---\n')
    assert broken in old  # nothing of the original lost
    hits = [h.path for h in await v.search("acme renewal march")]
    assert SUCCESSOR in hits
    assert TARGET not in hits  # actually retired
    v.close()


async def test_supersede_keeps_the_successor_and_reports_the_old_note_unmarked_if_it_moves(
    make_vault: MakeVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The window the hash check at the top of apply cannot cover: the human
    # saves *after* the successor is written but before the old note is patched.
    root = make_vault(VAULT)

    def write(abs_path: Path, text: str) -> None:
        write_atomic(abs_path, text)
        if "acme-renewal-2026" in str(abs_path):
            monkeypatch.undo()  # the successor has landed: now a human saves the old note
            write_note(root, TARGET, "# Acme renewal\n\nhand edit\n")

    monkeypatch.setattr("wilcus_vault.gate_write.write_atomic", write)
    v = await open_gate(root, SUPERSEDE)
    r = await v.propose(CANDIDATE)
    assert (r.action, r.path, r.unmarked) == ("supersede", SUCCESSOR, TARGET)
    assert r.superseded is None
    assert read(root, TARGET) == "# Acme renewal\n\nhand edit\n"  # untouched
    assert "March 2026" in read(root, SUCCESSOR)  # successor stands
    v.close()


async def test_supersede_patches_a_note_with_no_usable_frontmatter_without_eating_its_body(
    make_vault: MakeVault,
) -> None:
    broken = "---\ntype: customer\n# no closing fence\n\nAcme renewal notes, unterminated.\n"
    v = await open_gate(make_vault({TARGET: broken}), SUPERSEDE)
    await v.propose(CANDIDATE)

    old = read(v.root, TARGET)
    assert old.startswith(f'---\nsuperseded_by: "{SUCCESSOR}"\n---\n')
    assert broken in old  # every original byte still there
    hits = [h.path for h in await v.search("acme renewal unterminated")]
    assert TARGET not in hits
    v.close()


async def test_supersede_aborts_before_writing_anything_when_the_old_note_moved_under_it(
    make_vault: MakeVault,
) -> None:
    root = make_vault(VAULT)
    calls = 0

    async def decider(_input: DeciderInput) -> Decision:
        nonlocal calls
        calls += 1
        write_note(root, TARGET, f"# Acme renewal\n\nhand edit {calls}\n")
        return Decision("supersede", TARGET)

    v = await open_gate(root, decider)
    r = await v.propose(CANDIDATE)
    assert r.fell_back is True
    assert "superseded_by" not in read(root, TARGET)
    # No orphan successor: had either aborted attempt written one, the fallback
    # create would have had to suffix around it.
    assert r.path == SUCCESSOR
    assert not os.path.exists(root / "notes/acme-renewal-2026-2.md")
    v.close()

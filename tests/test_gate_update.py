"""Write-gate evals: update, and the check-and-write rail around it."""

import os
import re
from pathlib import Path

import pytest
from conftest import MakeVault, write_note
from fakes import fixed_decider
from gate_common import CANDIDATE, OLD_NOTE, VAULT, open_gate, read

from wilcus_vault.decision import DeciderInput, Decision
from wilcus_vault.term import VaultError

TARGET = "notes/acme-renewal.md"


async def test_update_rewrites_the_body_and_bumps_updated_frontmatter_otherwise_identical(
    make_vault: MakeVault,
) -> None:
    body = "# Acme renewal\n\nRenewal closes 2026-03-01 at the agreed pricing.\n"
    v = await open_gate(make_vault(VAULT), fixed_decider(Decision("update", TARGET, body)))
    r = await v.propose(CANDIDATE)
    assert (r.action, r.path, r.fell_back) == ("update", TARGET, False)

    raw = read(v.root, TARGET)
    # A YAML round-trip would drop the comment and turn 01234 into 1234 and 1.0
    # into 1: the gate never re-serializes a note it did not author.
    assert "id: 01234 # legacy account number, must survive a gate write" in raw
    assert "rate: 1.0" in raw
    assert "type: customer" in raw
    assert "updated: 2020-01-01" not in raw
    assert re.search(r"^updated: 20\d\d-\d\d-\d\dT[\d:.]+Z$", raw, re.M)  # YAML-native, unquoted
    assert raw.endswith("Renewal closes 2026-03-01 at the agreed pricing.\n")
    assert "closes in March." not in raw
    # written through a temp file and renamed into place, leaving no debris
    assert [f for f in os.listdir(Path(v.root) / "notes") if ".tmp-" in f] == []
    v.close()


async def test_a_human_edit_between_search_and_apply_aborts_and_regates(
    make_vault: MakeVault,
) -> None:
    root = make_vault(VAULT)
    calls = 0

    async def decider(_input: DeciderInput) -> Decision:
        # the mid-flight edit: a human saves the target while the decider thinks
        nonlocal calls
        calls += 1
        if calls == 1:
            write_note(root, TARGET, "# Acme renewal\n\nhand edit\n")
        return Decision("update", TARGET, "# Acme renewal\n\ngate body\n")

    v = await open_gate(root, decider)
    r = await v.propose(CANDIDATE)
    assert calls == 2  # aborted, re-ran the whole gate once
    assert (r.action, r.path, r.fell_back) == ("update", TARGET, False)
    assert read(root, TARGET).endswith("gate body\n")
    v.close()


async def test_a_second_mid_flight_edit_falls_back_to_create_and_clobbers_nothing(
    make_vault: MakeVault,
) -> None:
    root = make_vault(VAULT)
    calls = 0

    async def decider(_input: DeciderInput) -> Decision:
        nonlocal calls
        calls += 1
        write_note(root, TARGET, f"# Acme renewal\n\nhand edit {calls}\n")
        return Decision("update", TARGET, "gate body\n")

    v = await open_gate(root, decider)
    r = await v.propose(CANDIDATE)
    assert calls == 2
    assert (r.action, r.path, r.fell_back) == ("create", "notes/acme-renewal-2026.md", True)
    # the human's edit survived intact: nothing was clobbered silently
    assert read(root, TARGET) == "# Acme renewal\n\nhand edit 2\n"
    assert "March 2026" in read(root, "notes/acme-renewal-2026.md")
    v.close()


@pytest.mark.parametrize("body", ["", "   \n\t"])
async def test_a_body_the_decider_left_empty_is_malformed_not_an_instruction_to_blank(
    make_vault: MakeVault, body: str
) -> None:
    v = await open_gate(make_vault(VAULT), fixed_decider(Decision("update", TARGET, body)))
    with pytest.raises(VaultError, match="write gate"):
        await v.propose(CANDIDATE)
    assert read(v.root, TARGET) == OLD_NOTE  # nothing emptied
    v.close()


async def test_the_decider_cannot_name_a_path_the_search_did_not_return(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(
        make_vault(VAULT), fixed_decider(Decision("update", "../../../etc/passwd", "pwned"))
    )
    with pytest.raises(VaultError, match="not among the similar notes"):
        await v.propose(CANDIDATE)
    v.close()

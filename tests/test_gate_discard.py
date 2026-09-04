"""Write-gate evals: discard keeps the whole candidate in <root>/.discarded.log."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import MakeVault
from fakes import fixed_decider
from gate_common import CANDIDATE, NOTHING_SIMILAR, OLD_NOTE, VAULT, open_gate, read

from wilcus_vault.decision import Decision
from wilcus_vault.doctor import DoctorOptions

DISCARD = fixed_decider(Decision("discard"))


def log_lines(root: Path) -> list[str]:
    return read(root, ".discarded.log").strip().split("\n")


async def test_discard_appends_the_whole_candidate_to_the_discard_log(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), DISCARD)
    r = await v.propose(CANDIDATE)
    assert (r.action, r.fell_back, r.path) == ("discard", False, None)

    lines = log_lines(v.root)
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["candidate"] == CANDIDATE.to_json()  # recoverable in full
    assert entry["at"].startswith("20")

    await v.propose(replace(CANDIDATE, title="Second thought"))
    assert len(log_lines(v.root)) == 2  # appended

    # durable history does not live in the disposable index directory: a
    # `.vault/` nuke or a --rebuild must not take it with them
    assert not (v.root / ".vault" / "discarded.log").exists()
    await v.doctor(DoctorOptions(rebuild=True))
    assert len(log_lines(v.root)) == 2

    # the log sits at a fixed path in the user-visible tree and carries whole
    # candidate bodies: a symlink left in its place is refused, not followed
    elsewhere = make_vault({})
    os.remove(v.root / ".discarded.log")
    os.symlink(elsewhere / "stolen.log", v.root / ".discarded.log")
    with pytest.raises(OSError):
        await v.propose(replace(CANDIDATE, title="Third thought"))
    assert not (elsewhere / "stolen.log").exists()
    v.close()


async def test_a_discard_line_records_the_similar_set_the_decider_saw(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), DISCARD)
    await v.propose(CANDIDATE)

    # The justification rides with the entry: which notes the decider judged the
    # candidate against, at which hash and score. Staleness detection consumes
    # this, and it cannot be retrofitted onto lines logged without it.
    entry = json.loads(read(v.root, ".discarded.log").strip())
    assert entry["decision"] == {"action": "discard"}
    hit = next(s for s in entry["similar"] if s["path"] == "notes/acme-renewal.md")
    assert hit["hash"] == hashlib.sha256(OLD_NOTE.encode()).hexdigest()
    assert hit["score"] > 0
    v.close()


async def test_a_discard_with_nothing_similar_carries_an_empty_array(
    make_vault: MakeVault,
) -> None:
    v = await open_gate(make_vault(VAULT), DISCARD, NOTHING_SIMILAR)
    await v.propose(
        replace(
            CANDIDATE,
            title="Zeppelin fuselage torque",
            type=None,
            body="Torque values for the zeppelin fuselage struts.\n",
        )
    )
    entry = json.loads(read(v.root, ".discarded.log").strip())
    assert entry["similar"] == []
    v.close()

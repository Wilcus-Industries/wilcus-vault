"""vault promote on the command line: the proposals/ layout, the gate's line, and
what became of the proposal."""

import json

import pytest
from conftest import MakeVault
from test_cli import cli
from test_cli_scoped import decide
from test_promote import FILES, POLICY, PROPOSAL

from wilcus_vault.cli.usage import gate_line
from wilcus_vault.db import db_path
from wilcus_vault.promote import PromoteResult


async def test_promote_moves_a_proposal_into_shared_and_says_what_became_of_it(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(FILES)
    (root / ".vault-policy.json").write_text(json.dumps(POLICY))
    transport = decide(monkeypatch, {"action": "create"})
    gated = ("--ceiling", "0.9", "--lexical", "--vault", root)

    # a peer may not write shared/: refused before any model spend
    denied = await cli(capsys, "promote", PROPOSAL, "--agent", "peer", *gated)
    assert (denied.code, denied.out) == (1, "")
    assert '"peer" may not write to shared/' in denied.err
    assert transport.calls == []

    orchestrator = ("--agent", "orchestrator", *gated)
    r = await cli(capsys, "promote", f"./{PROPOSAL}", *orchestrator)
    assert (r.code, r.out) == (0, "create  shared/acme-renewal-2026.md\nproposal removed")
    assert r.err.startswith("indexed ")  # the summary stays off stdout
    assert not (root / PROPOSAL).exists()
    assert f"vault_source: {PROPOSAL}" in (root / "shared/acme-renewal-2026.md").read_text()
    shown = await cli(capsys, "discards", "show", "1", *orchestrator)
    assert json.loads(shown.out)["path"] == "shared/acme-renewal-2026.md"  # where it landed

    missing = await cli(capsys, "promote", PROPOSAL, *orchestrator)
    assert (missing.code, missing.out) == (1, "")
    assert missing.err.endswith(f"promote: no note at {PROPOSAL}")


async def test_promote_refuses_a_path_outside_proposals_before_anything_runs(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(FILES)
    transport = decide(monkeypatch, {"action": "create"})
    gated = ("--ceiling", "0.9", "--lexical", "--vault", root)
    outside = (
        "shared/acme-renewal.md",
        # starts with proposals/, and names a shared note
        "proposals/../shared/acme-renewal.md",
        "proposals-old/a.md",
        "../proposals/a.md",
    )
    for path in outside:
        r = await cli(capsys, "promote", path, *gated)
        assert (r.code, r.out, r.err) == (1, "", f"promote: {path} is not under proposals/")
    assert (await cli(capsys, "promote", *gated)).code == 1
    assert (await cli(capsys, "promote", PROPOSAL, PROPOSAL, *gated)).code == 1
    assert transport.calls == []
    assert not db_path(root).exists()  # refused before the reindex, too
    assert (root / "shared/acme-renewal.md").exists()


def test_a_kept_proposal_is_reported_on_the_line_after_the_gates() -> None:
    kept = PromoteResult("update", "shared/acme-renewal.md", removed=False)
    assert gate_line(kept) == (
        "update  shared/acme-renewal.md\nproposal kept: it changed during promote"
    )

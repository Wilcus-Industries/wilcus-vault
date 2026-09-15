"""The discards subcommands, driven in-process through `main`."""

import json

import pytest
from conftest import MakeVault
from test_cli import GRAPH, cli
from test_cli_scoped import decide, with_policy
from test_scope_policy import POLICY

from wilcus_vault.scope import ScopeRule


def entry(title: str, namespace: str) -> dict[str, object]:
    candidate = {"title": title, "namespace": namespace, "body": f"{title}, refused.\n"}
    return {"at": "2026-01-01T00:00:00.000Z", "candidate": candidate, "similar": []}


async def test_discards_list_show_and_gated_restore(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault(GRAPH)
    lex = ("--lexical", "--vault", root)
    # an empty log is an answer, not an error
    r = await cli(capsys, "discards", "list", *lex)
    assert (r.code, r.out) == (0, "no discards")

    entry = {
        "at": "2026-01-01T00:00:00.000Z",
        "candidate": {"title": "Dropped thought", "namespace": "notes", "body": "body\n"},
        "decision": {"action": "discard"},
        "similar": [],
    }
    (root / ".discarded.log").write_text(json.dumps(entry) + "\n")

    listed = await cli(capsys, "discards", "list", *lex)
    assert listed.code == 0
    assert "Dropped thought" in listed.out
    assert "discard" in listed.out

    show = await cli(capsys, "discards", "show", "1", *lex)
    assert show.code == 0
    assert json.loads(show.out)["candidate"]["title"] == "Dropped thought"
    assert (await cli(capsys, "discards", "show", "9", *lex)).code == 1

    # restore needs a ceiling (like consolidate) and a configured chat model:
    # refused with the fix named, before any database is opened
    assert (await cli(capsys, "discards", "restore", "1", *lex)).code == 1
    no_model = await cli(capsys, "discards", "restore", "1", "--ceiling", "0.5", *lex)
    assert no_model.code == 1
    assert "VAULT_DECIDE_MODEL" in no_model.err

    # a missing or malformed subcommand is usage, not a crash
    assert (await cli(capsys, "discards", *lex)).code == 1
    assert (await cli(capsys, "discards", "show", "x", *lex)).code == 1

    # doctor surfaces the log so normal maintenance sees it
    doctored = await cli(capsys, "doctor", *lex)
    assert "discard log: 1 entries (0 recent)" in doctored.out


async def test_under_a_policy_discards_needs_an_agent_that_reads_the_whole_vault(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # reads the root, but not secret/: not the whole vault
    nosecret: list[ScopeRule] = [
        {"prefix": "", "read": True, "write": True},
        {"prefix": "secret/", "read": False, "write": False},
    ]
    root = with_policy(make_vault, {**POLICY, "core/nosecret": nosecret})
    log = [entry("Ledger line", "ledger"), entry("Acme renewal 2026", "notes")]  # newest last
    (root / ".discarded.log").write_text("".join(json.dumps(e) + "\n" for e in log))
    lex = ("--lexical", "--vault", root)

    # The log holds candidate bodies from every namespace, so a narrower reader sees none of it.
    for sub in (["list"], ["show", "1"], ["restore", "1", "--ceiling", "1"]):
        anon = await cli(capsys, "discards", *sub, *lex)
        assert (anon.code, anon.out) == (1, "")
        assert "needs a VaultContext" in anon.err
        for agent in ("core/notes", "core/nosecret"):
            r = await cli(capsys, "discards", *sub, "--agent", agent, *lex)
            assert (r.code, r.out) == (1, "")
            assert f'"{agent}" may not read the whole vault' in r.err
    scheduler = ("--agent", "core/scheduler", *lex)
    assert "Ledger line" in (await cli(capsys, "discards", "list", *scheduler)).out

    # restore writes through the gate as that agent: write-checked first, then stamped
    transport = decide(monkeypatch, {"action": "create"})
    denied = await cli(capsys, "discards", "restore", "2", "--ceiling", "1", *scheduler)
    assert denied.code == 1
    assert '"core/scheduler" may not write to ledger/' in denied.err
    assert transport.calls == []
    r = await cli(capsys, "discards", "restore", "1", "--ceiling", "1", *scheduler)
    assert (r.code, r.out) == (0, "create  notes/acme-renewal-2026.md")
    assert "vault_agent: core/scheduler" in (root / "notes/acme-renewal-2026.md").read_text()

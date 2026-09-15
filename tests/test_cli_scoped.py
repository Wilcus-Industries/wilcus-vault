"""propose, get, list and search on the command line, with and without .vault-policy.json."""

import io
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import MakeVault
from fakes import StubTransport, chat_reply, stub_transport
from test_cli import Run, cli
from test_scope_policy import POLICY
from test_scope_vault import VAULT

from wilcus_vault.decide import fetch_decider
from wilcus_vault.scope import ScopePolicy

NOTE = (
    "---\ntype: customer\n---\n# Acme renewal 2026\n\n"
    "The Acme renewal closes in March 2026 at the agreed renewal pricing.\n"
)
SECRET = VAULT["secret/plans.md"].rstrip("\n")  # as `cli` captures it, trailing newline dropped


def with_policy(make_vault: MakeVault, policy: ScopePolicy = POLICY) -> Path:
    root = make_vault(VAULT)
    (root / ".vault-policy.json").write_text(json.dumps(policy))
    return root


def decide(monkeypatch: pytest.MonkeyPatch, answer: dict[str, str]) -> StubTransport:
    """The CLI's own chat decider over a stub transport: a fixed reply, no model, no network."""
    transport = stub_transport(0, chat_reply(json.dumps(answer)))
    monkeypatch.setattr(
        "wilcus_vault.cli.scoped.fetch_decider",
        lambda: fetch_decider(model="stub", transport=transport),
    )
    return transport


async def propose(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    note: str,
    root: Path,
    *argv: str,
) -> Run:
    monkeypatch.setattr("sys.stdin", io.StringIO(note))
    return await cli(capsys, "propose", *argv, "--lexical", "--vault", root)


async def test_without_a_policy_every_agent_may_do_anything(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(VAULT)
    lex = ("--lexical", "--vault", root)
    # --agent is optional until a policy is in force, and grants nothing special
    for agent in ((), ("--agent", "anyone")):
        listed = await cli(capsys, "list", *agent, *lex)  # first: it reindexes for search
        assert listed.code == 0
        assert "secret/plans.md" in listed.out.split("\n")
        assert (await cli(capsys, "get", "secret/plans.md", *agent, *lex)).out == SECRET
        found = await cli(capsys, "search", "acme", "renewal", "pricing", *agent, *lex)
        assert "secret/plans.md" in found.out

    decide(monkeypatch, {"action": "create"})
    r = await propose(capsys, monkeypatch, NOTE, root, "--namespace", "ledger", "--ceiling", "0.9")
    assert (r.code, r.out) == (0, "create  ledger/acme-renewal-2026.md")
    written = (root / "ledger/acme-renewal-2026.md").read_text()
    assert "title: Acme renewal 2026" in written
    assert "type: customer" in written
    assert "vault_agent" not in written  # no --agent, no provenance


async def test_list_reindexes_first_and_prints_one_path_per_line(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({"notes/a.md": "# A\n", "b.md": "# B\n"})
    lex = ("--lexical", "--vault", root)
    r = await cli(capsys, "list", *lex)
    assert (r.code, r.out) == (0, "b.md\nnotes/a.md")
    assert r.err == "indexed 2 new, 0 changed, 0 removed, 0 unchanged"
    # a note written by hand since the last pass is listed, not missed
    (root / "notes/c.md").write_text("# C\n")
    assert (await cli(capsys, "list", "notes", *lex)).out == "notes/a.md\nnotes/c.md"
    assert (await cli(capsys, "list", "nothing-here", *lex)).out == ""
    assert (await cli(capsys, "list", "notes", "b", *lex)).code == 1


async def test_a_policy_scopes_get_list_and_search_to_the_agent(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = with_policy(make_vault)
    lex = ("--lexical", "--vault", root)
    notes = ("--agent", "core/notes")
    listed = await cli(capsys, "list", *notes, *lex)
    assert (listed.code, listed.out.split("\n")) == (
        0,
        ["ledger/q3.md", "notes/acme-renewal.md", "notes/hub.md", "notes/support-rota.md"],
    )

    found = await cli(capsys, "search", "acme", "renewal", "pricing", *notes, *lex)
    assert found.code == 0
    assert "notes/acme-renewal.md" in found.out
    assert "secret/plans.md" not in found.out
    wide = await cli(capsys, "search", "acme renewal pricing", "--agent", "core/scheduler", *lex)
    assert "secret/plans.md" in wide.out

    # a note the agent may not read and a note that is not there are one answer
    denied = await cli(capsys, "get", "secret/plans.md", *notes, *lex)
    absent = await cli(capsys, "get", "secret/nothing.md", *notes, *lex)
    assert (denied.code, denied.out, denied.err) == (1, "", "no note at secret/plans.md")
    assert (absent.code, absent.out, absent.err) == (1, "", "no note at secret/nothing.md")
    allowed = await cli(capsys, "get", "secret/plans.md", "--agent", "core/scheduler", *lex)
    assert (allowed.code, allowed.out) == (0, SECRET)

    # fails closed: no agent, or one the policy never names, is refused
    for command in (["list"], ["get", "notes/hub.md"], ["search", "acme"]):
        anon = await cli(capsys, *command, *lex)
        assert (anon.code, anon.out) == (1, "")
        assert "needs a VaultContext" in anon.err
        typo = await cli(capsys, *command, "--agent", "core/typo", *lex)
        assert (typo.code, typo.out) == (1, "")
        assert "has no scope" in typo.err


async def test_propose_under_a_policy_checks_and_stamps_the_agent(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = with_policy(make_vault)
    transport = decide(monkeypatch, {"action": "create"})
    notes = ("--agent", "core/notes", "--ceiling", "0.9")

    denied = await propose(capsys, monkeypatch, NOTE, root, *notes, "--namespace", "ledger")
    assert denied.code == 1
    assert 'write gate: "core/notes" may not write to ledger/' in denied.err
    assert transport.calls == []  # refused before any model spend
    anon = await propose(capsys, monkeypatch, NOTE, root, "--namespace", "notes", "--ceiling", "1")
    assert anon.code == 1
    assert "needs a VaultContext" in anon.err

    r = await propose(capsys, monkeypatch, NOTE, root, *notes, "--namespace", "notes")
    assert (r.code, r.out) == (0, "create  notes/acme-renewal-2026.md")
    assert r.err.startswith("indexed ")  # the summary stays off stdout
    assert "vault_agent: core/notes" in (root / "notes/acme-renewal-2026.md").read_text()
    # reindexed before the gate searched, so the decider saw the notes already on disk
    assert "notes/acme-renewal.md" in str(transport.calls[0].body)

    # a decision the agent may not write lands as a create, and the line says so
    update = {"action": "update", "target": "ledger/q3.md", "body": "Rewritten.\n"}
    decide(monkeypatch, update)
    fell = await propose(capsys, monkeypatch, NOTE, root, *notes, "--namespace", "notes")
    assert (fell.code, fell.out) == (0, "create  notes/acme-renewal-2026-2.md (fell back)")


async def test_propose_refuses_a_note_it_cannot_gate_before_the_decider_runs(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(VAULT)
    transport = decide(monkeypatch, {"action": "create"})
    no_ceiling = await propose(capsys, monkeypatch, NOTE, root)
    assert no_ceiling.code == 1
    assert "propose needs --ceiling" in no_ceiling.err

    untitled = await propose(capsys, monkeypatch, "a body, no heading\n", root, "--ceiling", "1")
    assert untitled.code == 1
    assert "no title" in untitled.err
    # the broken block would otherwise be written into the new note's body
    broken = "---\ntitle: [unclosed\n---\n# Heading\n\nbody\n"
    malformed = await propose(capsys, monkeypatch, broken, root, "--ceiling", "1")
    assert malformed.code == 1
    assert "frontmatter" in malformed.err
    # the note comes on stdin; a path argument would wait on a terminal forever
    assert (await propose(capsys, monkeypatch, NOTE, root, "note.md", "--ceiling", "1")).code == 1
    assert transport.calls == []
    assert list(root.glob("**/acme-renewal-2026*")) == []


MALFORMED: dict[str, Callable[[Path], object]] = {
    "not json": lambda p: p.write_text("{"),
    "null, not an object": lambda p: p.write_text("null"),
    "a rule open refuses": lambda p: p.write_text('{"core/notes": [{"prefix": "", "read": "no"}]}'),
    # a typo that would otherwise grant: `wirte` is ignored, so ledger/ stays writable
    "a misspelt permission": lambda p: p.write_text(
        '{"core/notes": [{"prefix": "", "read": true, "write": true},'
        ' {"prefix": "ledger", "wirte": false}]}'
    ),
    # JSON keeps a repeated key's last value, so this rule would read as a grant
    "a key given twice": lambda p: p.write_text(
        '{"core/notes": [{"prefix": "", "read": false, "read": true}]}'
    ),
    "a dangling symlink": lambda p: p.symlink_to(p.parent / "moved-away.json"),
    "a directory": lambda p: p.mkdir(),
}


@pytest.mark.parametrize("make_policy", MALFORMED.values(), ids=list(MALFORMED))
async def test_a_policy_file_that_cannot_be_used_exits_1_never_allow_all(
    make_policy: Callable[[Path], object],
    make_vault: MakeVault,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_vault(VAULT)
    make_policy(root / ".vault-policy.json")
    lex = ("--lexical", "--vault", root)
    for command in (["search", "acme"], ["list"], ["get", "secret/plans.md"]):
        r = await cli(capsys, *command, "--agent", "core/notes", *lex)
        assert (r.code, r.out) == (1, "")
        assert r.err.startswith(("vault: cannot load .vault-policy.json", "vault: scope policy"))
    # maintenance is unscoped, so it never reads the policy
    assert (await cli(capsys, "reindex", *lex)).code == 0


async def test_a_directory_inside_a_scoped_vault_is_refused_never_allow_all(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = with_policy(make_vault)
    here = root.resolve()
    # Below the root the policy file is out of sight, and every agent would be allow-all.
    below = ("--agent", "core/notes", "--lexical", "--vault", root / "secret")
    commands = [["get", "plans.md"], ["list"], ["search", "secret", "plans"], ["discards", "list"]]
    for command in commands:
        r = await cli(capsys, *command, *below)
        assert (r.code, r.out) == (1, "")
        assert f"{here / 'secret'} is inside the scoped vault {here}; use --vault {here}" in r.err

    # a read-only peer would otherwise write into a namespace the policy keeps from it
    transport = decide(monkeypatch, {"action": "create"})
    ledger = root / "ledger"
    r = await propose(capsys, monkeypatch, NOTE, ledger, "--agent", "core/notes", "--ceiling", "1")
    assert r.code == 1
    assert f"use --vault {here}" in r.err
    assert transport.calls == []
    assert sorted(p.name for p in (root / "ledger").iterdir()) == ["q3.md"]

    # and a symlink into the vault is the same directory by another name
    link = root.parent / f"{root.name}-secret"
    link.symlink_to(root / "secret")
    r = await cli(capsys, "get", "plans.md", "--agent", "core/notes", "--lexical", "--vault", link)
    assert (r.code, r.out) == (1, "")
    assert f"use --vault {here}" in r.err

    # `attach/..` spells the vault, but through the link it is the target's parent: the
    # policy and the notes must both come from one of those directories, never one each
    outside = root.parent / f"{root.name}-outside" / "target"
    outside.mkdir(parents=True)
    (root / "attach").symlink_to(outside)
    dotdot = ("--agent", "core/notes", "--lexical", "--vault", root / "attach" / "..")
    r = await cli(capsys, "get", "secret/plans.md", *dotdot)
    assert (r.code, r.out, r.err) == (1, "", "no note at secret/plans.md")


async def test_get_prints_the_file_with_only_line_endings_and_tabs_left_raw(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({"notes/evil.md": "# Evil\r\n\tindented\x1b[2K\rover\x07written\n"})
    lex = ("--lexical", "--vault", root)
    r = await cli(capsys, "get", "notes/evil.md", *lex)
    assert (r.code, r.err) == (0, "")
    assert r.out == "# Evil\r\n\tindented?[2K?over?written"
    assert (await cli(capsys, "get", *lex)).code == 1

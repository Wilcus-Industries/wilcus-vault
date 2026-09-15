"""vault init --layout swarm: a roster becomes tiered namespaces and the scope policy over them."""

import json
from pathlib import Path

import pytest
from conftest import MakeVault, write_note
from test_cli import Run, cli
from test_cli_scoped import NOTE, decide, propose

ROSTER = {
    "lead": {"kind": "orchestrator"},
    # verbatim: a name is what --agent passes, so it is never slugified or case-folded
    "Planner Two": {"kind": "manager", "model": "other keys are ignored"},
    "coder": {"kind": "doer"},
    "tester": {"kind": "doer", "manager": "Planner Two"},
    "grunt": {"kind": "worker", "manager": "Planner Two"},
}


def owns(role: str) -> list[dict[str, object]]:
    return [
        {"prefix": "shared/", "read": True},
        {"prefix": f"roles/{role}/", "read": True, "write": True},
        {"prefix": f"proposals/{role}/", "read": True, "write": True},
    ]


POLICY = {
    "lead": [{"prefix": "", "read": True, "write": True}],
    "Planner Two": owns("Planner Two"),
    "coder": owns("coder"),
    "tester": owns("tester"),
    "grunt": [{"prefix": "shared/", "read": True}, {"prefix": "roles/Planner Two/", "read": True}],
}


def roster_file(root: Path, roster: object) -> Path:
    """Beside the vault, not in it, so a refused init can be seen to leave the vault empty."""
    path = root.parent / f"{root.name}-roster.json"
    path.write_text(roster if isinstance(roster, str) else json.dumps(roster))
    return path


async def init(capsys: pytest.CaptureFixture[str], root: Path, roster: object) -> Run:
    return await cli(
        capsys, "init", "--layout", "swarm", "--roster", roster_file(root, roster), "--vault", root
    )


async def test_init_lays_out_every_kind_and_writes_its_policy(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({})
    r = await init(capsys, root, ROSTER)
    assert (r.code, r.err) == (0, "")
    assert r.out.split("\n") == [
        "created shared/",
        "created roles/Planner Two/",
        "created proposals/Planner Two/",
        "created roles/coder/",
        "created proposals/coder/",
        "created roles/tester/",
        "created proposals/tester/",
        "wrote .vault-policy.json",
    ]
    # the orchestrator and the worker get a scope, never a directory
    dirs = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())
    assert dirs == [
        "proposals",
        "proposals/Planner Two",
        "proposals/coder",
        "proposals/tester",
        "roles",
        "roles/Planner Two",
        "roles/coder",
        "roles/tester",
        "shared",
    ]
    assert json.loads((root / ".vault-policy.json").read_text()) == POLICY


INVALID: dict[str, tuple[object, str]] = {
    "not JSON": ("{", "init: cannot load roster"),
    "not an object": ("[]", "init: the roster must be a JSON object"),
    # JSON keeps the last, so one of the two rows would vanish without a word
    "a role given twice": ('{"a": {"kind": "doer"}, "a": {"kind": "worker"}}', "duplicate key"),
    "an empty name": ({"": {"kind": "doer"}}, 'roster row "": a role name must be one path'),
    "a slash": ({"core/notes": {"kind": "doer"}}, 'row "core/notes": a role name must be one'),
    "a backslash": ({"core\\notes": {"kind": "doer"}}, "a role name must be one path segment"),
    "a parent segment": ({"..": {"kind": "doer"}}, 'row "..": a role name must be one path'),
    "a hidden name": ({".git": {"kind": "doer"}}, 'row ".git": a role name must be one path'),
    "a control character": ({"a\x1bb": {"kind": "doer"}}, 'row "a?b": a role name must be'),
    "a NUL": ({"a\x00b": {"kind": "doer"}}, 'row "a?b": a role name must be one path'),
    # the gate refuses every propose from a blank agent, so its scope would be dead weight
    "a blank name": ({"   ": {"kind": "doer"}}, 'row "   ": a role name must be one path'),
    # a JSON escape can spell half a surrogate pair, which no filename can hold
    "a lone surrogate": ({"bad\ud800": {"kind": "doer"}}, "a role name must be one path segment"),
    # refused even where no directory is made: the name is still what --agent passes
    "an orchestrator with a slash": ({"a/b": {"kind": "orchestrator"}}, "must be one path segment"),
    "a row that is not an object": ({"a": "doer"}, 'row "a": kind must be orchestrator, manager'),
    "an unknown kind": ({"a": {"kind": "boss"}}, 'row "a": kind must be orchestrator, manager'),
    "no kind": ({"a": {}}, 'row "a": kind must be orchestrator, manager, doer or worker'),
    "a worker with no manager": ({"w": {"kind": "worker"}}, 'row "w": a worker\'s manager must'),
    "a worker under a doer": (
        {"d": {"kind": "doer"}, "w": {"kind": "worker", "manager": "d"}},
        'row "w": a worker\'s manager must name a row whose kind is manager',
    ),
    "a worker under nobody": (
        {"w": {"kind": "worker", "manager": "ghost"}},
        'row "w": a worker\'s manager must name a row whose kind is manager',
    ),
    "a manager that is not a name": (
        {"w": {"kind": "worker", "manager": ["m"]}, "m": {"kind": "manager"}},
        'row "w": a worker\'s manager must name a row whose kind is manager',
    ),
}


@pytest.mark.parametrize(("roster", "why"), INVALID.values(), ids=list(INVALID))
async def test_an_invalid_roster_is_refused_and_nothing_is_written(
    roster: object, why: str, make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({})
    r = await init(capsys, root, roster)
    assert (r.code, r.out) == (1, "")
    assert why in r.err
    assert list(root.iterdir()) == []


REFUSED_ARGS: dict[str, tuple[list[str], str]] = {  # ROSTER stands for a valid roster file
    "no layout": (["--roster", "ROSTER"], "init needs --layout swarm"),
    "another layout": (["--layout", "flat", "--roster", "ROSTER"], "init needs --layout swarm"),
    "no roster": (["--layout", "swarm"], "init needs --roster <file>"),
    "a roster flag with no file": (["--layout", "swarm", "--roster"], "--roster needs a value"),
    "a roster that is not there": (["--layout", "swarm", "--roster", "MISSING"], "cannot load"),
    "a word": (["extra", "--layout", "swarm", "--roster", "ROSTER"], "init takes no arguments"),
}


@pytest.mark.parametrize(("argv", "why"), REFUSED_ARGS.values(), ids=list(REFUSED_ARGS))
async def test_init_needs_the_swarm_layout_and_a_roster(
    argv: list[str], why: str, make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({})
    files = {"ROSTER": str(roster_file(root, ROSTER)), "MISSING": str(root.parent / "gone.json")}
    r = await cli(capsys, "init", "--vault", root, *(files.get(a, a) for a in argv))
    assert (r.code, r.out) == (1, "")
    assert why in r.err
    assert list(root.iterdir()) == []


async def test_reinit_keeps_what_is_there_and_replaces_the_policy(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({})
    assert (await init(capsys, root, ROSTER)).code == 0
    write_note(root, "roles/coder/habits.md", "# Habits\n")
    policy = (root / ".vault-policy.json").read_text()

    again = await init(capsys, root, ROSTER)
    assert (again.code, again.out) == (0, "wrote .vault-policy.json")
    assert (root / ".vault-policy.json").read_text() == policy
    assert (root / "roles/coder/habits.md").read_text() == "# Habits\n"

    # the roster is the source: a row it drops leaves the policy, never merged back in
    changed = {k: v for k, v in ROSTER.items() if k != "tester"} | {"reviewer": {"kind": "doer"}}
    r = await init(capsys, root, changed)
    assert r.out.split("\n") == [
        "created roles/reviewer/",
        "created proposals/reviewer/",
        "wrote .vault-policy.json",
    ]
    written = json.loads((root / ".vault-policy.json").read_text())
    assert sorted(written) == sorted(changed)
    assert (root / "roles/tester").is_dir()  # a dropped role's notes are kept


async def test_init_refuses_a_symlinked_namespace_before_writing_anything(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({".vault-policy.json": "{}"})  # a re-init, so the directory may hold things
    outside = root.parent / f"{root.name}-elsewhere"
    outside.mkdir()
    (root / "roles").symlink_to(outside)
    r = await init(capsys, root, ROSTER)
    assert (r.code, r.out) == (1, "")
    assert "roles/Planner Two passes through a symlink" in r.err
    assert list(outside.iterdir()) == []
    assert sorted(p.name for p in root.iterdir()) == [".vault-policy.json", "roles"]  # no shared/
    assert (root / ".vault-policy.json").read_text() == "{}"  # and no new policy


async def test_init_refuses_a_directory_holding_anything_but_its_own_policy(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = make_vault({})
    swarm = parent / ".swarm"  # hidden, as a swarm's memory may be: dot-entries count too
    roster = roster_file(parent, ROSTER)  # beside parent, so parent holds only the swarm
    lay_out = ("init", "--layout", "swarm", "--roster", roster, "--vault")
    assert (await cli(capsys, *lay_out, swarm)).code == 0  # a directory not there yet is made

    # --vault defaults to the working directory: a policy in a project root above a live
    # swarm would put the swarm inside another scoped vault, and lock every command out
    r = await cli(capsys, *lay_out, parent)
    assert (r.code, r.out) == (1, "")
    assert f"init: {parent.resolve()} is not empty and has no .vault-policy.json" in r.err
    assert [p.name for p in parent.iterdir()] == [".swarm"]
    assert (await cli(capsys, "list", "--agent", "lead", "--lexical", "--vault", swarm)).code == 0

    # nor is an unscoped vault converted: every agent the roster leaves out would lose it
    plain = make_vault({"notes/a.md": "# A\n"})
    r = await init(capsys, plain, ROSTER)
    assert (r.code, r.out) == (1, "")
    assert "is not empty and has no .vault-policy.json" in r.err
    assert [p.name for p in plain.iterdir()] == ["notes"]


async def test_init_inside_a_scoped_vault_is_refused(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({})
    assert (await init(capsys, root, ROSTER)).code == 0
    # empty, but a policy there would sit inside the swarm's, out of sight of its commands
    r = await init(capsys, root / "shared", ROSTER)
    here = root.resolve()
    assert (r.code, r.out) == (1, "")
    # never "use --vault <root>": re-running init there would replace the swarm's policy
    inside = f"init: {here / 'shared'} is inside the scoped vault {here}"
    assert r.err == f"{inside}; init a directory outside it"
    assert list((root / "shared").iterdir()) == []


async def test_a_vault_opened_with_the_written_policy_enforces_it(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault({})
    assert (await init(capsys, root, ROSTER)).code == 0
    write_note(root, "roles/coder/habits.md", "# Habits\n\ncoder only\n")
    write_note(root, "roles/Planner Two/plan.md", "# Plan\n")
    lex = ("--lexical", "--vault", root)

    # a peer cannot read another peer's roles/: the same answer as no note at all
    own = await cli(capsys, "get", "roles/coder/habits.md", "--agent", "coder", *lex)
    assert (own.code, own.out) == (0, "# Habits\n\ncoder only")
    peer = await cli(capsys, "get", "roles/coder/habits.md", "--agent", "tester", *lex)
    assert (peer.code, peer.err) == (1, "no note at roles/coder/habits.md")
    assert (await cli(capsys, "get", "roles/coder/habits.md", "--agent", "lead", *lex)).code == 0

    # a worker reads its manager's roles/, and writes nowhere: refused before the decider runs
    plan = await cli(capsys, "get", "roles/Planner Two/plan.md", "--agent", "grunt", *lex)
    assert (plan.code, plan.out) == (0, "# Plan")
    transport = decide(monkeypatch, {"action": "create"})
    grunt = ("--agent", "grunt", "--ceiling", "1")
    for ns in ("roles/Planner Two", "shared", "proposals/Planner Two"):
        r = await propose(capsys, monkeypatch, NOTE, root, *grunt, "--namespace", ns)
        assert r.code == 1
        assert f'write gate: "grunt" may not write to {ns}/' in r.err
    assert transport.calls == []

    doer = ("--agent", "coder", "--ceiling", "1", "--namespace", "proposals/coder")
    r = await propose(capsys, monkeypatch, NOTE, root, *doer)
    assert (r.code, r.out) == (0, "create  proposals/coder/acme-renewal-2026.md")

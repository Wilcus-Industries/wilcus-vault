"""The discards subcommands, driven in-process through `main`."""

import json

import pytest
from conftest import MakeVault
from test_cli import GRAPH, cli


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

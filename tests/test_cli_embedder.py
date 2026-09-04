"""Which embedder the CLI builds, and how its failures reach the terminal."""

import pytest
from conftest import MakeVault
from test_cli import GRAPH, NO_KEY, REMOTE, cli


@pytest.mark.parametrize("command", [["reindex"], ["doctor"], ["watch"], ["search", "acme"]])
async def test_default_embedder_is_the_real_one(
    command: list[str],
    make_vault: MakeVault,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_vault(GRAPH)
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", REMOTE)
    r = await cli(capsys, *command, "--vault", root)
    assert (r.code, r.err) == (1, NO_KEY)


async def test_lexical_is_the_offline_escape_hatch(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # same environment, --lexical: no provider is consulted, so the vault indexes
    root = make_vault(GRAPH)
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", REMOTE)
    r = await cli(capsys, "reindex", "--lexical", "--vault", root)
    assert (r.code, r.err) == (0, "")


async def test_unreachable_embedder_is_one_line_never_a_stack_trace(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(GRAPH)
    # Nothing listens on port 1, and it is loopback: a refusal, not a network
    # call. (That a refused *default* endpoint says "no embedder configured:
    # start Ollama ..." is the embedder's test; this is that the CLI prints
    # whatever the embedder said as one sentence and exits 1.)
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", "http://127.0.0.1:1/v1/embeddings")
    r = await cli(capsys, "reindex", "--vault", root)
    assert r.code == 1
    assert r.out == ""
    assert r.err != ""
    assert "Traceback" not in r.err  # a stack frame is not an error message


@pytest.mark.parametrize("arg", ["--\x1b[2K\rx", "\x1b[2K\rx"])
async def test_error_message_cannot_rewrite_the_terminal(
    arg: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Everything on stderr was written by someone else: an argument, a note, or
    # up to 200 bytes of whatever answered on :11434. A carriage return or an
    # ESC in any of them redraws the line the terminal has already printed, so
    # stderr gets the same scrub stdout has. Argv is the shortest untrusted
    # string that reaches the catch all of them land in.
    r = await cli(capsys, arg)
    assert r.code == 1
    # every control character is gone but the newlines the CLI itself writes
    controls = [c for c in r.err if ord(c) < 32 or ord(c) == 127]
    assert controls == [c for c in r.err if c == "\n"]
    assert "?[2K?x" in r.err
    assert "vault <command>" in r.err  # and a multi-line usage error stays readable


async def test_search_with_no_query_reports_a_broken_embedder_first(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(GRAPH)
    # The embedder is built before the query is checked, so a misconfigured one
    # is what a bare `vault search` complains about. Pinned because the order is
    # arbitrary: swapping it would silently change this message.
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", REMOTE)
    r = await cli(capsys, "search", "--vault", root)
    assert (r.code, r.err) == (1, NO_KEY)

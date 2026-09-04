"""The CLI's commands, exit codes and output, driven in-process through `main`."""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import MakeVault

from wilcus_vault.cli import main
from wilcus_vault.db import db_path

GRAPH = {
    "notes/acme.md": "---\ntype: customer\n---\n# Acme Corp\n\n"
    "renewal terms, see [[globex]], [[ghost]], [[dup]]\n",
    "notes/globex.md": "# Globex\n\nno outgoing links\n",
    "notes/lonely.md": "# Lonely\n\nnothing here\n",
    "one/dup.md": "# Dup one\n",
    "two/dup.md": "# Dup two\n",
}
NO_KEY = "FetchEmbedder: no API key — pass api_key or set VAULT_EMBED_API_KEY"
# A remote endpoint with no key that FetchEmbedder refuses to construct at all,
# so this reaches no network: it is offline proof of *which* embedder the CLI
# built, because TokenOverlapEmbedder never reads the environment.
REMOTE = "https://api.example.invalid/v1/embeddings"


@dataclass
class Run:
    code: int
    out: str
    err: str


async def cli(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> Run:
    """Run the CLI with both streams captured (trailing newline dropped, as console.log would)."""
    capsys.readouterr()
    code = await main([str(a) for a in argv])
    captured = capsys.readouterr()
    return Run(code, captured.out.rstrip("\n"), captured.err.rstrip("\n"))


async def test_reindex_doctor_and_doctor_rebuild(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault(GRAPH)
    r = await cli(capsys, "reindex", "--lexical", "--vault", root)
    assert (r.code, r.err) == (0, "")
    assert db_path(root).exists()

    # GRAPH has a broken link, an ambiguous link and a duplicate stem: doctor
    # repaired what it could, so the exit code has to say work is left
    doctored = await cli(capsys, "doctor", "--lexical", "--vault", root)
    assert doctored.code == 1
    assert "ghost" in doctored.out
    assert doctored.err == ""
    assert (await cli(capsys, "doctor", "--rebuild", "--lexical", "--vault", root)).code == 1


async def test_doctor_names_candidates_for_an_ambiguous_link(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault(GRAPH)
    r = await cli(capsys, "doctor", "--lexical", "--vault", root)
    assert r.code == 1
    assert "broken link:    notes/acme.md -> [[ghost]]" in r.out
    # the line is the fix: copy one candidate into the note as written
    assert "ambiguous link: notes/acme.md -> [[dup]] (one/dup, two/dup)" in r.out


async def test_control_characters_never_reach_the_terminal_raw(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    # a link target and a filename are note-controlled strings; a bare \r or ESC
    # in one would rewrite the line the CLI just printed
    root = make_vault({"notes/evil.md": "# Evil\n\n[[gh\rost\x1bX]]\n"})
    r = await cli(capsys, "doctor", "--lexical", "--vault", root)
    assert "gh?ost?X" in r.out
    # any control character other than the newlines the CLI itself writes
    assert not re.search(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]", r.out)


async def test_clean_vault_exits_0_and_usage_goes_to_stderr(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({"a.md": "# A\n\n[[b]]\n", "b.md": "# B\n\n[[a]]\n"})
    r = await cli(capsys, "doctor", "--lexical", "--vault", root)
    assert (r.code, r.err) == (0, "")

    assert (await cli(capsys)).code == 1
    assert (await cli(capsys, "nope")).code == 1
    assert "unknown command nope" in (await cli(capsys, "nope")).err
    # a flag swallowed as the vault path would index the wrong directory
    assert (await cli(capsys, "doctor", "--vault", "--rebuild")).code == 1
    assert (await cli(capsys, "doctor", "--vault")).code == 1
    assert "unknown flag --wat" in (await cli(capsys, "doctor", "--wat")).err
    assert "vault <command>" in (await cli(capsys)).err


@pytest.mark.parametrize(
    "argv", [["--help"], ["-h"], ["search", "--help"], ["watch", "-h"], ["doctor", "--help"]]
)
async def test_help_documents_every_command_and_exits_0(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    r = await cli(capsys, *argv)
    assert r.code == 0  # help was asked for; answering it is not a failure
    assert r.err == ""
    for word in (
        "reindex",
        "doctor",
        "search",
        "watch",
        "consolidate",
        "discards",
        "--vault",
        "--lexical",
        "--ceiling",
    ):
        assert word in r.out


async def test_consolidate_reports_clusters_and_writes_nothing(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    twins = "# Renewal\n\nthe acme renewal closes in march\n"
    root = make_vault(
        {
            "notes/one.md": twins,
            "notes/two.md": twins,
            "notes/far.md": "# Zeppelin\n\ntorque values for the fuselage struts\n",
        }
    )
    r = await cli(capsys, "consolidate", "--ceiling", "0.2", "--lexical", "--vault", root)
    assert (r.code, r.err) == (0, "")
    assert "notes/one.md notes/two.md" in r.out
    assert "notes/far.md" not in r.out
    # Report-only: merging needs an injected merger (an LLM), which the CLI has
    # no way to wire, so nothing on disk may have moved.
    assert sorted(p.name for p in (root / "notes").iterdir()) == ["far.md", "one.md", "two.md"]
    assert (root / "notes/one.md").read_text() == twins

    # the ceiling is mandatory here too, and it is a cosine distance
    lex = ("--lexical", "--vault", root)
    assert (await cli(capsys, "consolidate", *lex)).code == 1
    assert (await cli(capsys, "consolidate", "--ceiling", "9", *lex)).code == 1
    assert (await cli(capsys, "consolidate", "--ceiling", *lex)).code == 1


async def test_search_prints_one_line_per_hit(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault(GRAPH)
    await cli(capsys, "reindex", "--lexical", "--vault", root)
    r = await cli(capsys, "search", "renewal", "terms", "--lexical", "--vault", root)
    assert (r.code, r.err) == (0, "")
    assert re.search(r"^0\.\d{4}\s+notes/acme\.md\b.*Acme Corp$", r.out, re.MULTILINE)
    # every hit is on one line, scores first, so the ranking reads down the page
    assert len(r.out.split("\n")) == 5
    # a query with nothing to search on is an answer, not an error: the CLI sets
    # no relevance cutoffs (none is meaningful for a bag-of-tokens embedder), so
    # this is the empty case, both signals had nothing to ask
    empty = await cli(capsys, "search", "!!!", "--lexical", "--vault", root)
    assert (empty.code, empty.out) == (0, "no matches")
    # and a query is required
    assert (await cli(capsys, "search", "--lexical", "--vault", root)).code == 1


async def test_search_on_an_unindexed_vault_says_so(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault(GRAPH)
    r = await cli(capsys, "search", "acme", "--lexical", "--vault", root)
    assert (r.code, r.out) == (0, "vault is not indexed (run vault reindex)")

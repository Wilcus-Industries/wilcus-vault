"""Regressions the review of the Python port (PR #46) caught: places where a
Python default quietly differed from what the TypeScript did."""

import http.server
import re
import threading
from datetime import UTC, datetime

import pytest
from conftest import MakeVault
from fakes import fixed_decider

from wilcus_vault import Candidate, Cutoffs, Decision, GateOptions, VaultError, open
from wilcus_vault.cli import main
from wilcus_vault.db import db_path, open_db
from wilcus_vault.decision import check_decision
from wilcus_vault.discards import count_discards
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.frontmatter import patch_frontmatter
from wilcus_vault.http import default_transport
from wilcus_vault.indexer import reindex
from wilcus_vault.note import parse_note, serialize_note, usable_frontmatter
from wilcus_vault.scope import normalize_prefix

embedder = TokenOverlapEmbedder()


async def test_frontmatter_yaml_cannot_serialize_is_malformed_not_fatal(
    make_vault: MakeVault,
) -> None:
    # An explicit tag still builds bytes/dates/sets under SafeLoader; the index
    # stores frontmatter as JSON, so such a note is malformed, never a crash.
    raw = "---\ntitle: Signed\nsigned: !!binary aGk=\n---\nbody\n"
    note = parse_note(raw, "a.md")
    assert note.malformed_frontmatter is True
    assert note.frontmatter == {}
    root = make_vault({"a.md": raw})
    db = open_db(db_path(root))
    try:
        stats = await reindex(db, root, embedder)
        assert stats.added == 1
        assert db.execute("select malformed from notes").fetchone()[0] == 1
    finally:
        db.close()


def test_normalize_prefix_treats_dot_as_the_root() -> None:
    assert normalize_prefix(".") == ""
    assert normalize_prefix("./") == ""


async def test_root_namespace_creates_and_supersedes_with_canonical_paths(
    make_vault: MakeVault,
) -> None:
    root = make_vault({"acme.md": "---\ntitle: Acme\n---\nAcme renewal terms.\n"})
    gate = GateOptions(fixed_decider(Decision("create")), Cutoffs(distance_ceiling=2.0))
    vault = open(root, embedder, gate=gate)
    try:
        created = await vault.propose(Candidate("Root Note", "Root body."))
        assert created.path == "root-note.md"
        gate2 = GateOptions(
            fixed_decider(Decision("supersede", target="acme.md")), Cutoffs(distance_ceiling=2.0)
        )
        vault2 = open(root, embedder, gate=gate2)
        try:
            r = await vault2.propose(Candidate("Acme 2026", "Acme renewal terms, updated."))
            assert r.path == "acme-2026.md"
            assert r.superseded == "acme.md"
            text = (root / "acme.md").read_text()
            assert re.search(r"superseded_by: \"?acme-2026\.md\"?", text)
            assert "[[acme-2026]]" in text
            report = await vault2.doctor()
            assert report.broken_links == []
        finally:
            vault2.close()
    finally:
        vault.close()


def test_serialized_frontmatter_is_one_line_per_key_so_the_patcher_can_edit_it() -> None:
    long = "task-43 " + "again because the scheduler decided to reconcile the ledger " * 3
    raw = serialize_note({"title": "T", "vault_source": long}, "body\n")
    assert raw.count("\n") == 5  # ---, title, vault_source, ---, body
    patched = patch_frontmatter(raw, "updated", "2026-01-01T00:00:00.000Z")
    note = parse_note(patched, "t.md")
    assert note.malformed_frontmatter is False
    assert note.frontmatter["vault_source"] == long
    assert note.frontmatter["updated"] == "2026-01-01T00:00:00.000Z"


class _Redirecting(http.server.BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []

    def do_POST(self) -> None:  # noqa: N802
        self.seen.append((self.path, self.headers.get("authorization")))
        self.send_response(302)
        self.send_header("location", "/elsewhere")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self.seen.append((self.path, self.headers.get("authorization")))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        pass


async def test_transport_never_follows_a_redirect_with_the_key() -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Redirecting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/embeddings"
        status, _ = await default_transport(url, {"authorization": "Bearer k"}, b"{}", 5.0)
        assert status == 302
        assert _Redirecting.seen == [("/v1/embeddings", "Bearer k")]  # /elsewhere never hit
    finally:
        server.shutdown()
        server.server_close()


def test_a_timezone_less_discard_stamp_counts_as_not_recent(make_vault: MakeVault) -> None:
    root = make_vault({})
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    line = '{"at":"%s","candidate":{"title":"t","body":"b"}}\n'
    (root / ".discarded.log").write_text(line % now + line % (now + ".000Z"))
    assert count_discards(root) == {"entries": 2, "recent": 1}


def test_frontmatter_scalars_follow_the_yaml_1_2_core_schema() -> None:
    fm = usable_frontmatter(
        "a: yes\nb: on\nc: 01234\nd: 12:30:00\ne: true\nf: 0x1f\ng: 1.5\n"
        "h: 2026-01-02\ni: ~\nj: =\nk: 0o17\nl: -3\n"
    )
    assert fm == {
        "a": "yes",
        "b": "on",
        "c": 1234,
        "d": "12:30:00",
        "e": True,
        "f": 31,
        "g": 1.5,
        "h": "2026-01-02",
        "i": None,
        "j": "=",
        "k": 15,
        "l": -3,
    }


@pytest.mark.parametrize("field", ["target", "body"])
def test_explicit_null_in_a_decision_is_malformed(field: str) -> None:
    with pytest.raises(VaultError, match=re.escape(f"a non-string {field}")):
        check_decision({"action": "create", field: None})


async def test_discards_show_with_a_unicode_digit_prints_usage(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault({})
    assert await main(["--lexical", "--vault", str(root), "discards", "show", "²"]) == 1
    assert "discards needs list, show <n> or restore <n>" in capsys.readouterr().err


def test_watch_pass_is_only_reported_when_something_changed() -> None:
    from wilcus_vault.cli_commands import pass_line
    from wilcus_vault.indexer import IndexStats

    assert pass_line(["a.md"], IndexStats(unchanged=1)) is None
    line = pass_line(["a.md"], IndexStats(updated=1))
    assert line is not None and line.startswith("a.md — ")

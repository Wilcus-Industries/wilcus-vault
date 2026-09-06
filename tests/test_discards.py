"""Discard log tooling: the read side (list/show/restore) plus the write-side
rails it leans on: size rotation, the auto-gitignore, and the CLI-only decider.
Fake deciders and injected transports, no network."""

import json
import os
import re
from datetime import UTC, datetime

import pytest
from conftest import MakeVault
from fakes import chat_reply, stub_transport

from wilcus_vault.decide import fetch_decider
from wilcus_vault.decision import Candidate, DeciderInput, Decision
from wilcus_vault.discard_log import discard_log, log_candidate
from wilcus_vault.discards import count_discards, get_discard, list_discards, restore_discard
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.gate import GateOptions
from wilcus_vault.search_sql import Cutoffs
from wilcus_vault.term import VaultError
from wilcus_vault.vault import open

CUTOFFS = Cutoffs(distance_ceiling=0.9, bm25_ceiling=0)


def cand(title: str) -> Candidate:
    return Candidate(title, "body\n")


async def test_list_show_restore_round_trip(
    make_vault: MakeVault, embedder: TokenOverlapEmbedder
) -> None:
    root = make_vault({"notes/rota.md": "# Support rota\n\nwho carries the pager\n"})
    decision = Decision("discard")

    async def decider(_input: DeciderInput) -> Decision:
        return decision

    v = open(root, embedder, gate=GateOptions(decider, CUTOFFS))
    await v.reindex()
    await v.propose(Candidate("Pager duty", "Pager rotation is weekly.\n", namespace="notes"))
    await v.propose(Candidate("Second thought", "Another pager candidate.\n", namespace="notes"))

    entries, malformed = list_discards(root)
    assert malformed == 0
    # newest first: 1 is the entry most recently refused
    assert [(e.n, e.candidate.title) for e in entries] == [
        (1, "Second thought"),
        (2, "Pager duty"),
    ]
    assert entries[0].decision == Decision("discard")
    assert re.match(r"^20\d\d-", entries[0].at)

    # show: the full candidate, recoverable byte for byte
    full = get_discard(root, 2)
    assert full is not None
    assert full.candidate == Candidate(
        "Pager duty", "Pager rotation is weekly.\n", namespace="notes"
    )
    assert get_discard(root, 9) is None

    # restore re-runs search+decide against *current* state: the decider now
    # says create, so the candidate lands as a brand-new note through the gate
    decision = Decision("create")
    created = await restore_discard(v, 2)
    assert (created.action, created.path) == ("create", "notes/pager-duty.md")
    assert "Pager rotation is weekly." in (root / "notes/pager-duty.md").read_text()

    # ...and with the vault in a different state the same mechanism lands as an
    # update: current state decides, not the state at discard time
    decision = Decision(
        "update",
        target="notes/rota.md",
        body="# Support rota\n\nwho carries the pager, restored detail\n",
    )
    updated = await restore_discard(v, 1)
    assert (updated.action, updated.path) == ("update", "notes/rota.md")
    assert "restored detail" in (root / "notes/rota.md").read_text()

    with pytest.raises(VaultError, match="no entry 99"):
        await restore_discard(v, 99)
    v.close()


def test_rotation_at_the_cap(make_vault: MakeVault) -> None:
    root = make_vault({})
    # A 1-byte cap makes every write-after-first rotate: .discarded.log fills,
    # moves to .discarded.1.log, then .2, oldest entries in the lowest number.
    log_candidate(root, cand("one"), {"similar": []}, 1)
    log_candidate(root, cand("two"), {"similar": []}, 1)
    log_candidate(root, cand("three"), {"similar": []}, 1)

    assert "one" in (root / ".discarded.1.log").read_text()
    assert "two" in (root / ".discarded.2.log").read_text()
    assert "three" in discard_log(root).read_text()

    # every entry still readable, numbered newest-first across all files
    entries, _ = list_discards(root)
    assert [e.candidate.title for e in entries] == ["three", "two", "one"]
    third = get_discard(root, 3)
    assert third is not None and third.candidate.title == "one"


def test_first_write_creates_gitignore_entry_once(make_vault: MakeVault) -> None:
    root = make_vault({})
    gi = root / ".gitignore"
    log_candidate(root, cand("a"), {"similar": []})
    assert gi.read_text() == ".discarded.log*\n"
    log_candidate(root, cand("b"), {"similar": []})
    assert gi.read_text() == ".discarded.log*\n"  # byte-identical


def test_existing_gitignore_extended_removed_line_stays_removed(make_vault: MakeVault) -> None:
    root = make_vault({})
    gi = root / ".gitignore"
    gi.write_text("node_modules")  # no trailing newline: the append must not fuse lines
    log_candidate(root, cand("a"), {"similar": []})
    assert gi.read_text() == "node_modules\n.discarded.log*\n"

    # The user strips the line while the log exists: their choice, not drift to
    # repair. Only a *first* write (no log on disk) ever touches .gitignore.
    gi.write_text("node_modules\n")
    log_candidate(root, cand("b"), {"similar": []})
    assert gi.read_text() == "node_modules\n"


def test_malformed_log_line_counted_and_skipped(make_vault: MakeVault) -> None:
    root = make_vault({})
    log_candidate(root, cand("good"), {"similar": []})
    with discard_log(root).open("a") as f:
        f.write("not json at all\n")
        f.write('{"at":"2026-01-01T00:00:00Z"}\n')  # json, but no candidate
    entries, malformed = list_discards(root)
    assert [e.candidate.title for e in entries] == ["good"]
    assert malformed == 2


def test_symlink_among_log_files_is_refused(make_vault: MakeVault) -> None:
    # The write side refuses a symlinked log (O_NOFOLLOW); the read side must
    # match, or `discards list` becomes a way to print any readable file, and
    # `restore` a way to propose its content into the vault.
    root = make_vault({})
    log_candidate(root, cand("real"), {"similar": []})
    elsewhere = make_vault({})
    line = {"at": "2026-01-01T00:00:00.000Z", "candidate": {"title": "stolen", "body": "s"}}
    (elsewhere / "secrets.log").write_text(json.dumps(line) + "\n")
    os.symlink(elsewhere / "secrets.log", root / ".discarded.1.log")
    with pytest.raises(VaultError, match="symlink"):
        list_discards(root)


def test_count_discards_splits_total_from_recent(make_vault: MakeVault) -> None:
    root = make_vault({})
    assert count_discards(root) == {"entries": 0, "recent": 0}

    def line(at: str) -> str:
        return json.dumps({"at": at, "candidate": {"title": "t", "body": "b"}}) + "\n"

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    discard_log(root).write_text(line("2020-01-01T00:00:00.000Z") + line(now))
    assert count_discards(root) == {"entries": 2, "recent": 1}


async def test_fetch_decider_model_mandatory_reply_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    # No model, no default: a chat model name cannot be guessed the way the
    # local embedding model can.
    with pytest.raises(VaultError, match="VAULT_DECIDE_MODEL"):
        fetch_decider()

    # The decider posts the gate prompt and parses one strict JSON object back.
    transport = stub_transport(0, chat_reply('{"action":"create"}'))
    decide = fetch_decider(model="test-chat", transport=transport)
    decision = await decide(DeciderInput(Candidate("T", "B\n"), []))
    assert decision == Decision("create")
    call = transport.calls[0]
    assert "localhost:11434" in call.url  # unconfigured stays local
    body = json.dumps(call.body)
    assert "test-chat" in body
    assert "write gate" in body  # the gate prompt, not a bespoke one

    # a chatty or fenced reply is malformed, exactly as the library treats it
    chatty = fetch_decider(
        model="test-chat", transport=stub_transport(0, chat_reply('Sure! {"action":"create"}'))
    )
    with pytest.raises(VaultError, match="write gate"):
        await chatty(DeciderInput(Candidate("T", "B\n"), []))

    # a remote endpoint without a key is refused at construction, like FetchEmbedder
    monkeypatch.setenv("VAULT_DECIDE_ENDPOINT", "https://api.example.invalid/v1/chat/completions")
    with pytest.raises(VaultError, match="API key"):
        fetch_decider(model="m")


def test_a_log_line_whose_values_are_the_wrong_type_is_malformed(make_vault: MakeVault) -> None:
    """Known keys used to be passed through unchecked, so a corrupted or
    hand-edited line built a Candidate holding a number and only failed later,
    inside the write gate, once `restore` was already running."""
    root = make_vault({})
    lines = [
        {"at": "2026-01-01T00:00:00.000Z", "candidate": {"title": 7, "body": "b"}},
        {"at": "2026-01-01T00:00:00.000Z", "candidate": {"title": "t", "body": ["b"]}},
        {
            "at": "2026-01-01T00:00:00.000Z",
            "candidate": {"title": "t", "body": "b"},
            "decision": {"action": "nonsense"},
        },
        {
            "at": "2026-01-01T00:00:00.000Z",
            "candidate": {"title": "t", "body": "b"},
            "similar": [{"path": "a.md", "hash": "h", "score": "not a number"}],
        },
    ]
    ok = {"at": "2026-01-01T00:00:00.000Z", "candidate": {"title": "keeper", "body": "b"}}
    text = "".join(json.dumps(line) + "\n" for line in [*lines, ok])
    discard_log(root).write_text(text, encoding="utf-8")

    entries, malformed = list_discards(root)
    assert malformed == len(lines)
    assert [e.candidate.title for e in entries] == ["keeper"]

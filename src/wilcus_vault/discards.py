"""The read side of the discard log: list, fetch and restore refused candidates.

The log is JSONL on disk, read whole on every call. No index, no cache.
ponytail: a full parse per call, bounded by the rotation cap per file.
"""

import errno
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from .decision import Candidate, Decision
from .discard_log import ROTATED, discard_log, read_nofollow
from .gate_write import GateResult
from .scope import VaultContext
from .term import VaultError

RECENT = timedelta(days=7)


@dataclass(frozen=True)
class DiscardedSimilar:
    path: str
    hash: str
    score: float


@dataclass(frozen=True)
class DiscardEntry:
    # 1 = most recent, numbered across the live log and every rotation. A
    # rotation between a list and a restore can shift numbers: `show <n>` first.
    n: int
    at: str
    candidate: Candidate
    decision: Decision | None  # the decider's verdict; None for a placement failure
    reason: str | None  # why the gate could not place the note
    similar: list[DiscardedSimilar]  # what the decider judged against


def _log_files(root: Path) -> list[Path]:
    """Every log file, oldest first: `.discarded.1.log`, `.2`, ..., then the live log."""
    rotated = sorted(
        (int(m.group(1)), root / m.group(0)) for f in os.listdir(root) if (m := ROTATED.match(f))
    )
    return [path for _, path in rotated] + [discard_log(root)]


def _read_log(file: Path) -> str | None:
    try:
        return read_nofollow(file)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise VaultError(
                f"discards: {file} is a symlink — the log is read in place, never through a link"
            ) from None
        raise


def _parse_line(line: str) -> DiscardEntry | None:
    """One entry without its number, or None for a line that is not one."""
    try:
        raw = json.loads(line)
    except ValueError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("at"), str) or "candidate" not in raw:
        return None
    decision = raw.get("decision")
    similar = raw.get("similar")
    return DiscardEntry(
        n=0,
        at=raw["at"],
        candidate=Candidate(**raw["candidate"]),
        decision=Decision(**decision) if isinstance(decision, dict) else None,
        reason=raw.get("reason"),
        similar=[DiscardedSimilar(**s) for s in similar] if isinstance(similar, list) else [],
    )


def list_discards(root: str | Path) -> tuple[list[DiscardEntry], int]:
    """(entries newest first, malformed line count). A bad line is counted and
    skipped, never fatal: one cannot brick the review of every good one."""
    parsed: list[DiscardEntry] = []
    malformed = 0
    for file in _log_files(Path(root)):
        text = _read_log(file)
        if text is None:
            continue
        for line in text.split("\n"):
            if line.strip() == "":
                continue
            entry = _parse_line(line)
            if entry is None:
                malformed += 1
            else:
                parsed.append(entry)
    parsed.reverse()
    numbered = [
        DiscardEntry(i + 1, e.at, e.candidate, e.decision, e.reason, e.similar)
        for i, e in enumerate(parsed)
    ]
    return numbered, malformed


def get_discard(root: str | Path, n: int) -> DiscardEntry | None:
    entries, _ = list_discards(root)
    return next((e for e in entries if e.n == n), None)


class _Proposer(Protocol):
    @property
    def root(self) -> Path: ...

    async def propose(
        self, candidate: Candidate, ctx: VaultContext | None = None
    ) -> GateResult: ...


async def restore_discard(vault: _Proposer, n: int, ctx: VaultContext | None = None) -> GateResult:
    """Feed a discarded candidate back through the write gate, the only write door.
    It lands by what the vault holds today; the entry stays in the log as history."""
    entry = get_discard(vault.root, n)
    if entry is None:
        raise VaultError(f"discards: no entry {n} — run vault discards list")
    return await vault.propose(entry.candidate, ctx)


def count_discards(root: str | Path) -> dict[str, int]:
    """What doctor reports: how much history there is, and how much is from the last 7 days."""
    entries, _ = list_discards(root)
    cutoff = datetime.now(UTC) - RECENT
    stamps = [_parse_at(e.at) for e in entries]
    recent = sum(1 for at in stamps if at is not None and at >= cutoff)
    return {"entries": len(entries), "recent": recent}


def _parse_at(at: str) -> datetime | None:
    try:
        return datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return None


def entry_to_json(entry: DiscardEntry) -> dict[str, Any]:
    """The entry as the CLI prints it."""
    out: dict[str, Any] = {"n": entry.n, "at": entry.at, "candidate": entry.candidate.to_json()}
    if entry.decision is not None:
        out["decision"] = {k: v for k, v in vars(entry.decision).items() if v is not None}
    if entry.reason is not None:
        out["reason"] = entry.reason
    out["similar"] = [vars(s) for s in entry.similar]
    return out

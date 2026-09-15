"""The read side of the discard log: list, fetch and restore refused candidates.

The log is JSONL on disk, read whole on every call. No index, no cache.
ponytail: a full parse per call, bounded by the rotation cap per file.
"""

import errno
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Protocol, Union, get_args, get_origin, get_type_hints

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
    path: str | None = None  # where a promoted note landed


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
    # A hand-edited or older line may miss a field or carry an extra one: it is
    # a malformed line to skip, never an exception out of doctor.
    try:
        return DiscardEntry(
            n=0,
            at=raw["at"],
            candidate=_build(Candidate, raw["candidate"]),
            decision=_build(Decision, decision) if isinstance(decision, dict) else None,
            reason=raw.get("reason"),
            path=raw.get("path"),
            similar=[_build(DiscardedSimilar, s) for s in similar]
            if isinstance(similar, list)
            else [],
        )
    except TypeError:
        return None


def _fits(annotation: object, value: object) -> bool:
    """Does a JSON value fit a field's annotation? Only the shapes a log line can
    hold: `str`, `float`, a `Literal` of strings, and the optional forms of those."""
    if get_origin(annotation) is Literal:
        return value in get_args(annotation)
    if get_origin(annotation) in (UnionType, Union):  # `str | None`
        return any(_fits(member, value) for member in get_args(annotation))
    if annotation is float:  # JSON writes a whole number without its `.0`
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(annotation, type) and isinstance(value, annotation)


def _build[T](cls: type[T], raw: object) -> T:
    """A dataclass from a JSON object, known fields only, each checked against the
    type it is declared as. Raises TypeError otherwise.

    The values come off disk, so they are read like any other untrusted input: a
    corrupted line must be one skipped entry, not a Candidate holding a number
    that only fails later, inside the write gate a restore already started.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"expected an object for {cls.__name__}")
    # Resolved, not `Field.type`: that is the annotation as written, and a string
    # under `from __future__ import annotations` would fit nothing and drop every entry.
    known = get_type_hints(cls)
    taken = {k: v for k, v in raw.items() if k in known}
    for name, value in taken.items():
        if not _fits(known[name], value):
            raise TypeError(f"{cls.__name__}.{name} is not a {known[name]}")
    return cls(**taken)


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
    return [replace(e, n=i + 1) for i, e in enumerate(parsed)], malformed


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
        parsed = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None  # naive: not comparable, not recent


def entry_to_json(entry: DiscardEntry) -> dict[str, Any]:
    """The entry as the CLI prints it."""
    out: dict[str, Any] = {"n": entry.n, "at": entry.at, "candidate": entry.candidate.to_json()}
    if entry.decision is not None:
        out["decision"] = {k: v for k, v in vars(entry.decision).items() if v is not None}
    if entry.reason is not None:
        out["reason"] = entry.reason
    if entry.path is not None:
        out["path"] = entry.path
    out["similar"] = [vars(s) for s in entry.similar]
    return out

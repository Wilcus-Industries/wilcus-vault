"""Promote: one note through the write gate into a namespace, then removed.

Generic: it knows no layout. Which notes are proposals, and which namespace they
go to, is the caller's business.
"""

import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from .decision import Candidate
from .embed import Embedder
from .gate_write import GateResult
from .indexer import index_paths, read_note
from .note import Note
from .paths import confined_path
from .scope import Scope, VaultContext
from .term import VaultError, safe

# The gate as the vault runs it: the candidate, who asks, and a note never shown as similar.
Gate = Callable[[Candidate, VaultContext | None, str], Awaitable[GateResult]]


@dataclass(frozen=True)
class PromoteResult(GateResult):
    # False when the note changed while the gate ran: that edit is kept, and what
    # was promoted already landed in the gate's note or the discard log.
    removed: bool = field(kw_only=True)


async def promote(
    db: sqlite3.Connection,
    root: Path,
    embedder: Embedder,
    note: Note,  # read through `get`: read-checked, at its canonical path
    namespace: str,
    scope: Scope,
    gate: Gate,
) -> PromoteResult:
    """The note through the gate into `namespace`; removed if it is still what was read."""
    ctx = scope.ctx
    # Before the gate, so a note this agent could never remove costs no model call.
    if not scope.may("write", note.path):
        agent = ctx.agent if ctx else ""
        raise VaultError(
            f'promote: "{safe(agent)}" may not write {safe(note.path)}, so it cannot remove it'
        )
    if ctx is not None and ctx.source is None:
        ctx = replace(ctx, source=note.path)  # the note the gate writes records its origin
    candidate = Candidate(note.title, note.body, note.type, namespace)
    # Kept out of `similar`: shown its own text, a decider finds the candidate
    # already written down and discards it.
    result = await gate(candidate, ctx, note.path)

    # Check-and-remove, like the gate's check-and-write: a note edited meanwhile is kept.
    # ponytail: an edit landing between this re-read and the unlink is lost, the
    # same window check-and-write has; claim the file by rename if that ever matters.
    abs_path = confined_path(root, note.path)
    current = read_note(root, note.path)
    removed = current is not None and current.hash == note.hash
    if removed:
        abs_path.unlink(missing_ok=True)
    # Purges a removed note's row, and re-reads a kept one.
    await index_paths(db, root, embedder, [note.path])
    return PromoteResult(**vars(result), removed=removed)

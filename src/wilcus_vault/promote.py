"""Promote: one note through the write gate into a namespace, then removed.

Generic: it knows no layout. Which notes are proposals, and which namespace they
go to, is the caller's business.
"""

import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

from .decision import Candidate
from .discard_log import log_candidate
from .embed import Embedder
from .gate import GateOptions, propose
from .gate_write import GateResult, close_gate
from .indexer import read_note
from .note import Note
from .paths import canonical_namespace, confined_path
from .scope import Rule, Scope
from .term import VaultError, safe


@dataclass(frozen=True)
class PromoteResult(GateResult):
    # False when the note changed while the gate ran: that edit is kept, not removed.
    removed: bool = field(kw_only=True)


async def promote(
    db: sqlite3.Connection,
    root: Path,
    embedder: Embedder,
    options: GateOptions,
    note: Note,  # read through `get`: read-checked, at its canonical path
    namespace: str,
    scope: Scope,
    walk: bool,  # the closing pass re-reads the whole vault; the caller owns the clock
) -> PromoteResult:
    """The note through the gate into `namespace`; removed if it is still what was read."""
    ctx = scope.ctx
    # Refused before the gate, so a note that cannot be promoted costs no model call.
    if not scope.may("write", note.path):
        agent = ctx.agent if ctx else ""
        raise VaultError(
            f'promote: "{safe(agent)}" may not write {safe(note.path)}, so it cannot remove it'
        )
    if note.malformed_frontmatter:
        raise VaultError(
            f"promote: {safe(note.path)} has malformed frontmatter (unterminated, not YAML, "
            "or a title or type that is not a string), and its block would land in the body"
        )
    if ctx is not None and ctx.source is None:
        ctx = replace(ctx, source=note.path)  # the note the gate writes records its origin
    candidate = Candidate(note.title, note.body, note.type, namespace)
    within = _within(Scope(ctx, scope.rules), canonical_namespace(root, namespace))
    # Excluded as well, for a note that already sits inside the namespace. The
    # closing pass waits until the note is dealt with, below.
    result = await propose(
        db, root, embedder, candidate, options, within, exclude=note.path, close=False
    )

    # Check-and-remove, like the gate's check-and-write: a note edited meanwhile is kept.
    # ponytail: an edit landing between this re-read and the unlink is lost, the
    # same window check-and-write has; claim the file by rename if that ever matters.
    abs_path = confined_path(root, note.path)
    current = read_note(root, note.path)
    removed = current is not None and current.hash == note.hash
    if removed:
        # A decider's own body can drop what the note said, so it is logged whole
        # before the file goes. A discard is in the log already, from the gate.
        if result.action != "discard":
            log_candidate(
                root, candidate, {"reason": "promoted", "path": result.path, "similar": []}
            )
        abs_path.unlink(missing_ok=True)
    # After the removal, with the note's path in it: the row goes in the pass that
    # indexes the new note, so a new note taking the stem is a rename. Run before, the
    # pass sees a stem collision and qualifies every bare link to this note's path,
    # which the removal then breaks. A kept note is re-read, and links to it rightly
    # qualified: it is still there.
    await close_gate(db, root, embedder, result, walk, [note.path])
    return PromoteResult(**vars(result), removed=removed)


def _within(scope: Scope, namespace: str) -> Scope:
    """`scope` confined to one namespace: the policy's rules under it, one rule at it
    carrying the policy's answer there, and the rest of the vault denied. The gate's
    search (in SQL, before its cut to n), read re-check and target write check then
    see only the namespace; anywhere else a note could absorb the promotion."""
    rules = scope.rules or []
    under = [r for r in rules if r.prefix.startswith(namespace) and r.prefix != namespace]
    at = Rule(namespace, scope.may("read", namespace), scope.may("write", namespace))
    elsewhere = [Rule("", False, False)] if namespace else []
    return Scope(scope.ctx, [*under, at, *elsewhere])  # still longest prefix first

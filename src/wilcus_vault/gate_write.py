"""The notes the gate authors itself: a fresh file, or a supersede mark on an old one."""

import hashlib
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .decision import Action, Candidate
from .discard_log import log_candidate
from .frontmatter import patch_frontmatter
from .indexer import read_raw
from .note import link_target, parse_note, serialize_note
from .paths import confined_path, now, slugify, write_atomic, write_new
from .scope import VaultContext
from .term import VaultError

MAX_SLUG_TRIES = 50  # distinct notes may share a title; give up rather than loop forever


# The gate's own frontmatter keys. Namespaced because `agent:` and `source:` are
# exactly what a human's frontmatter plausibly holds.
PROVENANCE_KEYS = ("vault_agent", "vault_source")


@dataclass(frozen=True)
class GateResult:
    action: Action  # what was actually applied; `create` when the decision fell back
    path: str | None = None  # vault-relative path written; None for discard
    superseded: str | None = None  # the note marked superseded_by, for supersede
    # supersede wrote the successor but could not mark this note: it changed in
    # the window, and a human's edit is not overwritten for bookkeeping.
    unmarked: str | None = None
    fell_back: bool = False  # the decision was abandoned; the candidate was created instead


async def mark_superseded(
    root: Path, rel: str, hash_at_read: str, successor: str
) -> tuple[str | None, str | None]:
    """Retire one note in favour of another: (superseded, unmarked), one of them set.

    A textual patch, never a re-serialization, and not restamped: marking is
    bookkeeping, not authorship. Check-and-write against `hash_at_read`.
    """
    abs_path = confined_path(root, rel)
    current = read_raw(root, rel)
    if current is None or parse_note(current, rel).hash != hash_at_read:
        return None, rel
    marked = patch_frontmatter(current, "superseded_by", successor)
    # Linked by path, so a namespaced successor cannot go ambiguous later.
    link = f"Superseded by [[{link_target(successor)}]].\n"
    glue = "\n" if marked.endswith("\n") else "\n\n"
    write_atomic(abs_path, f"{marked}{glue}{link}")
    return rel, None


async def create(
    db: sqlite3.Connection,
    root: Path,
    candidate: Candidate,
    namespace: str,
    body: str | None,
    ctx: VaultContext | None,
) -> GateResult:
    """Write a note we authored ourselves at `<namespace><slug>.md`. `namespace`
    is canonical and already write-checked by the caller.

    The filename is claimed by the write itself rather than checked and then
    written, so a name another writer takes in that window costs this note its
    first choice of slug (`acme-2`), never its content.
    """
    at = now()
    frontmatter: dict[str, object] = {"title": candidate.title}
    if candidate.type is not None:
        frontmatter["type"] = candidate.type
    frontmatter.update(created=at, updated=at, **provenance(ctx))
    text = serialize_note(frontmatter, body if body is not None else candidate.body)
    for slug in _free_slugs(db, candidate):
        rel = f"{namespace}{slug}.md"
        abs_path = confined_path(root, rel)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if write_new(abs_path, text):
            return GateResult("create", rel)
    # Nowhere to put it is still not a reason to drop it on the floor.
    reason = f'write gate: no free filename for "{candidate.title}"'
    log_candidate(root, candidate, {"reason": reason, "similar": []})
    raise VaultError(reason)


def _free_slugs(db: sqlite3.Connection, candidate: Candidate) -> Iterator[str]:
    """Filename stems to try, in order, skipping those another note's stem holds.
    The index is a hint that saves a syscall; the write is what decides."""
    base = slugify(candidate.title)
    if base is None:
        digest = hashlib.sha256(f"{candidate.title}\n\n{candidate.body}".encode()).hexdigest()
        base = f"note-{digest[:8]}"
    for i in range(1, MAX_SLUG_TRIES + 1):
        slug = base if i == 1 else f"{base}-{i}"
        if db.execute("select 1 from notes where slug = ?", (slug,)).fetchone() is None:
            yield slug


def provenance(ctx: VaultContext | None) -> dict[str, str]:
    if ctx is None:
        return {}
    stamp = {"vault_agent": ctx.agent}
    if ctx.source is not None:
        stamp["vault_source"] = ctx.source
    return stamp

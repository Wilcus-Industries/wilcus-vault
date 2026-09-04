"""The notes the gate authors itself: a fresh file, or a supersede mark on an old one."""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .decision import Action, Candidate
from .discard_log import log_candidate
from .frontmatter import patch_frontmatter
from .indexer import read_raw
from .note import link_target, parse_note, serialize_note
from .paths import confined_path, now, slugify, write_atomic
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
    """Write a note we authored ourselves. `namespace` is canonical and already
    write-checked by the caller."""
    rel, abs_path = _free_path(db, root, candidate, namespace)
    at = now()
    frontmatter: dict[str, object] = {"title": candidate.title}
    if candidate.type is not None:
        frontmatter["type"] = candidate.type
    frontmatter.update(created=at, updated=at, **provenance(ctx))
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        abs_path, serialize_note(frontmatter, body if body is not None else candidate.body)
    )
    return GateResult("create", rel)


def _free_path(
    db: sqlite3.Connection, root: Path, candidate: Candidate, namespace: str
) -> tuple[str, Path]:
    """`<namespace><slug>.md`, confined and not taken by a file or another note's
    stem. A collision suffixes (`acme-2`) rather than overwriting."""
    base = slugify(candidate.title)
    if base is None:
        digest = hashlib.sha256(f"{candidate.title}\n\n{candidate.body}".encode()).hexdigest()
        base = f"note-{digest[:8]}"
    for i in range(1, MAX_SLUG_TRIES + 1):
        slug = base if i == 1 else f"{base}-{i}"
        rel = f"{namespace}{slug}.md"
        abs_path = confined_path(root, rel)
        taken = db.execute("select 1 from notes where slug = ?", (slug,)).fetchone()
        if not abs_path.exists() and taken is None:
            return rel, abs_path
    # Nowhere to put it is still not a reason to drop it on the floor.
    reason = f'write gate: no free filename for "{candidate.title}"'
    log_candidate(root, candidate, {"reason": reason, "similar": []})
    raise VaultError(reason)


def provenance(ctx: VaultContext | None) -> dict[str, str]:
    if ctx is None:
        return {}
    stamp = {"vault_agent": ctx.agent}
    if ctx.source is not None:
        stamp["vault_source"] = ctx.source
    return stamp

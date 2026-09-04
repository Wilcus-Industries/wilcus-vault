"""Auto-qualify bare wikilinks when an index pass creates a stem collision.

At the moment a new note's stem matches exactly one note the index already
knew, every bare `[[stem]]` in the vault still unambiguously means that incumbent,
and only this moment can know it. Those links are rewritten to the incumbent's
path-qualified form. The invariant is "never guesses", not "never ambiguous":
anything uncertain is skipped and left to doctor's ambiguous report.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .frontmatter import qualify_links
from .note import Note, is_writable_target, link_target, parse_note
from .paths import confined_path, write_atomic
from .term import printable


@dataclass
class Qualified:
    """One collision's outcome. `rewritten` and `skipped` together account for
    every note that linked the stem bare."""

    stem: str
    target: str | None  # the incumbent's qualified link, or None when it has no usable one
    rewritten: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def detect_collisions(
    root: Path, rows: list[sqlite3.Row], new_notes: list[Note], gone: set[int]
) -> dict[str, str]:
    """stem -> incumbent path, for every new note whose stem exactly one surviving
    indexed note holds. Zero holders is no collision; two or more means the stem
    was already ambiguous. A row in `gone` is a rename, not an incumbent."""
    collisions: dict[str, str] = {}
    for note in new_notes:
        holders = [r for r in rows if r["slug"] == note.slug and r["id"] not in gone]
        if len(holders) != 1:
            continue
        incumbent = holders[0]["path"]
        confined_path(root, incumbent)  # the index is derived data, not a trusted path source
        # The incumbent's file must still exist: a move the watcher sees as
        # create-then-delete would otherwise qualify links to a path about to go.
        if (root / incumbent).is_file() and not (root / incumbent).is_symlink():
            collisions[note.slug] = incumbent
    return collisions


def qualify_collisions(
    db: sqlite3.Connection,
    root: Path,
    collisions: dict[str, str],
    new_paths: set[str],
    cap: int,
) -> tuple[list[Qualified], list[str], str | None]:
    """Rewrite the bare links for each collision. Returns the per-collision report,
    every path rewritten, and the first error behind a skip (if any)."""
    qualified: list[Qualified] = []
    rewritten: list[str] = []
    error: str | None = None
    for stem, incumbent in collisions.items():
        # A root incumbent has no qualified form (its path minus `.md` is the
        # stem), and a path a wikilink cannot carry would destroy the links.
        target: str | None = link_target(incumbent)
        if "/" not in incumbent or not is_writable_target(link_target(incumbent)):
            target = None
        linkers = db.execute(
            """select n.path, n.hash from edges e join notes n on n.id = e.from_id
               where e.to_slug = ? order by n.path""",
            (stem,),
        ).fetchall()
        entry = Qualified(stem, target)
        for linker in linkers:
            path = linker["path"]
            # A bare link inside a note new to this same pass has no settled meaning.
            if target is None or path in new_paths or len(entry.rewritten) >= cap:
                entry.skipped.append(path)
                continue
            abs_path = confined_path(root, path)
            try:
                done = _rewrite(abs_path, path, linker["hash"], stem, target)
            except Exception as e:
                done = False
                error = error or printable(e)
            (entry.rewritten if done else entry.skipped).append(path)
            if done:
                rewritten.append(path)
        if linkers:
            qualified.append(entry)
    return qualified, rewritten, error


def _rewrite(abs_path: Path, rel: str, hash_at_index: str, stem: str, target: str) -> bool:
    """Rewrite one linker in place. False when it was skipped: gone, edited
    since the index read it (never clobbered), or holding no matching link."""
    try:
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    if parse_note(raw, rel).hash != hash_at_index:
        return False
    out = qualify_links(raw, stem, target)
    if out == raw:
        return False
    write_atomic(abs_path, out)
    return True

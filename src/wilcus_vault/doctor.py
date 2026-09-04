"""The safety net for files-are-truth: compare the index to the files, repair
what drifted, and report what only a human can fix."""

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .db import db_path, open_db
from .discard_log import append_nofollow, discard_log, ensure_gitignore, read_nofollow
from .discards import count_discards
from .embed import Embedder
from .indexer import read_note, reindex, scan_vault
from .note import link_target


@dataclass(frozen=True)
class LinkProblem:
    """An unresolved link. `slug` is the target as written: a bare stem or a path."""

    from_path: str
    slug: str


@dataclass(frozen=True)
class AmbiguousLink(LinkProblem):
    """A bare-stem link several notes answer to. `candidates` are link targets
    (`customers/acme`), so writing one into the note is the whole fix."""

    candidates: list[str] = field(default_factory=list)


@dataclass
class DoctorReport:
    stale: list[str]  # files whose indexed hash is wrong or missing
    missing: list[str]  # indexed rows whose file is gone
    broken_links: list[LinkProblem]  # `[[target]]` matching no note at all
    ambiguous_links: list[AmbiguousLink]  # `[[stem]]` matching several notes
    orphans: list[str]  # no resolved link in and no link out
    malformed: list[str]  # frontmatter the parser could not use
    reembedded: bool  # every note was re-embedded: model or dims changed, or a rebuild
    migrated_discard_log: bool  # a log left in `.vault/` was moved beside the notes
    discards: dict[str, int]  # discard log: total entries, and those from the last 7 days


@dataclass(frozen=True)
class DoctorOptions:
    repair: bool = True  # reindex stale notes and purge deleted ones
    rebuild: bool = False  # index into a temp DB and rename it over index.db


async def doctor(
    root: str | Path, embedder: Embedder, options: DoctorOptions | None = None
) -> DoctorReport:
    opts = options or DoctorOptions()
    root = Path(root)
    path = db_path(root)
    # First, before anything touches `.vault/`: history still in there is one
    # `rm -rf .vault` from gone. Only on a repairing run; a report moves nothing.
    migrated = (opts.repair or opts.rebuild) and _migrate_discard_log(root)
    # Drift is measured before any repair, so the report says what was wrong.
    stale, missing = _with_db(path, lambda db: _disk_drift(db, root))
    reembedded = opts.rebuild
    if opts.rebuild:
        await _rebuild_index(root, embedder)
    elif opts.repair:
        db = open_db(path)
        try:
            reembedded = (await reindex(db, root, embedder)).reembedded
        finally:
            db.close()
    broken, ambiguous, orphans, malformed = _with_db(path, _graph_report)
    return DoctorReport(
        stale=stale,
        missing=missing,
        broken_links=broken,
        ambiguous_links=ambiguous,
        orphans=orphans,
        malformed=malformed,
        reembedded=reembedded,
        migrated_discard_log=migrated,
        discards=count_discards(root),
    )


def _migrate_discard_log(root: Path) -> bool:
    """Move a discard log out of the disposable `.vault/` directory, once. Appended
    rather than replaced: both files may hold lines and neither is disposable.
    ponytail: append-then-remove is not atomic; a crash between duplicates lines
    on the next run, never loses them."""
    old = root / ".vault" / "discarded.log"
    if not old.exists():
        return False
    lines = read_nofollow(old)
    if lines:
        log = discard_log(root)
        if not log.exists():
            ensure_gitignore(root)
        append_nofollow(log, lines if lines.endswith("\n") else lines + "\n")
    old.unlink()
    return True


def _with_db[T](path: Path, fn: Callable[[sqlite3.Connection], T]) -> T:
    db = open_db(path)
    try:
        return fn(db)
    finally:
        db.close()


def _disk_drift(db: sqlite3.Connection, root: Path) -> tuple[list[str], list[str]]:
    """(stale, missing): what the files say that the index does not.
    ponytail: re-reads and re-hashes every note; gate on mtime if a vault gets huge."""
    indexed = {r["path"]: r["hash"] for r in db.execute("select path, hash from notes")}
    stale = []
    for rel in scan_vault(root):
        note = read_note(root, rel)
        if note is None:
            continue  # deleted while we looked: it counts as missing
        if indexed.get(rel) != note.hash:
            stale.append(rel)
        indexed.pop(rel, None)
    return stale, sorted(indexed)


def _graph_report(
    db: sqlite3.Connection,
) -> tuple[list[LinkProblem], list[AmbiguousLink], list[str], list[str]]:
    unresolved = db.execute(
        """select n.path as from_path, e.to_slug as slug
           from edges e join notes n on n.id = e.from_id
           where e.to_id is null order by n.path, e.to_slug"""
    ).fetchall()
    # Stems several notes share: legitimate on its own, only reported as the
    # candidate list of a bare link that lands on one.
    shared: dict[str, list[str]] = {}
    for row in db.execute(
        """select slug, path from notes
           where slug in (select slug from notes group by slug having count(*) > 1)
           order by slug, path"""
    ):
        shared.setdefault(row["slug"], []).append(link_target(row["path"]))

    broken: list[LinkProblem] = []
    ambiguous: list[AmbiguousLink] = []
    for row in unresolved:
        # A path-qualified target matches one note or none: never ambiguous.
        candidates = None if "/" in row["slug"] else shared.get(row["slug"])
        if candidates:
            ambiguous.append(AmbiguousLink(row["from_path"], row["slug"], list(candidates)))
        else:
            broken.append(LinkProblem(row["from_path"], row["slug"]))

    orphans = [
        r["path"]
        for r in db.execute(
            """select path from notes n
               where not exists (select 1 from edges e where e.from_id = n.id)
                 and not exists (select 1 from edges e where e.to_id = n.id)
               order by path"""
        )
    ]
    malformed = [
        r["path"] for r in db.execute("select path from notes where malformed = 1 order by path")
    ]
    return broken, ambiguous, orphans, malformed


async def _rebuild_index(root: Path, embedder: Embedder) -> None:
    """Rebuild from the files into a temp DB, then rename it over index.db: a
    half-finished rebuild can never become the live index.
    ponytail: a watcher holding the old file keeps writing to the replaced inode;
    those writes are lost, not corrupting, and the next doctor run recovers them."""
    target = db_path(root)
    tmp = target.with_name(f"{target.name}.rebuild-{uuid.uuid4().hex}")
    db = open_db(tmp)
    try:
        await reindex(db, root, embedder)
        db.execute("pragma wal_checkpoint(truncate)")  # fold the WAL in before the rename
    except BaseException:
        db.close()
        _rm_db(tmp)
        raise
    db.close()
    _rm_db(target, journals_only=True)  # the old journal describes a database about to vanish
    tmp.replace(target)


def _rm_db(path: Path, journals_only: bool = False) -> None:
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    if not journals_only:
        path.unlink(missing_ok=True)

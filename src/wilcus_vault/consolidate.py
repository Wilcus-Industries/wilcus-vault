"""The consolidation pass: merge the near-duplicates a vault accretes.

Manually triggered, never a daemon. Discovery is embedding distance under a
caller-set ceiling; merging reuses the gate's rails with an injected merger in
place of the decider. Dry-run by default, and nothing is ever deleted.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .cluster import Cluster, clusters
from .decision import Candidate
from .embed import Embedder
from .gate_write import create, mark_superseded
from .indexer import read_raw, reindex
from .merge import MergedNote, MergeInput, Merger, check_merged
from .note import Note, parse_note
from .paths import confined_path
from .scope import VaultContext
from .term import VaultError, printable

DEFAULT_CAP = 5  # single digits: a pass that rewrites half the vault is a wrong ceiling


@dataclass(frozen=True)
class ConsolidateOptions:
    merger: Merger


@dataclass(frozen=True)
class ConsolidateRun:
    # Mandatory: there is no universal number for "duplicate", it is a property
    # of the embedder, and a default would merge notes on the library's guess.
    ceiling: float
    cap: int = DEFAULT_CAP  # clusters this run may merge before reporting the rest
    write: bool = False  # actually write; a dry run reports what it would do
    ctx: VaultContext | None = None  # provenance stamped on merged notes, never a permission check


@dataclass
class Merge:
    cluster: Cluster
    candidate: MergedNote  # what the merger answered, written or not
    path: str | None = None  # the merged note; None on a dry run
    superseded: list[str] | None = None  # members marked superseded_by; None on a dry run
    # Members that changed on disk between the read and the marking. A human's
    # edit is not overwritten for bookkeeping; the operator reconciles.
    unmarked: list[str] | None = None


@dataclass
class MergeError:
    cluster: Cluster
    error: str
    path: str | None = None  # the merged note, when the failure landed after it was written


@dataclass
class ConsolidateReport:
    dry_run: bool
    merges: list[Merge] = field(default_factory=list)
    cross_namespace: list[Cluster] = field(default_factory=list)  # reported, never merged
    remaining: list[Cluster] = field(default_factory=list)  # over the cap, or moved under us
    errors: list[MergeError] = field(default_factory=list)  # write runs only; a dry run raises
    index_error: str | None = None  # the closing reindex failed after a merge landed


async def consolidate(
    db: sqlite3.Connection,
    root: str | Path,
    embedder: Embedder,
    options: ConsolidateOptions,
    run: ConsolidateRun,
) -> ConsolidateReport:
    """Find near-duplicate clusters and merge each into one note through the gate's
    rails: `create` for the merged note, then `mark_superseded` for every member."""
    ceiling = run.ceiling
    if isinstance(ceiling, bool) or not isinstance(ceiling, int | float) or not 0 <= ceiling <= 2:
        raise VaultError(
            "consolidate: ceiling must be a cosine distance in 0..2 — there is no universal "
            'number for "duplicate", it is a property of the embedder'
        )
    if isinstance(run.cap, bool) or not isinstance(run.cap, int) or run.cap < 1:
        raise VaultError(f"consolidate: cap must be a positive integer, got {run.cap}")
    if run.ctx is not None and run.ctx.agent.strip() == "":
        raise VaultError("consolidate: ctx.agent must name the calling agent")
    base = Path(root).absolute()
    # Discovery runs against a current index: a note edited or deleted behind
    # the vault's back must not be merged as it used to read.
    await reindex(db, base, embedder)

    found = clusters(db, ceiling)
    for cluster in found:
        for path in cluster.members:
            confined_path(base, path)  # a member path escaping the vault is a tampered index
    report = ConsolidateReport(dry_run=not run.write)
    report.cross_namespace = [c for c in found if c.namespace is None]
    wrote = False
    acted = 0  # clusters whose merger ran; a pre-call failure burns no slot
    for cluster in (c for c in found if c.namespace is not None):
        if acted >= run.cap:
            report.remaining.append(cluster)
            continue
        created: str | None = None
        try:
            notes = _read_members(base, cluster.members)
            if notes is None:
                # A member vanished: not the cluster we would ask about.
                report.remaining.append(cluster)
                continue
            acted += 1
            candidate = check_merged(await options.merger(MergeInput(notes)))
            if not run.write:
                report.merges.append(Merge(cluster, candidate))
                continue
            merged = Candidate(candidate.title, candidate.body, candidate.type)
            assert cluster.namespace is not None
            created = (await create(db, base, merged, cluster.namespace, None, run.ctx)).path
            assert created is not None
            wrote = True
            superseded: list[str] = []
            unmarked: list[str] = []
            for note in notes:
                marked, _ = await mark_superseded(base, note.path, note.hash, created)
                (superseded if marked else unmarked).append(note.path)
            report.merges.append(Merge(cluster, candidate, created, superseded, unmarked))
        except Exception as e:
            # A dry run raises. On a write run one bad cluster must not discard
            # the report of the merges that already landed.
            if not run.write:
                raise
            report.errors.append(MergeError(cluster, printable(e), created))
    if wrote:
        try:
            await reindex(db, base, embedder)
        except Exception as e:
            report.index_error = printable(e)
    return report


def _read_members(root: Path, members: list[str]) -> list[Note] | None:
    """Every member re-read from disk, or None when one is gone."""
    notes = []
    for path in members:
        raw = read_raw(root, path)
        if raw is None:
            return None
        notes.append(parse_note(raw, path))
    return notes

"""The write gate: every programmatic write goes through `propose`.

Search for what already exists, ask an injected decider what to do, then apply
the answer behind two rails: check-and-write (nothing a human touched mid-flight
is clobbered) and path confinement (an LLM-derived string never names a raw path).
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .decision import (
    Candidate,
    Decider,
    DeciderInput,
    Decision,
    SimilarNote,
    check_decision,
)
from .discard_log import log_candidate
from .embed import Embedder
from .frontmatter import patch_frontmatter, replace_body
from .gate_write import PROVENANCE_KEYS, GateResult, create, mark_superseded, provenance
from .indexer import index_paths, read_raw, reindex
from .note import parse_note
from .paths import canonical_namespace, confined_path, now, write_atomic
from .scope import ALLOW_ALL, Scope, VaultContext
from .search import SearchOptions, hybrid_search
from .search_sql import Cutoffs
from .term import VaultError, safe


@dataclass(frozen=True)
class GateOptions:
    decider: Decider
    # Mandatory: without cutoffs the search always returns something, and "most
    # similar note" degrades into "least unrelated note".
    cutoffs: Cutoffs
    n: int = 5  # similar notes to show the decider
    # How stale the gate's view of the files may be, in seconds. The closing pass
    # re-reads every note, so a burst of writes otherwise pays for the whole vault
    # once per note. Zero — the default — walks on every write, as it always has.
    # Only edits made outside the vault API go unseen for the window; a note the
    # gate wrote itself is indexed before it returns, whatever this is set to.
    freshness: float = 0.0


async def propose(
    db: sqlite3.Connection,
    root: str | Path,
    embedder: Embedder,
    candidate: Candidate,
    options: GateOptions,
    scope: Scope = ALLOW_ALL,
    refresh: bool = True,  # walk the vault on the way out; the caller owns the clock
    exclude: str | None = None,  # a note never shown as similar: the one being promoted
) -> GateResult:
    """Search, decide, apply. A target that changed under us re-runs the gate once
    against fresh state; a second mismatch falls back to create."""
    cutoffs = options.cutoffs
    if cutoffs.distance_ceiling is None and cutoffs.bm25_ceiling is None:
        raise VaultError(
            "write gate: cutoffs must set distance_ceiling or bm25_ceiling — without one, "
            "'most similar note' is only 'least unrelated note'"
        )
    ctx = scope.ctx
    if ctx is not None and ctx.agent.strip() == "":
        raise VaultError("write gate: ctx.agent must name the calling agent")
    base = Path(root).absolute()
    # Canonicalized through the confinement rail and used from here on: checking
    # one spelling and writing another (`notes/../ledger`) would be a scope bypass.
    namespace = canonical_namespace(base, candidate.namespace)
    if not scope.may("write", namespace):
        agent = ctx.agent if ctx else ""
        where = "the vault root" if namespace == "" else safe(namespace)
        raise VaultError(f'write gate: "{safe(agent)}" may not write to {where}')

    applied: GateResult | None = None
    stale: str | None = None  # a target that changed under us: on disk, not yet indexed
    for _ in range(2):
        if stale is not None:
            await index_paths(db, base, embedder, [stale])
        similar = await _find_similar(db, base, embedder, candidate, options, scope, exclude)
        decision = check_decision(await options.decider(DeciderInput(candidate, similar)))
        if decision.target is not None and not any(s.note.path == decision.target for s in similar):
            raise VaultError(
                f"write gate: decider targeted {decision.target}, "
                "which was not among the similar notes"
            )
        # A target the agent may read but not write falls back to create below.
        if decision.target is not None and not scope.may("write", decision.target):
            break
        applied = await _apply(db, base, candidate, namespace, decision, similar, ctx, exclude)
        if applied is not None:
            break
        stale = decision.target
    fell_back = applied is None
    if applied is None:
        applied = await create(db, base, candidate, namespace, None, ctx, exclude)
    # The whole walk is the expensive part of a propose: dirtiness is decided by
    # content hash, so it re-reads every note in the vault. What it buys is the
    # next call's search seeing a note a human edited behind our back — without
    # which the gate re-creates notes that already exist. Skipping it is the
    # caller's call; indexing what we just wrote is not optional either way.
    if refresh:
        await reindex(db, base, embedder)
    else:
        touched = [p for p in (applied.path, applied.superseded, applied.unmarked) if p is not None]
        if touched:
            await index_paths(db, base, embedder, touched)
    return GateResult(applied.action, applied.path, applied.superseded, applied.unmarked, fell_back)


async def _find_similar(
    db: sqlite3.Connection,
    root: Path,
    embedder: Embedder,
    candidate: Candidate,
    options: GateOptions,
    scope: Scope,
    exclude: str | None,
) -> list[SimilarNote]:
    """Top-k similar notes, re-read from disk so their hashes are current."""
    query = f"{candidate.title}\n\n{candidate.body}"  # the shape the indexer embeds
    # One more when a note is excluded, so the decider still sees up to n others.
    n = options.n if exclude is None else options.n + 1
    hits = await hybrid_search(db, embedder, query, SearchOptions(n, options.cutoffs), scope)
    similar = []
    for hit in hits:
        if hit.path == exclude:
            continue
        # Filtered in SQL already; checked again here because this is where note
        # bodies leave the vault and enter a prompt.
        if not scope.may("read", hit.path):
            continue
        confined_path(root, hit.path)
        raw = read_raw(root, hit.path)
        if raw is None:
            continue  # indexed but gone: the row is stale
        note = parse_note(raw, hit.path)
        read_only = not scope.may("write", hit.path)
        similar.append(SimilarNote(note, hit.score, note.hash, read_only))
    return similar


async def _apply(
    db: sqlite3.Connection,
    root: Path,
    candidate: Candidate,
    namespace: str,
    decision: Decision,
    similar: list[SimilarNote],
    ctx: VaultContext | None,
    exclude: str | None,
) -> GateResult | None:
    """Apply one decision, or None if the target changed under us."""
    if decision.action == "discard":
        judged = [{"path": s.note.path, "hash": s.hash, "score": s.score} for s in similar]
        log_candidate(root, candidate, {"decision": _decision_json(decision), "similar": judged})
        return GateResult("discard")
    if decision.action == "create":
        return await create(db, root, candidate, namespace, decision.body, ctx, exclude)

    hit = next(s for s in similar if s.note.path == decision.target)
    rel = hit.note.path
    abs_path = confined_path(root, rel)
    raw = read_raw(root, rel)
    if raw is None or parse_note(raw, rel).hash != hit.hash:
        return None

    if decision.action == "update":
        # Frontmatter is the note's identity: keep it, bump `updated`, and set or
        # unset every gate-owned key to match this call exactly.
        stamp = provenance(ctx)
        patched = patch_frontmatter(raw, "updated", now())
        for key in PROVENANCE_KEYS:
            patched = patch_frontmatter(patched, key, stamp.get(key))
        body = decision.body if decision.body is not None else candidate.body
        write_atomic(abs_path, replace_body(patched, body))
        return GateResult("update", rel)

    # supersede: the successor is written first, then the old note is marked.
    created = await create(db, root, candidate, namespace, decision.body, ctx, exclude)
    assert created.path is not None
    marked, unmarked = await mark_superseded(root, rel, hit.hash, created.path)
    return GateResult("supersede", created.path, marked, unmarked)


def _decision_json(decision: Decision) -> dict[str, str]:
    return {k: v for k, v in vars(decision).items() if v is not None}

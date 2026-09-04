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
from .indexer import read_raw, reindex
from .note import parse_note
from .paths import confined_path, now, write_atomic
from .scope import ALLOW_ALL, Scope, VaultContext, normalize_prefix
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


async def propose(
    db: sqlite3.Connection,
    root: str | Path,
    embedder: Embedder,
    candidate: Candidate,
    options: GateOptions,
    scope: Scope = ALLOW_ALL,
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
    namespace_abs = confined_path(base, candidate.namespace or "")
    namespace = normalize_prefix(namespace_abs.relative_to(base).as_posix())
    if not scope.may("write", namespace):
        agent = ctx.agent if ctx else ""
        where = "the vault root" if namespace == "" else safe(namespace)
        raise VaultError(f'write gate: "{safe(agent)}" may not write to {where}')

    applied: GateResult | None = None
    for attempt in range(2):
        if attempt > 0:
            await reindex(db, base, embedder)  # the aborting edit is on disk, not yet indexed
        similar = await _find_similar(db, base, embedder, candidate, options, scope)
        decision = check_decision(await options.decider(DeciderInput(candidate, similar)))
        if decision.target is not None and not any(s.note.path == decision.target for s in similar):
            raise VaultError(
                f"write gate: decider targeted {decision.target}, "
                "which was not among the similar notes"
            )
        # A target the agent may read but not write falls back to create below.
        if decision.target is not None and not scope.may("write", decision.target):
            break
        applied = await _apply(db, base, candidate, namespace, decision, similar, ctx)
        if applied is not None:
            break
    fell_back = applied is None
    if applied is None:
        applied = await create(db, base, candidate, namespace, None, ctx)
    await reindex(db, base, embedder)  # the index never lags a write we made ourselves
    return GateResult(applied.action, applied.path, applied.superseded, applied.unmarked, fell_back)


async def _find_similar(
    db: sqlite3.Connection,
    root: Path,
    embedder: Embedder,
    candidate: Candidate,
    options: GateOptions,
    scope: Scope,
) -> list[SimilarNote]:
    """Top-k similar notes, re-read from disk so their hashes are current."""
    query = f"{candidate.title}\n\n{candidate.body}"  # the shape the indexer embeds
    hits = await hybrid_search(
        db, embedder, query, SearchOptions(options.n, options.cutoffs), scope
    )
    similar = []
    for hit in hits:
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
) -> GateResult | None:
    """Apply one decision, or None if the target changed under us."""
    if decision.action == "discard":
        judged = [{"path": s.note.path, "hash": s.hash, "score": s.score} for s in similar]
        log_candidate(root, candidate, {"decision": _decision_json(decision), "similar": judged})
        return GateResult("discard")
    if decision.action == "create":
        return await create(db, root, candidate, namespace, decision.body, ctx)

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
    created = await create(db, root, candidate, namespace, decision.body, ctx)
    assert created.path is not None
    marked, unmarked = await mark_superseded(root, rel, hit.hash, created.path)
    return GateResult("supersede", created.path, marked, unmarked)


def _decision_json(decision: Decision) -> dict[str, str]:
    return {k: v for k, v in vars(decision).items() if v is not None}

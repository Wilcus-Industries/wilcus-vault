"""The write gate: every programmatic write goes through `propose`.

Search for what already exists, ask an injected decider what to do, then apply
the answer behind two rails: check-and-write (nothing a human touched mid-flight
is clobbered) and path confinement (an LLM-derived string never names a raw path).
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .decision import (
    Action,
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
from .indexer import read_raw, reindex
from .note import link_target, parse_note, serialize_note
from .paths import confined_path, now, slugify, write_atomic
from .scope import ALLOW_ALL, Scope, VaultContext, normalize_prefix
from .search import Cutoffs, SearchOptions, hybrid_search
from .term import VaultError, safe

MAX_SLUG_TRIES = 50  # distinct notes may share a title; give up rather than loop forever
# The gate's own frontmatter keys. Namespaced because `agent:` and `source:` are
# exactly what a human's frontmatter plausibly holds.
PROVENANCE_KEYS = ("vault_agent", "vault_source")


@dataclass(frozen=True)
class GateOptions:
    decider: Decider
    # Mandatory: without cutoffs the search always returns something, and "most
    # similar note" degrades into "least unrelated note".
    cutoffs: Cutoffs
    n: int = 5  # similar notes to show the decider


@dataclass(frozen=True)
class GateResult:
    action: Action  # what was actually applied; `create` when the decision fell back
    path: str | None = None  # vault-relative path written; None for discard
    superseded: str | None = None  # the note marked superseded_by, for supersede
    # supersede wrote the successor but could not mark this note: it changed in
    # the window, and a human's edit is not overwritten for bookkeeping.
    unmarked: str | None = None
    fell_back: bool = False  # the decision was abandoned; the candidate was created instead


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
        stamp = _provenance(ctx)
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
    frontmatter.update(created=at, updated=at, **_provenance(ctx))
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


def _provenance(ctx: VaultContext | None) -> dict[str, str]:
    if ctx is None:
        return {}
    stamp = {"vault_agent": ctx.agent}
    if ctx.source is not None:
        stamp["vault_source"] = ctx.source
    return stamp


def _decision_json(decision: Decision) -> dict[str, str]:
    return {k: v for k, v in vars(decision).items() if v is not None}

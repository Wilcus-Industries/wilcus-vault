"""Hybrid retrieval: vector KNN and FTS5 BM25, each cut to relevance, then fused by RRF.

RRF scores are ordinal (the top hit scores 1/61 however bad it is), so a cutoff
on the fused score cannot filter irrelevance. Cutoffs apply per signal, before
fusion, and RRF only orders the survivors.
"""

import sqlite3
from dataclasses import dataclass, field

from .embed import Embedder, Vector, l2_normalize
from .scope import ALLOW_ALL, Scope, VaultContext
from .search_sql import Cutoffs, SearchHit, expand, fuse
from .term import VaultError

# A whole note body is a legitimate query (the write gate passes one), but a
# MATCH with a term per word of it is a slow way to ask a vague question.
# ponytail: first N distinct terms; weight by IDF if the truncation ever costs a hit.
MAX_FTS_TERMS = 32


@dataclass(frozen=True)
class SearchOptions:
    n: int = 10  # hits to return; expansion may append up to N more
    cutoffs: Cutoffs = field(default_factory=Cutoffs)
    expand_links: bool = False  # append one-hop wikilink neighbours below every hit
    ctx: VaultContext | None = None  # who is asking; resolved by the vault facade


def fts_query(text: str) -> str | None:
    """User text as an FTS5 MATCH expression that cannot be FTS5 syntax.

    Each whitespace-separated run becomes one quoted phrase, so `NEAR(`, `OR`
    and `*` are matched as words. Terms with no letter or digit are dropped;
    a query with none at all yields None.
    """
    terms = dict.fromkeys(t for t in text.split() if any(ch.isalnum() for ch in t))
    quoted = ['"' + t.replace('"', '""') + '"' for t in list(terms)[:MAX_FTS_TERMS]]
    return " OR ".join(quoted) if quoted else None


async def hybrid_search(
    db: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    options: SearchOptions | None = None,
    scope: Scope | None = None,
) -> list[SearchHit]:
    opts = options or SearchOptions()
    if isinstance(opts.n, bool) or not isinstance(opts.n, int) or opts.n < 1:
        raise VaultError(f"search: n must be a positive integer, got {opts.n}")
    # Only the facade holds the policy, so only it can turn a ctx into a scope.
    if opts.ctx is not None and scope is None:
        raise VaultError("search: options.ctx is resolved by the vault — call vault.search()")
    scoped = scope or ALLOW_ALL
    _assert_indexed_with(db, embedder)

    match = fts_query(query)
    vector = await _query_vector(db, embedder, query)
    if match is None and vector is None:
        return []
    try:
        rows = fuse(db, vector, match, opts.n, opts.cutoffs, scoped)
    except sqlite3.OperationalError as e:
        # The quoting above should make an FTS5 parse error unreachable; if one
        # slips through anyway, the keyword signal drops out rather than
        # taking the whole search down.
        if match is None or "fts5" not in str(e).lower():
            raise
        rows = fuse(db, vector, None, opts.n, opts.cutoffs, scoped)
    if not opts.expand_links:
        return rows
    return rows + expand(db, rows, opts.n, scoped)


async def _query_vector(db: sqlite3.Connection, embedder: Embedder, query: str) -> Vector | None:
    """The normalized query vector, or None when there is no usable vector signal:
    nothing embedded yet, or a query with no tokens the embedder knows (a zero
    vector has no direction, and cosine distance against it is NaN)."""
    has_vectors = db.execute("select 1 from sqlite_master where name = 'vectors'").fetchone()
    if has_vectors is None:
        return None
    vectors = await embedder.embed([query])
    v = vectors[0] if vectors else None
    if v is None or len(v) != embedder.dims:
        width = 0 if v is None else len(v)
        raise VaultError(
            f"embedder {embedder.model} returned a vector of width {width} for the query, "
            f"expected {embedder.dims}"
        )
    q = l2_normalize(v)
    return q if any(x != 0 for x in q) else None


def _assert_indexed_with(db: sqlite3.Connection, embedder: Embedder) -> None:
    """Vectors from another model are not comparable; doctor re-embeds them."""
    other = db.execute(
        "select model, dims from vector_meta where model <> ? or dims <> ? limit 1",
        (embedder.model, embedder.dims),
    ).fetchone()
    if other is not None:
        raise VaultError(
            f"index was embedded with {other['model']}/{other['dims']}, searching with "
            f"{embedder.model}/{embedder.dims} — run vault doctor to re-embed"
        )

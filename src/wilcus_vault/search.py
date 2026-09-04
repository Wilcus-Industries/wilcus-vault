"""Hybrid retrieval: vector KNN and FTS5 BM25, each cut to relevance, then fused by RRF.

RRF scores are ordinal (the top hit scores 1/61 however bad it is), so a cutoff
on the fused score cannot filter irrelevance. Cutoffs apply per signal, before
fusion, and RRF only orders the survivors.
"""

import sqlite3
from dataclasses import dataclass, field

from .db import to_blob
from .embed import Embedder, Vector, l2_normalize
from .scope import ALLOW_ALL, Scope, VaultContext
from .term import VaultError

RRF_K = 60
OVERFETCH = 3  # k = 3×N per signal, so filters cannot starve the result set
# A whole note body is a legitimate query (the write gate passes one), but a
# MATCH with a term per word of it is a slow way to ask a vague question.
# ponytail: first N distinct terms; weight by IDF if the truncation ever costs a hit.
MAX_FTS_TERMS = 32


@dataclass(frozen=True)
class SearchHit:
    id: int
    path: str
    title: str
    score: float  # fused RRF score; 0 for an expansion hit
    vec_rank: int | None  # rank on the vector signal after its cutoff
    fts_rank: int | None  # rank on the BM25 signal after its cutoff
    expansion: bool = False  # reached by a one-hop wikilink, not by a signal


@dataclass(frozen=True)
class Cutoffs:
    """Per-signal relevance cutoffs. Both are upper bounds on a lower-is-better
    quantity: cosine distance (0 identical, 1 orthogonal) and FTS5's negative
    BM25 rank. Off by default: the right ceiling is a property of the embedder."""

    distance_ceiling: float | None = None
    bm25_ceiling: float | None = None


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
        rows = _fuse(db, vector, match, opts.n, opts.cutoffs, scoped)
    except sqlite3.OperationalError as e:
        # The quoting above should make an FTS5 parse error unreachable; if one
        # slips through anyway, the keyword signal drops out rather than
        # taking the whole search down.
        if match is None or "fts5" not in str(e).lower():
            raise
        rows = _fuse(db, vector, None, opts.n, opts.cutoffs, scoped)
    if not opts.expand_links:
        return rows
    return rows + _expand(db, rows, opts.n, scoped)


def _fuse(
    db: sqlite3.Connection,
    vector: Vector | None,
    match: str | None,
    n: int,
    cutoffs: Cutoffs,
    scope: Scope,
) -> list[SearchHit]:
    """One statement: both signals, their cutoffs, and the RRF fusion over them."""
    readable_sql, readable_params = scope.read_sql
    params: list[object] = []
    knn = "select null as id, 0.0 as distance where 0"
    if vector is not None:
        knn = "select note_id as id, distance from vectors where emb match ? and k = ?"
        params += [to_blob(vector), OVERFETCH * n]
    params += readable_params
    vec_cutoff = ""
    if cutoffs.distance_ceiling is not None:
        vec_cutoff = "and knn.distance <= ?"
        params.append(cutoffs.distance_ceiling)
    bm25 = "select null as id, 0.0 as score where 0"
    if match is not None:
        bm25 = """select rowid as id, rank as score from notes_fts
                  where notes_fts match ? order by rank, rowid limit ?"""
        params += [match, OVERFETCH * n]
    params += readable_params
    fts_cutoff = ""
    if cutoffs.bm25_ceiling is not None:
        fts_cutoff = "and hits.score <= ?"
        params.append(cutoffs.bm25_ceiling)
    params.append(n)

    # Cutoffs, the supersede filter and the scope check sit in the WHERE, so
    # each side ranks among survivors and fusion never sees the rest.
    rows = db.execute(
        f"""with knn as materialized ({knn}),
          vecq as (
            select knn.id as id, row_number() over (order by knn.distance, knn.id) as r
            from knn join notes n on n.id = knn.id
            where n.superseded_by is null and ({readable_sql}) {vec_cutoff}
          ),
          hits as materialized ({bm25}),
          ftsq as (
            select hits.id as id, row_number() over (order by hits.score, hits.id) as r
            from hits join notes n on n.id = hits.id
            where n.superseded_by is null and ({readable_sql}) {fts_cutoff}
          )
        select n.id as id, n.path as path, n.title as title,
               coalesce(1.0/({RRF_K}+vecq.r), 0) + coalesce(1.0/({RRF_K}+ftsq.r), 0) as score,
               vecq.r as vec_rank, ftsq.r as fts_rank
        from vecq full outer join ftsq on vecq.id = ftsq.id
        join notes n on n.id = coalesce(vecq.id, ftsq.id)
        order by score desc, n.path
        limit ?""",
        params,
    ).fetchall()
    return [
        SearchHit(r["id"], r["path"], r["title"], r["score"], r["vec_rank"], r["fts_rank"])
        for r in rows
    ]


def _expand(db: sqlite3.Connection, hits: list[SearchHit], n: int, scope: Scope) -> list[SearchHit]:
    """Notes one wikilink away from a hit, either direction, capped at N of their own.
    Neighbours pass the same read check as direct hits."""
    if not hits:
        return []
    ids = [h.id for h in hits]
    marks = ",".join("?" * len(ids))
    readable_sql, readable_params = scope.read_sql
    rows = db.execute(
        f"""select n.id as id, n.path as path, n.title as title
        from notes n
        where n.superseded_by is null and ({readable_sql})
          and n.id not in ({marks})
          and exists (
            select 1 from edges e
            where (e.from_id = n.id and e.to_id in ({marks}))
               or (e.to_id = n.id and e.from_id in ({marks}))
          )
        order by n.path
        limit ?""",
        [*readable_params, *ids, *ids, *ids, n],
    ).fetchall()
    return [SearchHit(r["id"], r["path"], r["title"], 0, None, None, expansion=True) for r in rows]


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

"""The two SQL statements behind hybrid search: the fused ranking and the link expansion."""

import sqlite3
from dataclasses import dataclass

from .db import to_blob
from .embed import Vector
from .scope import Scope

RRF_K = 60


OVERFETCH = 3  # k = 3×N per signal, so filters cannot starve the result set


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


def fuse(
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


def expand(db: sqlite3.Connection, hits: list[SearchHit], n: int, scope: Scope) -> list[SearchHit]:
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

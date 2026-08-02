// Hybrid retrieval per DESIGN.md § Retrieval. Two independent signals, each
// over-fetched and cut to relevance *before* fusion: RRF scores are ordinal
// (the top hit scores 1/61 however bad it is), so a threshold on the fused
// score cannot filter irrelevance — only a per-signal cutoff can. RRF then
// does one job, ordering the survivors.
import type { Database, SQLQueryBindings } from "bun:sqlite";
import { l2normalize, type Embedder } from "./embed";
import { ALLOW_ALL, type Scope } from "./scope";
import type { VaultContext } from "./gate";

/** RRF constant from the literature: score = Σ 1/(60 + rank). */
const RRF_K = 60;
/** k = 3×N per signal, so the supersede filter and the cutoffs cannot starve N. */
const OVERFETCH = 3;
/**
 * A whole note body is a legitimate query — the write gate passes one — and an
 * FTS5 MATCH with a term per word of it is a slow way to ask a vague question.
 * ponytail: first N distinct terms, no scoring; weight by IDF if the truncation
 * ever costs a real hit.
 */
const MAX_FTS_TERMS = 32;

export type SearchHit = {
  id: number;
  /** vault-relative path — the note's identity */
  path: string;
  title: string;
  /** fused RRF score; 0 for an expansion hit, which has no rank on any signal */
  score: number;
  /** rank on the vector signal after its cutoff, or null if it did not survive */
  vecRank: number | null;
  /** rank on the BM25 signal after its cutoff, or null if it did not survive */
  ftsRank: number | null;
  /** reached by a one-hop wikilink from a direct hit, not by a signal */
  expansion: boolean;
};

/**
 * Per-signal relevance cutoffs, applied before fusion. Named as one field so a
 * caller has to decide about them: the write gate MUST pass cutoffs — without
 * them "most similar note" always returns *something*, and the gate would
 * update the least-unrelated note instead of creating a new one.
 */
export type Cutoffs = {
  /**
   * Drop vector hits whose cosine distance is above this (0 = identical,
   * 1 = orthogonal). Off by default: the right ceiling is a property of the
   * embedder, so it is the caller's policy, not the library's.
   */
  distanceCeiling?: number;
  /**
   * Drop FTS hits whose BM25 score is above this. FTS5's `rank` is negative
   * and lower is better, so this is an upper bound like `distanceCeiling`:
   * −1 is stricter than 0. Off by default.
   */
  bm25Ceiling?: number;
};

export type SearchOptions = {
  /** hits to return (default 10); expansion may append up to N more */
  n?: number;
  cutoffs?: Cutoffs;
  /** append one-hop wikilink neighbours of the survivors, below every hit */
  expandLinks?: boolean;
  /**
   * Who is asking (DESIGN.md § Scopes and context). The facade resolves this
   * against the vault's `ScopePolicy` and hands `hybridSearch` the resolved
   * scope — required when a policy is in force, ignored when there is none.
   */
  ctx?: VaultContext;
};

/**
 * Turn user text into an FTS5 MATCH expression that cannot be FTS5 syntax:
 * every whitespace-separated run becomes one quoted phrase (embedded quotes
 * doubled), so `NEAR(`, `OR`, `*`, `^` and friends are matched as words.
 * Terms carrying no letter or digit produce no phrase; a query with none at
 * all yields null — there is nothing to ask FTS5 for. Capped at the first
 * `MAX_FTS_TERMS` distinct terms.
 */
export function ftsQuery(text: string): string | null {
  const terms = [...new Set(text.split(/\s+/).filter((t) => /[\p{L}\p{N}]/u.test(t)))]
    .slice(0, MAX_FTS_TERMS)
    .map((t) => `"${t.replaceAll('"', '""')}"`);
  return terms.length > 0 ? terms.join(" OR ") : null;
}

export async function hybridSearch(
  db: Database,
  embedder: Embedder,
  query: string,
  { n = 10, cutoffs = {}, expandLinks = false, ctx }: SearchOptions = {},
  scope?: Scope,
): Promise<SearchHit[]> {
  if (!Number.isInteger(n) || n < 1) {
    throw new Error(`search: n must be a positive integer, got ${n}`);
  }
  // Only the facade holds the policy, so only the facade can turn a `ctx` into
  // a scope. A direct caller passing one would otherwise get allow-all — the
  // silent grant an allowlist exists to prevent.
  if (ctx !== undefined && scope === undefined) {
    throw new Error("search: options.ctx is resolved by the vault — call vault.search()");
  }
  const scoped = scope ?? ALLOW_ALL;
  assertIndexedWith(db, embedder);

  const match = ftsQuery(query);
  const vector = await queryVector(db, embedder, query);
  if (match === null && vector === null) return []; // nothing to search with

  let rows: Omit<SearchHit, "expansion">[];
  try {
    rows = fuse(db, vector, match, n, cutoffs, scoped);
  } catch (e) {
    // The quoting above should make an FTS5 parse error unreachable; if a
    // tokenizer or Unicode-table skew still produces one, the keyword signal
    // drops out rather than taking the whole search down with it.
    if (match === null || !/fts5/i.test(String(e))) throw e;
    rows = fuse(db, vector, null, n, cutoffs, scoped);
  }

  const direct = rows.map((r) => ({ ...r, expansion: false }));
  return expandLinks ? [...direct, ...expand(db, direct, n, scoped)] : direct;
}

/** One statement: both signals, their cutoffs, and the RRF fusion over them. */
function fuse(
  db: Database,
  vector: Float32Array | null,
  match: string | null,
  n: number,
  { distanceCeiling, bm25Ceiling }: Cutoffs,
  scope: Scope,
): Omit<SearchHit, "expansion">[] {
  // Both sides are optional, so the SQL is assembled around the ones we have
  // and the parameters are pushed in the order they appear in it.
  const params: SQLQueryBindings[] = [];
  const readable = scope.readSql;
  let knn = `select null as id, 0.0 as distance where 0`;
  if (vector !== null) {
    knn = `select note_id as id, distance from vectors where emb match ? and k = ?`;
    params.push(vector, OVERFETCH * n);
  }
  params.push(...readable.params);
  let vecCutoff = "";
  if (distanceCeiling !== undefined) {
    vecCutoff = `and knn.distance <= ?`;
    params.push(distanceCeiling);
  }
  let bm25 = `select null as id, 0.0 as score where 0`;
  if (match !== null) {
    bm25 = `select rowid as id, rank as score from notes_fts
            where notes_fts match ? order by rank, rowid limit ?`;
    params.push(match, OVERFETCH * n);
  }
  params.push(...readable.params);
  let ftsCutoff = "";
  if (bm25Ceiling !== undefined) {
    ftsCutoff = `and hits.score <= ?`;
    params.push(bm25Ceiling);
  }
  params.push(n);

  // Cutoffs, the supersede filter and the scope's read check sit in the WHERE,
  // so the rank each side contributes is a rank *among survivors* — fusion
  // never sees the rest, and a scoped agent gets up to N readable hits rather
  // than N hits with the forbidden ones cut out afterwards.
  return db
    .query(
      `with knn as materialized (${knn}),
         vecq as (
           select knn.id as id, row_number() over (order by knn.distance, knn.id) as r
           from knn join notes n on n.id = knn.id
           where n.superseded_by is null and (${readable.sql}) ${vecCutoff}
         ),
         hits as materialized (${bm25}),
         ftsq as (
           select hits.id as id, row_number() over (order by hits.score, hits.id) as r
           from hits join notes n on n.id = hits.id
           where n.superseded_by is null and (${readable.sql}) ${ftsCutoff}
         )
       select n.id as id, n.path as path, n.title as title,
              coalesce(1.0/(${RRF_K}+vecq.r), 0) + coalesce(1.0/(${RRF_K}+ftsq.r), 0) as score,
              vecq.r as vecRank, ftsq.r as ftsRank
       from vecq full outer join ftsq on vecq.id = ftsq.id
       join notes n on n.id = coalesce(vecq.id, ftsq.id)
       order by score desc, n.path
       limit ?`,
    )
    .all(...params) as Omit<SearchHit, "expansion">[];
}

type Neighbour = { id: number; path: string; title: string };

/**
 * Notes one wikilink away from a hit, in either direction, appended below
 * every direct hit and never mixed into the ranking — DESIGN.md § Retrieval:
 * an opt-in second pass, never an LLM graph walk. Capped at N of its own, so
 * `expandLinks` can return up to 2N.
 *
 * Its own enforcement point: a neighbour passes the same read check as a direct
 * hit, or a scoped agent would read forbidden titles one wikilink away.
 */
function expand(db: Database, hits: SearchHit[], n: number, scope: Scope): SearchHit[] {
  if (hits.length === 0) return [];
  const ids = hits.map((h) => h.id);
  const list = ids.map(() => "?").join(",");
  const readable = scope.readSql;
  const rows = db
    .query(
      `select n.id as id, n.path as path, n.title as title
       from notes n
       where n.superseded_by is null and (${readable.sql})
         and n.id not in (${list})
         and exists (
           select 1 from edges e
           where (e.from_id = n.id and e.to_id in (${list}))
              or (e.to_id = n.id and e.from_id in (${list}))
         )
       order by n.path
       limit ?`,
    )
    .all(...readable.params, ...ids, ...ids, ...ids, n) as Neighbour[];
  return rows.map((r) => ({ ...r, score: 0, vecRank: null, ftsRank: null, expansion: true }));
}

/**
 * The query vector, or null when there is no usable vector signal: nothing has
 * been embedded yet, or the query has no tokens this embedder knows. A zero
 * vector has no direction — cosine distance against it is NaN, which would
 * poison the KNN exactly as it would on the insert side.
 */
async function queryVector(
  db: Database,
  embedder: Embedder,
  query: string,
): Promise<Float32Array | null> {
  const hasVectors = db.query(`select 1 from sqlite_master where name = 'vectors'`).get() !== null;
  if (!hasVectors) return null;
  const [v] = await embedder.embed([query]);
  if (v === undefined || v.length !== embedder.dims) {
    throw new Error(
      `embedder ${embedder.model} returned a vector of width ${v?.length ?? 0} for the query, expected ${embedder.dims}`,
    );
  }
  // Normalized on query as on insert, so cosine distance is well-defined
  // whatever the provider returns.
  const q = l2normalize(v);
  return q.some((x) => x !== 0) ? q : null;
}

/** Vectors from another model are not comparable; doctor re-embeds them. */
function assertIndexedWith(db: Database, embedder: Embedder): void {
  const other = db
    .query(`select model, dims from vector_meta where model <> ? or dims <> ? limit 1`)
    .get(embedder.model, embedder.dims) as { model: string; dims: number } | null;
  if (other !== null) {
    throw new Error(
      `index was embedded with ${other.model}/${other.dims}, searching with ` +
        `${embedder.model}/${embedder.dims} — run vault doctor to re-embed`,
    );
  }
}

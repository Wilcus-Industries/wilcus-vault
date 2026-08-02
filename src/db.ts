// The index is derived and disposable: every row here is rebuildable from the
// .md files (`vault doctor --rebuild`). Nothing may treat it as authoritative.
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Embedder } from "./embed";

export const dbPath = (vaultRoot: string): string => join(vaultRoot, ".vault", "index.db");

/** Open (creating dirs and file) with sqlite-vec loaded and the schema applied. */
export function openDb(path: string): Database {
  mkdirSync(dirname(path), { recursive: true });
  const db = new Database(path, { create: true });
  sqliteVec.load(db);
  db.run("pragma journal_mode = wal"); // watcher, CLI and library callers share it
  db.run("pragma busy_timeout = 5000");
  migrate(db);
  return db;
}

function migrate(db: Database): void {
  // `slug` (filename stem) and `malformed` are denormalized off the parsed note
  // so link resolution and doctor's report are plain SQL.
  db.run(`create table if not exists notes (
    id integer primary key,
    path text not null unique,
    slug text not null,
    title text not null,
    type text,
    hash text not null,
    frontmatter text not null,
    superseded_by text,
    mtime integer not null,
    malformed integer not null default 0
  )`);
  db.run(`create index if not exists notes_slug on notes(slug)`);
  // `to_slug` is the link target as written — a bare filename stem or a
  // vault-relative path without `.md` (see `resolveEdges`); the column keeps
  // its name so an index built by an older version still opens. to_id null ⇒
  // the link is broken or ambiguous; backlinks and orphans are then trivial
  // SQL. Reindexing a note deletes its edges by from_id.
  db.run(`create table if not exists edges (
    from_id integer not null,
    to_slug text not null,
    to_id integer,
    unique(from_id, to_slug)
  )`);
  db.run(`create index if not exists edges_to_id on edges(to_id)`);
  db.run(`create virtual table if not exists notes_fts using fts5(title, body)`);
  db.run(`create table if not exists vector_meta (
    note_id integer primary key,
    model text not null,
    dims integer not null
  )`);
}

/**
 * dims reach the vec0 DDL by interpolation (vec0 cannot bind them), so an
 * injected embedder does not get to write SQL. Checked before either half of
 * the swap, so a nonsense embedder fails before a vault is embedded against it.
 */
function checkDims(embedder: Embedder): void {
  if (!Number.isInteger(embedder.dims) || embedder.dims < 1 || embedder.dims > 8192) {
    throw new Error(`embedder ${embedder.model}: dims must be an integer in 1..8192`);
  }
}

/**
 * Is every stored vector invalid — a different model, or a table of a different
 * width than this embedder returns? **Read-only, deliberately**: answering this
 * is what a caller does *before* embedding, and the answer is worthless if
 * asking for it has already thrown the old vectors away. See `resetVectors`.
 */
export function vectorsStale(db: Database, embedder: Embedder): boolean {
  checkDims(embedder);
  const staleModel =
    db
      .query(`select 1 from vector_meta where model <> ? or dims <> ? limit 1`)
      .get(embedder.model, embedder.dims) != null;
  const existing = db.query(`select sql from sqlite_master where name = 'vectors'`).get() as
    | { sql: string }
    | null;
  const currentDims = Number(existing?.sql.match(/float\[(\d+)\]/)?.[1] ?? 0);
  return staleModel || (existing != null && currentDims !== embedder.dims);
}

/**
 * Bring the vec0 table into line with `embedder`. `stale` (what `vectorsStale`
 * said) drops it and clears the meta — its rows belong to another vector space
 * — and is destructive, so call it with the replacement vectors already in
 * hand and inside the transaction that writes them: a re-embed that never
 * arrives must not leave a vault with no vectors and nothing saying so.
 * Otherwise the table is created only if it is missing; its dims are its DDL.
 */
export function resetVectors(db: Database, embedder: Embedder, stale: boolean): void {
  checkDims(embedder);
  if (stale) {
    db.run(`drop table if exists vectors`);
    db.run(`delete from vector_meta`);
  }
  db.run(`create virtual table if not exists vectors using vec0(
    note_id integer primary key,
    emb float[${embedder.dims}] distance_metric=cosine
  )`);
}

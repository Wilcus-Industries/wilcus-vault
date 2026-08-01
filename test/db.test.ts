import { test, expect, afterAll } from "bun:test";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { openDb, dbPath, ensureVectors } from "../src/db";
import { TokenOverlapEmbedder, l2normalize } from "../src/embed";
import { makeVault, cleanupVaults } from "./vault-fixture";

afterAll(cleanupVaults);

test("openDb creates <vault>/.vault/index.db in WAL with busy_timeout, sqlite-vec loaded", () => {
  const root = makeVault({});
  const db = openDb(dbPath(root));
  expect(existsSync(join(root, ".vault", "index.db"))).toBe(true);
  expect((db.query("pragma journal_mode").get() as { journal_mode: string }).journal_mode).toBe(
    "wal",
  );
  expect((db.query("pragma busy_timeout").get() as { timeout: number }).timeout).toBe(5000);
  expect((db.query("select vec_version() as v").get() as { v: string }).v).toStartWith("v0.1.7");
  db.close();
});

test("schema: notes, edges, notes_fts, vector_meta; edges are unique per (from_id, to_slug)", () => {
  const db = openDb(dbPath(makeVault({})));
  const names = (
    db.query("select name from sqlite_master where type in ('table','view')").all() as {
      name: string;
    }[]
  ).map((r) => r.name);
  expect(names).toContain("notes");
  expect(names).toContain("edges");
  expect(names).toContain("notes_fts");
  expect(names).toContain("vector_meta");
  expect(names).not.toContain("vectors"); // created lazily, from the embedder's dims

  db.run(
    "insert into notes (path, slug, title, hash, frontmatter, mtime) values ('a.md','a','A','h','{}',0)",
  );
  db.run("insert into edges (from_id, to_slug) values (1, 'b')");
  expect(() => db.run("insert into edges (from_id, to_slug) values (1, 'b')")).toThrow();
  db.close();
});

test("vectors table is created lazily with the embedder's dims and cosine distance", () => {
  const db = openDb(dbPath(makeVault({})));
  expect(ensureVectors(db, new TokenOverlapEmbedder(8))).toBe(false);
  const sql = (db.query("select sql from sqlite_master where name='vectors'").get() as {
    sql: string;
  }).sql;
  expect(sql).toContain("float[8]");
  expect(sql).toContain("distance_metric=cosine");
  db.close();
});

test("a model or dims change drops the vec0 table and its meta", () => {
  const db = openDb(dbPath(makeVault({})));
  ensureVectors(db, new TokenOverlapEmbedder(8));
  db.run("insert into vectors (note_id, emb) values (1, ?)", [new Float32Array(8).fill(0.25)]);
  db.run("insert into vector_meta (note_id, model, dims) values (1, 'token-overlap-v1', 8)");

  // same embedder: left alone
  expect(ensureVectors(db, new TokenOverlapEmbedder(8))).toBe(false);
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 1 });

  // different dims: dropped, recreated at the new width, meta cleared
  expect(ensureVectors(db, new TokenOverlapEmbedder(16))).toBe(true);
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 0 });
  expect(db.query("select count(*) as c from vector_meta").get()).toEqual({ c: 0 });
  expect(
    (db.query("select sql from sqlite_master where name='vectors'").get() as { sql: string }).sql,
  ).toContain("float[16]");
  db.close();
});

test("TokenOverlapEmbedder: deterministic, sized to dims, overlap-sensitive", async () => {
  const e = new TokenOverlapEmbedder(64);
  expect(e.dims).toBe(64);
  const [a, b, c] = await e.embed(["acme corp invoice", "acme corp invoice", "zebra habitat"]);
  expect(a).toHaveLength(64);
  expect(Array.from(a!)).toEqual(Array.from(b!));
  const cos = (x: Float32Array, y: Float32Array) =>
    l2normalize(x.slice()).reduce((s, v, i) => s + v * y[i]!, 0);
  expect(cos(a!, l2normalize(b!.slice()))).toBeCloseTo(1, 5);
  expect(cos(a!, l2normalize(c!.slice()))).toBeLessThan(0.5);
});

test("l2normalize scales to unit length and tolerates a zero vector", () => {
  const v = l2normalize(new Float32Array([3, 4]));
  expect(v[0]).toBeCloseTo(0.6, 6);
  expect(v[1]).toBeCloseTo(0.8, 6);
  expect(Array.from(l2normalize(new Float32Array([0, 0])))).toEqual([0, 0]);
});

// Bootstrap research: verify sqlite-vec loads under bun:sqlite, FTS5 works,
// RRF fusion across both is expressible in one SQL statement, Bun.YAML exists.
// The corpus is built so each side of the FULL OUTER JOIN has rows the other
// lacks (doc 2: vector-only; docs 3/4: FTS-only) — a LEFT JOIN would fail this.
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { test, expect } from "bun:test";

test("sqlite-vec KNN + FTS5 + RRF hybrid in one query", () => {
  const db = new Database(":memory:");
  sqliteVec.load(db);

  const { vec_version } = db
    .prepare("select vec_version() as vec_version")
    .get() as { vec_version: string };
  expect(vec_version).toBe("v0.1.7-alpha.2"); // pinned pre-1.0

  db.run(`create virtual table v using vec0(id integer primary key, emb float[4])`);
  db.run(`create virtual table f using fts5(id unindexed, body)`);

  const docs: [number, number[], string][] = [
    [1, [1, 0, 0, 0], "acme corp invoice INV-001"],
    [2, [0.95, 0.05, 0, 0], "globex vendor policy"],
    [3, [0.9, 0.1, 0, 0], "acme renewal decision meeting notes with extra detail words"],
    [4, [0, 0, 1, 0], "acme acme acme"],
  ];
  const iv = db.prepare("insert into v (id, emb) values (?, ?)");
  const it = db.prepare("insert into f (id, body) values (?, ?)");
  for (const [id, emb, body] of docs) {
    iv.run(id, new Float32Array(emb));
    it.run(id, body);
  }

  // RRF: k=60, fuse vector KNN ranks with BM25 ranks.
  // KNN top-2 = {1, 2}; FTS "acme" = {1, 3, 4} — asymmetric by construction.
  const rows = db
    .prepare(
      `with vecq as (
         select id, row_number() over (order by distance) as r
         from v where emb match ? and k = 2
       ),
       ftsq as (
         select id, row_number() over (order by rank) as r
         from f where f match ? limit 3
       )
       select coalesce(vecq.id, ftsq.id) as id,
              coalesce(1.0/(60+vecq.r),0) + coalesce(1.0/(60+ftsq.r),0) as score
       from vecq full outer join ftsq on vecq.id = ftsq.id
       order by score desc`,
    )
    .all(new Float32Array([1, 0, 0, 0]), "acme") as { id: number; score: number }[];

  const ids = rows.map((x) => x.id);
  expect(ids.length).toBe(4); // union of both sides
  // Doc 1 is the only doc on BOTH signals: worst-case 1/61+1/63 ≈ .0323 beats
  // any single-signal doc's best case 1/61 ≈ .0164 — a strict, tie-free winner.
  expect(rows[0]!.id).toBe(1);
  expect(rows[0]!.score).toBeGreaterThan(rows[1]!.score * 1.5);
  expect(ids).toContain(2); // vector-only side survived the outer join
  expect(ids).toContain(4); // FTS-only side survived the outer join

  db.close();
});

test("Bun.YAML parses frontmatter", () => {
  const fm = Bun.YAML.parse("type: customer\ntags: [a, b]\nsuperseded_by: null");
  expect(fm).toEqual({ type: "customer", tags: ["a", "b"], superseded_by: null });
});

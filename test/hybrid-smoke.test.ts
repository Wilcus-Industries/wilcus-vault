// Bootstrap research: verify sqlite-vec loads under bun:sqlite, FTS5 works,
// RRF fusion across both is expressible in one SQL statement, Bun.YAML exists.
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { test, expect } from "bun:test";

test("sqlite-vec KNN + FTS5 + RRF hybrid in one query", () => {
  const db = new Database(":memory:");
  sqliteVec.load(db);

  const { vec_version } = db
    .prepare("select vec_version() as vec_version")
    .get() as { vec_version: string };
  console.log("sqlite-vec", vec_version);

  db.exec(`create virtual table v using vec0(id integer primary key, emb float[4])`);
  db.exec(`create virtual table f using fts5(id unindexed, body)`);

  const docs: [number, number[], string][] = [
    [1, [1, 0, 0, 0], "acme corp invoice INV-001"],
    [2, [0, 1, 0, 0], "globex vendor policy"],
    [3, [0.9, 0.1, 0, 0], "acme renewal decision"],
  ];
  const iv = db.prepare("insert into v (id, emb) values (?, ?)");
  const it = db.prepare("insert into f (id, body) values (?, ?)");
  for (const [id, emb, body] of docs) {
    iv.run(id, new Float32Array(emb));
    it.run(id, body);
  }

  // RRF: k=60, fuse vector KNN ranks with BM25 ranks.
  const rows = db
    .prepare(
      `with vecq as (
         select id, row_number() over (order by distance) as rank
         from v where emb match ? and k = 3
       ),
       ftsq as (
         select id, row_number() over (order by rank) as rank
         from f where f match ?
       )
       select coalesce(vecq.id, ftsq.id) as id,
              coalesce(1.0/(60+vecq.rank),0) + coalesce(1.0/(60+ftsq.rank),0) as score
       from vecq full outer join ftsq on vecq.id = ftsq.id
       order by score desc`,
    )
    .all(new Float32Array([1, 0, 0, 0]), "acme") as { id: number; score: number }[];

  console.log(rows);
  expect(rows[0]!.id).toBe(1); // top on both signals
  expect(rows.length).toBe(3);
});

test("Bun.YAML parses frontmatter", () => {
  const fm = Bun.YAML.parse("type: customer\ntags: [a, b]\nsuperseded_by: null");
  expect(fm).toEqual({ type: "customer", tags: ["a", "b"], superseded_by: null });
});

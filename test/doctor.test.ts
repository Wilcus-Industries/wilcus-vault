import { test, expect, afterAll } from "bun:test";
import { readdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { openDb, dbPath } from "../src/db";
import { TokenOverlapEmbedder } from "../src/embed";
import { reindex } from "../src/indexer";
import { doctor } from "../src/doctor";
import { open as openVault } from "../src/vault";
import { makeVault, writeNote, cleanupVaults, stubEmbedder } from "./vault-fixture";

afterAll(cleanupVaults);

const embedder = new TokenOverlapEmbedder(32);

const GRAPH = {
  "notes/acme.md": "# Acme\n\n[[globex]], [[ghost]], [[dup]]\n",
  "notes/globex.md": "# Globex\n\nno outgoing links\n",
  "notes/lonely.md": "# Lonely\n\nnothing here\n",
  "one/dup.md": "# Dup one\n",
  "two/dup.md": "# Dup two\n",
};

/** notes + edges as comparable, id-free tuples */
function snapshot(root: string): { notes: unknown[]; edges: unknown[] } {
  const db = openDb(dbPath(root));
  const notes = db.query("select path, slug, title, hash, malformed from notes order by path").all();
  const edges = db
    .query(
      `select f.path as from_path, e.to_slug, t.path as to_path
       from edges e join notes f on f.id = e.from_id
       left join notes t on t.id = e.to_id
       order by f.path, e.to_slug`,
    )
    .all();
  db.close();
  return { notes, edges };
}

test("doctor separates broken links from ambiguous ones and names the candidates", async () => {
  const root = makeVault(GRAPH);
  const report = await doctor(root, embedder);
  // 0 candidates is a typo or a deleted note; 2+ is a link that needs
  // qualifying, and the report says what to qualify it with
  expect(report.brokenLinks).toEqual([{ from: "notes/acme.md", slug: "ghost" }]);
  expect(report.ambiguousLinks).toEqual([
    { from: "notes/acme.md", slug: "dup", candidates: ["one/dup.md", "two/dup.md"] },
  ]);
  expect(report.orphans).toEqual(["notes/lonely.md", "one/dup.md", "two/dup.md"]);
  expect(report.malformed).toEqual([]);
});

test("a qualified link is broken, never ambiguous, and duplicate stems alone are fine", async () => {
  const root = makeVault({
    // namespaced notes may share a stem — that is what namespaces are for. Only
    // a *bare* link to them is a problem, and qualifying it is the fix.
    "notes/acme.md": "# Acme\n\n[[one/dup]] and [[two/dup]]\n",
    "notes/gone.md": "# Gone\n\n[[one/ghost]]\n",
    "one/dup.md": "# Dup one\n",
    "two/dup.md": "# Dup two\n",
  });
  const report = await doctor(root, embedder);
  expect(report.ambiguousLinks).toEqual([]);
  expect(report.brokenLinks).toEqual([{ from: "notes/gone.md", slug: "one/ghost" }]);
});

test("doctor reports stale rows and missing files, then repairs them", async () => {
  const root = makeVault(GRAPH);
  const db = openDb(dbPath(root));
  await reindex(db, root, embedder);
  db.close();

  writeNote(root, "notes/globex.md", "# Globex\n\nedited by a human\n");
  writeNote(root, "notes/new.md", "# New\n\n[[globex]]\n");
  rmSync(join(root, "notes", "lonely.md"));

  const found = await doctor(root, embedder, { repair: false });
  expect(found.stale.sort()).toEqual(["notes/globex.md", "notes/new.md"]);
  expect(found.missing).toEqual(["notes/lonely.md"]);

  const repaired = await doctor(root, embedder);
  expect(repaired.stale.sort()).toEqual(["notes/globex.md", "notes/new.md"]); // what it fixed
  expect(await doctor(root, embedder, { repair: false })).toMatchObject({ stale: [], missing: [] });

  const after = openDb(dbPath(root));
  expect(after.query("select count(*) as c from notes where path='notes/lonely.md'").get()).toEqual(
    { c: 0 },
  );
  expect(after.query("select count(*) as c from notes").get()).toEqual({ c: 5 });
  after.close();
});

test("doctor reports malformed frontmatter", async () => {
  const root = makeVault({ "bad.md": "---\ntype: [unclosed\n---\n# Bad\n" });
  expect((await doctor(root, embedder)).malformed).toEqual(["bad.md"]);
});

test("--rebuild reproduces an identical index after the DB is deleted", async () => {
  const root = makeVault(GRAPH);
  await doctor(root, embedder);
  const before = snapshot(root);

  rmSync(dbPath(root));
  const report = await doctor(root, embedder, { rebuild: true });
  expect(report.missing).toEqual([]);
  expect(snapshot(root)).toEqual(before);

  // rebuilding over a live index leaves no temp files behind
  await doctor(root, embedder, { rebuild: true });
  expect(snapshot(root)).toEqual(before);
  expect(readdirSync(join(root, ".vault"))).toEqual(["index.db"]);
});

test("--rebuild re-embeds by definition and reports it", async () => {
  const root = makeVault(GRAPH);
  expect((await doctor(root, embedder, { rebuild: true })).reembedded).toBe(true);
});

test("a failed rebuild leaves no temp database behind", async () => {
  const root = makeVault(GRAPH);
  await doctor(root, embedder);
  const boom = stubEmbedder("boom-v1", 32, async () => {
    throw new Error("provider down");
  });
  await expect(doctor(root, boom, { rebuild: true })).rejects.toThrow("provider down");
  expect(readdirSync(join(root, ".vault"))).toEqual(["index.db"]);
  expect(snapshot(root).notes).toHaveLength(5); // the live index is untouched
});

test("doctor survives a file deleted mid-run", async () => {
  const root = makeVault(GRAPH);
  const racy = stubEmbedder("race-v1", 32, async (texts) => {
    rmSync(join(root, "notes", "lonely.md"), { force: true });
    return texts.map(() => new Float32Array(32).fill(1));
  });
  await doctor(root, racy); // must not throw
  expect((await doctor(root, racy)).missing).toEqual(["notes/lonely.md"]);
  expect((await doctor(root, racy, { repair: false })).missing).toEqual([]);
});

test("a model or dims change drops the vec0 table and re-embeds", async () => {
  const root = makeVault(GRAPH);
  await doctor(root, embedder);
  const report = await doctor(root, new TokenOverlapEmbedder(64));
  expect(report.reembedded).toBe(true);

  const db = openDb(dbPath(root));
  expect(db.query("select count(*) as c from vector_meta where dims = 64").get()).toEqual({ c: 5 });
  expect(
    (db.query("select sql from sqlite_master where name='vectors'").get() as { sql: string }).sql,
  ).toContain("float[64]");
  db.close();
});

test("reopening the vault with another embedder: doctor re-embeds and search works again", async () => {
  const root = makeVault(GRAPH);
  const before = openVault(root, { embedder });
  await before.reindex();
  expect((await before.search("globex vendor")).map((h) => h.path)).toContain("notes/globex.md");
  before.close();

  // a different model *and* width: the vec0 table's dims live in its DDL
  const swapped = stubEmbedder("swapped-v1", 64, (texts) => new TokenOverlapEmbedder(64).embed(texts));
  const after = openVault(root, { embedder: swapped });
  // vectors from another model are not comparable, so search refuses until doctor runs
  await expect(after.search("globex")).rejects.toThrow(/run vault doctor/);

  expect((await after.doctor()).reembedded).toBe(true);
  const db = openDb(dbPath(root));
  expect(
    (db.query("select sql from sqlite_master where name='vectors'").get() as { sql: string }).sql,
  ).toContain("float[64]");
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 5 });
  expect(
    db.query("select count(*) as c from vector_meta where model='swapped-v1' and dims=64").get(),
  ).toEqual({ c: 5 });
  expect(db.query("select count(*) as c from vector_meta where dims<>64").get()).toEqual({ c: 0 });
  db.close();

  // and the vault is usable again through the new embedder
  expect((await after.search("globex vendor")).map((h) => h.path)).toContain("notes/globex.md");
  after.close();
});


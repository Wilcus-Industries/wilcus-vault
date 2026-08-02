import { test, expect, afterAll } from "bun:test";
import type { Database } from "bun:sqlite";
import { mkdirSync, rmSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import { openDb, dbPath } from "../src/db";
import { TokenOverlapEmbedder } from "../src/embed";
import { reindex, indexPaths, scanVault, readNote } from "../src/indexer";
import { doctor } from "../src/doctor";
import { makeVault, writeNote, cleanupVaults, stubEmbedder, vecOf } from "./vault-fixture";

afterAll(cleanupVaults);

const embedder = new TokenOverlapEmbedder(32);

const FIXTURE = {
  "notes/acme.md": "---\ntype: customer\n---\n# Acme Corp\n\nrenewal, see [[globex]] and [[ghost]]\n",
  "notes/globex.md": "# Globex\n\nvendor policy\n",
  "notes/lonely.md": "# Lonely\n\nno links at all\n",
  ".obsidian/skip.md": "# Skipped\n\n[[acme]]\n",
  "notes/not-markdown.txt": "ignored",
};

function open(root: string): Database {
  return openDb(dbPath(root));
}

const changes = (db: Database) =>
  (db.query("select total_changes() as c").get() as { c: number }).c;

test("scanVault: .md only, dot-directories skipped, symlinks never followed", () => {
  const root = makeVault(FIXTURE);
  const other = makeVault({ "secret.md": "# Secret\n" });
  symlinkSync(other, join(root, "linked-dir"));
  symlinkSync(join(other, "secret.md"), join(root, "linked.md"));

  expect(scanVault(root)).toEqual(["notes/acme.md", "notes/globex.md", "notes/lonely.md"]);
});

test("reindex writes notes, fts rows, L2-normalized vectors and resolved edges", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  const stats = await reindex(db, root, embedder);
  expect(stats).toMatchObject({ added: 3, updated: 0, removed: 0, unchanged: 0 });

  const acme = db.query("select * from notes where path = 'notes/acme.md'").get() as {
    id: number;
    slug: string;
    title: string;
    type: string | null;
    hash: string;
    frontmatter: string;
    mtime: number;
    malformed: number;
  };
  expect(acme.slug).toBe("acme");
  expect(acme.title).toBe("Acme Corp");
  expect(acme.type).toBe("customer");
  expect(JSON.parse(acme.frontmatter)).toEqual({ type: "customer" });
  expect(acme.mtime).toBeGreaterThan(0);
  expect(acme.malformed).toBe(0);

  // FTS row is keyed by note id and searchable
  expect(
    db.query("select rowid from notes_fts where notes_fts match 'renewal'").get(),
  ).toEqual({ rowid: acme.id });

  // vector: one per note, unit length, meta records model + dims
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 3 });
  const v = vecOf(db, acme.id);
  expect(v).toHaveLength(32);
  expect(Math.hypot(...v)).toBeCloseTo(1, 5);
  expect(db.query("select model, dims from vector_meta where note_id = ?").get(acme.id)).toEqual({
    model: embedder.model,
    dims: 32,
  });

  // edges: unique stem resolves, unknown stem stays null
  const edges = db
    .query("select to_slug, to_id from edges where from_id = ? order by to_slug")
    .all(acme.id) as { to_slug: string; to_id: number | null }[];
  const globexId = (db.query("select id from notes where path='notes/globex.md'").get() as {
    id: number;
  }).id;
  expect(edges).toEqual([
    { to_slug: "ghost", to_id: null },
    { to_slug: "globex", to_id: globexId },
  ]);
  db.close();
});

test("a pass with nothing changed writes nothing", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const before = changes(db);
  // the watcher's entry point: an editor saving identical bytes must not
  // dirty the database
  const stats = await indexPaths(db, root, embedder, scanVault(root));
  expect(stats).toEqual({ added: 0, updated: 0, removed: 0, unchanged: 3, reembedded: false });
  expect(changes(db)).toBe(before);
  db.close();
});

test("reindex re-resolves every edge, even when no file changed", async () => {
  // An index written under an older resolution rule holds `to_id` values that
  // rule produced, and nothing on disk has to change for them to be wrong:
  // `[[one/dup]]` was unresolvable before this rule and is exact under it. The
  // whole-vault pass is the only place that can notice.
  const root = makeVault({ "hub.md": "# Hub\n\n[[one/dup]]\n", "one/dup.md": "# Dup one\n" });
  const db = open(root);
  await reindex(db, root, embedder);
  db.run("update edges set to_id = null"); // an index built before the rule
  expect(await reindex(db, root, embedder)).toMatchObject({ added: 0, updated: 0, unchanged: 2 });
  expect(
    db.query("select n.path from edges e join notes n on n.id = e.to_id").get(),
  ).toEqual({ path: "one/dup.md" });
  db.close();
});

test("edits update in place; deleted files purge every table", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const id = (db.query("select id from notes where path='notes/acme.md'").get() as { id: number })
    .id;

  writeNote(root, "notes/acme.md", "# Acme Corp\n\nnow points at [[lonely]] only\n");
  const stats = await reindex(db, root, embedder);
  expect(stats).toMatchObject({ added: 0, updated: 1, unchanged: 2 });
  expect(db.query("select id from notes where path='notes/acme.md'").get()).toEqual({ id }); // stable identity
  expect(
    (db.query("select to_slug from edges where from_id = ?").all(id) as { to_slug: string }[]).map(
      (e) => e.to_slug,
    ),
  ).toEqual(["lonely"]);

  rmSync(join(root, "notes", "acme.md"));
  expect(await reindex(db, root, embedder)).toMatchObject({ removed: 1, unchanged: 2 });
  expect(db.query("select count(*) as c from notes").get()).toEqual({ c: 2 });
  expect(db.query("select count(*) as c from edges where from_id = ?").get(id)).toEqual({ c: 0 });
  expect(db.query("select count(*) as c from vectors where note_id = ?").get(id)).toEqual({ c: 0 });
  expect(db.query("select count(*) as c from vector_meta where note_id = ?").get(id)).toEqual({
    c: 0,
  });
  expect(db.query("select count(*) as c from notes_fts where rowid = ?").get(id)).toEqual({ c: 0 });
  db.close();
});

test("bare-stem resolution: zero or several stem matches leave to_id null", async () => {
  const root = makeVault({
    "a.md": "# A\n\n[[dup]] and [[nowhere]]\n",
    "one/dup.md": "# Dup one\n",
    "two/dup.md": "# Dup two\n",
  });
  const db = open(root);
  await reindex(db, root, embedder);
  expect(db.query("select count(*) as c from edges where to_id is not null").get()).toEqual({
    c: 0,
  });

  // removing the duplicate makes the link resolvable on the next pass, even
  // though the linking note itself never changed
  rmSync(join(root, "two", "dup.md"));
  await reindex(db, root, embedder);
  const resolved = db.query(
    "select n.path from edges e join notes n on n.id = e.to_id where e.to_slug='dup'",
  ).get();
  expect(resolved).toEqual({ path: "one/dup.md" });
  expect(db.query("select to_id from edges where to_slug='nowhere'").get()).toEqual({ to_id: null });
  db.close();
});

test("path-qualified links resolve by exact path, past a duplicated stem", async () => {
  const root = makeVault({
    "hub.md":
      "# Hub\n\n[[customers/acme]], [[vendors/acme]], [[acme]], [[customers/ghost]], [[../outside/secret]]\n",
    "customers/acme.md": "# Acme the customer\n",
    "vendors/acme.md": "# Acme the vendor\n",
  });
  const db = open(root);
  await reindex(db, root, embedder);
  const idOf = (path: string) =>
    (db.query("select id from notes where path = ?").get(path) as { id: number }).id;
  const hub = idOf("hub.md");

  expect(
    db.query("select to_slug, to_id from edges where from_id = ? order by to_slug").all(hub),
  ).toEqual([
    // never path-joined to the filesystem: resolution is SQL against known
    // note paths, so a traversing target simply matches nothing
    { to_slug: "../outside/secret", to_id: null },
    { to_slug: "acme", to_id: null }, // bare stem, two candidates ⇒ unresolved
    { to_slug: "customers/acme", to_id: idOf("customers/acme.md") },
    { to_slug: "customers/ghost", to_id: null }, // qualified, but no such note
    { to_slug: "vendors/acme", to_id: idOf("vendors/acme.md") },
  ]);

  // the bare stem resolves again once only one note carries it, and the
  // qualified links are untouched by that
  rmSync(join(root, "vendors", "acme.md"));
  await reindex(db, root, embedder);
  expect(
    db.query("select to_id from edges where from_id = ? and to_slug = 'acme'").get(hub),
  ).toEqual({ to_id: idOf("customers/acme.md") });
  expect(
    db.query("select to_id from edges where from_id = ? and to_slug = 'vendors/acme'").get(hub),
  ).toEqual({ to_id: null });
  db.close();
});

test("a path-qualified link is a path, not a stem: no extension, no near-miss", async () => {
  const root = makeVault({
    "hub.md": "# Hub\n\n[[customers/acme.md]] [[/customers/acme]] [[./customers/acme]] [[acme]]\n",
    "customers/acme.md": "# Acme\n",
  });
  const db = open(root);
  await reindex(db, root, embedder);
  // only the bare stem resolves here: the qualified forms are matched against
  // `notes.path` minus its `.md`, exactly, with no normalization
  expect(
    db.query("select to_slug from edges where to_id is not null").all(),
  ).toEqual([{ to_slug: "acme" }]);
  db.close();
});

test("malformed frontmatter still indexes and is flagged", async () => {
  const root = makeVault({ "bad.md": "---\ntype: [unclosed\n---\n# Bad\n\nbody\n" });
  const db = open(root);
  await reindex(db, root, embedder);
  expect(db.query("select title, malformed from notes where path='bad.md'").get()).toEqual({
    title: "Bad",
    malformed: 1,
  });
  db.close();
});

test("an embedding model change re-embeds every note", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const wider = new TokenOverlapEmbedder(48);
  const stats = await reindex(db, root, wider);
  expect(stats).toMatchObject({ updated: 3, unchanged: 0, reembedded: true });
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 3 });
  expect(db.query("select count(*) as c from vector_meta where dims = 48").get()).toEqual({ c: 3 });
  db.close();
});

test("a new embedding model at the same dims still re-embeds everything", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const stats = await reindex(db, root, stubEmbedder("other-model-v1", 32));
  expect(stats).toMatchObject({ updated: 3, unchanged: 0, reembedded: true });
  expect(db.query("select count(*) as c from vector_meta where model='other-model-v1'").get()).toEqual(
    { c: 3 },
  );
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 3 });
  db.close();
});

test("a note with no tokens gets no vector row, and stays indexed and idempotent", async () => {
  const root = makeVault({ "cjk.md": "# 圏点\n\n日本語 🙂\n", "notes/acme.md": FIXTURE["notes/acme.md"] });
  const db = open(root);
  expect(await reindex(db, root, embedder)).toMatchObject({ added: 2 });
  const id = (db.query("select id from notes where path='cjk.md'").get() as { id: number }).id;
  // an all-zero vector has no direction: cosine distance against it is NaN
  expect(db.query("select count(*) as c from vectors where note_id = ?").get(id)).toEqual({ c: 0 });
  expect(db.query("select count(*) as c from vector_meta where note_id = ?").get(id)).toEqual({
    c: 1,
  });
  expect(db.query("select count(*) as c from notes_fts where rowid = ?").get(id)).toEqual({ c: 1 });
  // must not look half-indexed on the next pass (measured on `indexPaths`:
  // `reindex` re-resolves edges on every run by design)
  const before = changes(db);
  expect(await indexPaths(db, root, embedder, scanVault(root))).toMatchObject({
    unchanged: 2,
    updated: 0,
  });
  expect(changes(db)).toBe(before);
  db.close();
});

test("a note replaced by a symlink is purged, never read from outside the vault", async () => {
  const root = makeVault(FIXTURE);
  const outside = makeVault({ "secret.md": "# Secret\n\nnot in this vault at all\n" });
  const db = open(root);
  await reindex(db, root, embedder);

  // the scan skips symlinks, so without a check the row would keep its content
  // from *outside* the vault forever: doctor reports it missing and repair,
  // reading straight through the link, never removes it
  rmSync(join(root, "notes", "lonely.md"));
  symlinkSync(join(outside, "secret.md"), join(root, "notes", "lonely.md"));

  expect(await reindex(db, root, embedder)).toMatchObject({ removed: 1, added: 0, updated: 0 });
  expect(db.query("select count(*) as c from notes where path='notes/lonely.md'").get()).toEqual({
    c: 0,
  });
  expect(db.query("select rowid from notes_fts where notes_fts match 'secret'").get()).toBeNull();
  db.close();
});

test("a note replaced by a directory is purged instead of throwing", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);

  rmSync(join(root, "notes", "lonely.md"));
  mkdirSync(join(root, "notes", "lonely.md")); // `mv note.md dir/` leaves this behind
  expect(await reindex(db, root, embedder)).toMatchObject({ removed: 1 });
  expect(db.query("select count(*) as c from notes where path='notes/lonely.md'").get()).toEqual({
    c: 0,
  });
  db.close();

  // and doctor agrees rather than failing on EISDIR with no way to repair
  expect(await doctor(root, embedder)).toMatchObject({ missing: [] });
});

test("readNote returns null when the file vanished between scan and read", async () => {
  expect(await readNote(makeVault(FIXTURE), "notes/ghost.md")).toBeNull();
});

test("a file deleted mid-run does not abort the reindex", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  const racy = stubEmbedder("race-v1", 32, async (texts) => {
    rmSync(join(root, "notes", "globex.md")); // gone after we read it, before we write
    return texts.map(() => new Float32Array(32).fill(1));
  });
  await reindex(db, root, racy); // must not throw
  expect(await reindex(db, root, racy)).toMatchObject({ removed: 1, unchanged: 2 });
  db.close();
});

test("an embedder returning the wrong shape fails with a clear error", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  const narrow = stubEmbedder("narrow-v1", 32, async (texts) =>
    texts.map(() => new Float32Array(8)),
  );
  await expect(reindex(db, root, narrow)).rejects.toThrow(/narrow-v1.*width 8.*32/s);
  const short = stubEmbedder("short-v1", 32, async () => []);
  await expect(reindex(db, root, short)).rejects.toThrow(/short-v1.*0 vectors for 3/s);
  db.close();
});

test("an empty vault indexes to an empty database", async () => {
  const root = makeVault({});
  mkdirSync(join(root, "empty-dir"));
  const db = open(root);
  expect(await reindex(db, root, embedder)).toMatchObject({ added: 0, removed: 0 });
  expect(db.query("select count(*) as c from notes").get()).toEqual({ c: 0 });
  db.close();
});

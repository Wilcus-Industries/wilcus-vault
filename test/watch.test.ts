import { test, expect, afterAll } from "bun:test";
import type { Database } from "bun:sqlite";
import { rmSync } from "node:fs";
import { join } from "node:path";
import { openDb, dbPath } from "../src/db";
import { TokenOverlapEmbedder } from "../src/embed";
import { reindex, isNotePath } from "../src/indexer";
import { watch } from "../src/watch";
import { open as openVault } from "../src/vault";
import { makeVault, writeNote, cleanupVaults, stubEmbedder, vecOf } from "./vault-fixture";

afterAll(cleanupVaults);

const embedder = new TokenOverlapEmbedder(32);

const FIXTURE = {
  "notes/acme.md": "# Acme Corp\n\nrenewal, see [[globex]]\n",
  "notes/globex.md": "# Globex\n\nvendor policy\n",
};

const open = (root: string): Database => openDb(dbPath(root));

const noteRow = (db: Database, path: string): { id: number; hash: string } | null =>
  db.query("select id, hash from notes where path = ?").get(path) as {
    id: number;
    hash: string;
  } | null;

/**
 * Poll until `fn` returns something truthy. Real fs events have no promise to
 * await, so the end-to-end test waits for the *effect* — the index row — with a
 * deadline generous enough that a loaded machine is not a failing one.
 */
async function until<T>(
  what: string,
  fn: () => T | null | undefined | false,
  timeoutMs = 8000,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = fn();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await Bun.sleep(10);
  }
}

test("isNotePath: .md files only, never through a dot-directory", () => {
  expect(isNotePath("notes/acme.md")).toBe(true);
  expect(isNotePath("acme.md")).toBe(true);
  // the index itself lives in .vault/ — the watcher must not feed itself
  expect(isNotePath(".vault/index.db")).toBe(false);
  expect(isNotePath(".obsidian/workspace.md")).toBe(false);
  expect(isNotePath(".git/COMMIT_EDITMSG.md")).toBe(false);
  expect(isNotePath(".hidden.md")).toBe(false);
  expect(isNotePath("notes/photo.png")).toBe(false);
  expect(isNotePath("notes")).toBe(false);
});

// The edits below land *before* `watch` starts, so the only events in these
// two tests are the `touch` calls themselves — a live fs.watch on the same root
// would otherwise race its own inotify events into the flush sequence and make
// an exact assertion about passes flaky.
test("a burst on one path is debounced into a single index pass", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  writeNote(root, "notes/acme.md", "# Acme Corp\n\nedited five times\n");

  const flushes: string[][] = [];
  const w = watch(db, root, embedder, { debounceMs: 20, onChange: (paths) => flushes.push(paths) });
  for (let i = 0; i < 5; i++) w.touch("notes/acme.md");
  await w.idle();

  expect(flushes).toEqual([["notes/acme.md"]]);
  expect(noteRow(db, "notes/acme.md")!.hash).toBe(
    new Bun.CryptoHasher("sha256").update("# Acme Corp\n\nedited five times\n").digest("hex"),
  );
  await w.close();
  db.close();
});

test("paths ready together are indexed in one pass", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  writeNote(root, "notes/acme.md", "# Acme Corp\n\none\n");
  writeNote(root, "notes/globex.md", "# Globex\n\ntwo\n");

  const flushes: string[][] = [];
  const w = watch(db, root, embedder, { debounceMs: 10, onChange: (paths) => flushes.push(paths) });
  w.touch("notes/acme.md");
  w.touch("notes/globex.md");
  await w.idle();

  expect(flushes).toEqual([["notes/acme.md", "notes/globex.md"]]);
  await w.close();
  db.close();
});

test("touch ignores anything the scan would not index", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);

  const flushes: string[][] = [];
  const w = watch(db, root, embedder, { debounceMs: 1, onChange: (paths) => flushes.push(paths) });
  // the index's own files live under .vault/: acting on them would feed the
  // watcher its own tail
  w.touch(".vault/index.db");
  w.touch(".vault/index.db-wal");
  w.touch(".obsidian/workspace.md");
  w.touch("notes/photo.png");
  await w.idle();

  expect(flushes).toEqual([]);
  await w.close();
  db.close();
});

test("the watcher indexes a create, an edit and a delete", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const w = watch(db, root, embedder, { debounceMs: 5 });

  // create: row, vector and a wikilink edge that resolves
  writeNote(root, "notes/new.md", "# New\n\npoints at [[acme]]\n");
  w.touch("notes/new.md");
  await w.idle();
  const created = noteRow(db, "notes/new.md")!;
  expect(created).not.toBeNull();
  expect(vecOf(db, created.id)).toHaveLength(32);
  expect(
    db.query("select to_id from edges where from_id = ? and to_slug = 'acme'").get(created.id),
  ).toEqual({ to_id: noteRow(db, "notes/acme.md")!.id });

  // edit: hash and vector both move
  const before = vecOf(db, created.id);
  writeNote(root, "notes/new.md", "# New\n\nrewritten with entirely different words\n");
  w.touch("notes/new.md");
  await w.idle();
  expect(noteRow(db, "notes/new.md")!.hash).not.toBe(created.hash);
  expect([...vecOf(db, created.id)]).not.toEqual([...before]);

  // delete: every derived row goes with the file
  rmSync(join(root, "notes", "new.md"));
  w.touch("notes/new.md");
  await w.idle();
  expect(noteRow(db, "notes/new.md")).toBeNull();
  expect(db.query("select count(*) as c from vectors where note_id = ?").get(created.id)).toEqual({
    c: 0,
  });
  expect(db.query("select count(*) as c from notes_fts where rowid = ?").get(created.id)).toEqual({
    c: 0,
  });
  await w.close();
  db.close();
});

test("an unchanged file is reported but rewrites nothing", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const changes = () => (db.query("select total_changes() as c").get() as { c: number }).c;

  const seen: { added: number; unchanged: number }[] = [];
  const w = watch(db, root, embedder, { debounceMs: 1, onChange: (_p, s) => seen.push(s) });
  const before = changes();
  w.touch("notes/acme.md"); // editor rewrote identical bytes
  await w.idle();
  expect(seen).toMatchObject([{ added: 0, unchanged: 1 }]);
  expect(changes()).toBe(before);
  await w.close();
  db.close();
});

test("an embedder failure is reported and the watcher keeps working", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  let down = false;
  const flaky = stubEmbedder("flaky-v1", 32, async (texts) => {
    if (down) throw new Error("provider down");
    return embedder.embed(texts);
  });
  await reindex(db, root, flaky);

  const stale = noteRow(db, "notes/acme.md")!.hash;
  const errors: unknown[] = [];
  const w = watch(db, root, flaky, { debounceMs: 1, onError: (e) => errors.push(e) });
  down = true;
  writeNote(root, "notes/acme.md", "# Acme Corp\n\nedited while the provider is down\n");
  w.touch("notes/acme.md");
  await w.idle();
  expect(String(errors[0])).toContain("provider down");
  // the pass failed, and it failed without taking the watcher down with it
  expect(noteRow(db, "notes/acme.md")!.hash).toBe(stale);

  down = false;
  w.touch("notes/acme.md");
  await w.idle();
  expect(noteRow(db, "notes/acme.md")!.hash).toBe(
    new Bun.CryptoHasher("sha256")
      .update("# Acme Corp\n\nedited while the provider is down\n")
      .digest("hex"),
  );
  await w.close();
  db.close();
});

test("a reporter that throws does not wedge the watcher", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  let down = false;
  const flaky = stubEmbedder("flaky-v1", 32, async (texts) => {
    if (down) throw new Error("provider down");
    return embedder.embed(texts);
  });
  await reindex(db, root, flaky);
  writeNote(root, "notes/acme.md", "# Acme Corp\n\nfails to embed\n");

  let reports = 0;
  const w = watch(db, root, flaky, {
    debounceMs: 1,
    onError: () => {
      reports++;
      throw new Error("the error reporter is broken too");
    },
  });
  down = true;
  w.touch("notes/acme.md");
  await w.idle(); // must resolve: a throwing reporter used to leave the pass in flight forever
  expect(reports).toBe(1);

  // and the next change still gets indexed
  down = false;
  writeNote(root, "notes/globex.md", "# Globex\n\nindexes fine\n");
  w.touch("notes/globex.md");
  await w.idle();
  expect(noteRow(db, "notes/globex.md")!.hash).toBe(
    new Bun.CryptoHasher("sha256").update("# Globex\n\nindexes fine\n").digest("hex"),
  );
  await w.close();
  db.close();
});

test("close waits for the pass in flight and starts no more", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const changes = () => (db.query("select total_changes() as c").get() as { c: number }).c;

  let release = (): void => {};
  const held = new Promise<void>((r) => (release = r));
  const slow = stubEmbedder("token-overlap-v1", 32, async (texts) => {
    await held;
    return embedder.embed(texts);
  });
  writeNote(root, "notes/acme.md", "# Acme Corp\n\nindexed by the pass in flight\n");
  writeNote(root, "notes/globex.md", "# Globex\n\nqueued behind it, never indexed\n");

  const w = watch(db, root, slow, { debounceMs: 1 });
  w.touch("notes/acme.md");
  await Bun.sleep(20); // the pass is now inside embed()
  w.touch("notes/globex.md"); // queues behind it
  await Bun.sleep(20);

  const closing = w.close();
  release();
  await closing;
  const settled = changes();

  // closing resolves only once the database is nobody's: a caller may now close
  // it (the README example does exactly that) without racing a write
  await Bun.sleep(50);
  expect(changes()).toBe(settled);
  expect(noteRow(db, "notes/acme.md")!.hash).toBe(
    new Bun.CryptoHasher("sha256")
      .update("# Acme Corp\n\nindexed by the pass in flight\n")
      .digest("hex"),
  );
  // the queued path never ran: what a close drops, doctor picks up
  expect(noteRow(db, "notes/globex.md")!.hash).toBe(
    new Bun.CryptoHasher("sha256").update(FIXTURE["notes/globex.md"]).digest("hex"),
  );
  db.close();
});

test("a model swap under the watcher re-embeds the whole vault, not just the touched path", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder); // 32 dims

  let reported: { reembedded: boolean } | undefined;
  const wider = new TokenOverlapEmbedder(64);
  const w = watch(db, root, wider, { debounceMs: 1, onChange: (_paths, stats) => (reported = stats) });
  w.touch("notes/acme.md");
  await w.idle();

  expect(reported?.reembedded).toBe(true);
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 2 });
  expect(db.query("select count(*) as c from vector_meta where dims = 64").get()).toEqual({ c: 2 });
  await w.close();
  db.close();
});

test("close stops the watcher: later events are dropped", async () => {
  const root = makeVault(FIXTURE);
  const db = open(root);
  await reindex(db, root, embedder);
  const w = watch(db, root, embedder, { debounceMs: 1 });
  w.close();

  writeNote(root, "notes/acme.md", "# Acme Corp\n\nafter close\n");
  w.touch("notes/acme.md");
  await Bun.sleep(20);
  await w.idle();
  expect(noteRow(db, "notes/acme.md")!.hash).toBe(
    new Bun.CryptoHasher("sha256").update(FIXTURE["notes/acme.md"]).digest("hex"),
  );
  db.close();
});

test("real fs.watch events: an edit, a create and a delete reach the index", async () => {
  const root = makeVault(FIXTURE);
  const vault = openVault(root, { embedder });
  await vault.reindex();
  const w = vault.watch({ debounceMs: 25 });
  // a second connection: reading what the watcher's own handle writes
  const db = open(root);
  try {
    const acme = noteRow(db, "notes/acme.md")!;
    const vecBefore = [...vecOf(db, acme.id)];

    writeNote(root, "notes/acme.md", "# Acme Corp\n\nrewritten on disk by a human\n");
    await until("the edit to be indexed", () => noteRow(db, "notes/acme.md")!.hash !== acme.hash);
    expect([...vecOf(db, acme.id)]).not.toEqual(vecBefore);

    writeNote(root, "notes/fresh.md", "# Fresh\n\nwritten straight into the vault\n");
    const fresh = await until("the new note to be indexed", () => noteRow(db, "notes/fresh.md"));
    expect(vecOf(db, fresh.id)).toHaveLength(32);

    rmSync(join(root, "notes", "fresh.md"));
    await until("the deletion to be indexed", () => noteRow(db, "notes/fresh.md") === null);
    expect(db.query("select count(*) as c from vectors where note_id = ?").get(fresh.id)).toEqual({
      c: 0,
    });
  } finally {
    w.close();
    await w.idle();
    db.close();
    vault.close();
  }
}, 30_000);

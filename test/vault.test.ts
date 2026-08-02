// The facade's two direct read paths (DESIGN.md § Retrieval): `get` reads the
// file, `list` reads the index. Files are truth, so a stale or missing index
// row never changes what `get` hands back — and only a regular `.md` file under
// the root is a note at all.
import { test, expect, afterAll } from "bun:test";
import { mkdirSync, rmSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import { TokenOverlapEmbedder } from "../src/embed";
import { open, type Vault } from "../src/vault";
import { makeVault, cleanupVaults, writeNote } from "./vault-fixture";

afterAll(cleanupVaults);

const embedder = new TokenOverlapEmbedder();

const VAULT = {
  "ledger/q3.md": "---\ntitle: Q3\n---\n\nThe Q3 ledger.\n",
  "ledger/q4.md": "# Q4\n\nThe Q4 ledger.\n",
  // The sibling namespace a raw `startsWith` prefix would wrongly sweep in.
  "ledger-archive/q3.md": "# Archived Q3\n\nSuperseded numbers.\n",
  "root-note.md": "# Root note\n\nAt the vault root.\n",
};

async function openVault(files: Record<string, string> = VAULT): Promise<Vault> {
  const v = open(makeVault(files), { embedder });
  await v.reindex();
  return v;
}

test("get reads the file, never the index row", async () => {
  const v = await openVault();
  // One note edited behind the index's back, one never indexed at all.
  writeNote(v.root, "ledger/q3.md", "---\ntitle: Q3 (corrected)\n---\n\nRevised.\n");
  writeNote(v.root, "ledger/q5.md", "# Q5\n\nNot in the index.\n");

  const q3 = await v.get("ledger/q3.md");
  expect(q3).toMatchObject({ path: "ledger/q3.md", slug: "q3", title: "Q3 (corrected)" });
  expect(q3?.body).toContain("Revised.");

  expect(v.list()).not.toContain("ledger/q5.md"); // the index has not caught up
  expect((await v.get("ledger/q5.md"))?.title).toBe("Q5"); // the file is the truth
});

test("get: absent, a directory, a symlink and a non-.md path are all null", async () => {
  const v = await openVault();
  expect(await v.get("ledger/nope.md")).toBeNull();
  // Identity includes the extension: `ledger/q3` names no note.
  expect(await v.get("ledger/q3")).toBeNull();

  mkdirSync(join(v.root, "ledger", "sub.md"));
  expect(await v.get("ledger/sub.md")).toBeNull();

  // A symlink is a second name for a file that may not be a note at all — the
  // scan skips them, so `get` must not serve one either. Least of all one
  // pointing clean out of the vault.
  const outside = makeVault({ "secret.md": "# Secret\n\nNot in this vault.\n" });
  symlinkSync(join(v.root, "ledger", "q4.md"), join(v.root, "link.md"));
  symlinkSync(join(outside, "secret.md"), join(v.root, "ledger", "leak.md"));
  expect(await v.get("link.md")).toBeNull();
  expect(await v.get("ledger/leak.md")).toBeNull();
});

test("get: outside the root, through a dot-directory or a symlinked one is refused", async () => {
  const v = await openVault();
  await expect(v.get("../../etc/passwd")).rejects.toThrow(/outside the vault/);
  await expect(v.get("/etc/passwd")).rejects.toThrow(/outside the vault/);
  // The index lives in `.vault/`, and the scan skips dot-directories — nothing
  // in one is a note, whatever its extension.
  await expect(v.get(".vault/index.db")).rejects.toThrow(/hidden directory/);
  await expect(v.get(".vault/notes.md")).rejects.toThrow(/hidden directory/);

  symlinkSync(join(v.root, "ledger"), join(v.root, "linked"));
  await expect(v.get("linked/q4.md")).rejects.toThrow(/symlink/);
});

test("list: every indexed note path, sorted", async () => {
  const v = await openVault();
  expect(v.list()).toEqual([
    "ledger-archive/q3.md",
    "ledger/q3.md",
    "ledger/q4.md",
    "root-note.md",
  ]);
  expect(v.list("")).toEqual(v.list());
});

test("list: a prefix matches on segment boundaries, slash or no slash", async () => {
  const v = await openVault();
  const ledger = ["ledger/q3.md", "ledger/q4.md"];
  expect(v.list("ledger")).toEqual(ledger);
  expect(v.list("ledger/")).toEqual(ledger);
  expect(v.list("ledger-archive")).toEqual(["ledger-archive/q3.md"]);
  // A prefix is a namespace, not a string match: neither half a segment nor a
  // namespace that does not exist brings anything back.
  expect(v.list("ledg")).toEqual([]);
  expect(v.list("nope")).toEqual([]);
});

test("list follows the files: a reindex adds and drops paths", async () => {
  const v = await openVault();
  writeNote(v.root, "ledger/q5.md", "# Q5\n\nNew.\n");
  rmSync(join(v.root, "ledger", "q4.md"));
  await v.reindex();
  expect(v.list("ledger")).toEqual(["ledger/q3.md", "ledger/q5.md"]);
});

test("list is the note set, not the search set: superseded notes are listed", async () => {
  const v = await openVault({
    "a.md": "---\nsuperseded_by: b.md\n---\n\nThe old note.\n",
    "b.md": "# B\n\nThe new note.\n",
  });
  expect(v.list()).toEqual(["a.md", "b.md"]);
});

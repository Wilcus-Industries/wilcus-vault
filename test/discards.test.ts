// Discard log tooling (#33): the read side — list/show/restore — plus the
// write-side rails it leans on: size rotation, the auto-gitignore, and the
// CLI-only FetchDecider. Fake deciders and injected fetch, no network.
import { test, expect, afterAll } from "bun:test";
import { appendFileSync, existsSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TokenOverlapEmbedder } from "../src/embed";
import { fetchDecider } from "../src/decide";
import { countDiscards, getDiscard, listDiscards, restoreDiscard } from "../src/discards";
import { discardLog, logCandidate, type Decider, type Decision } from "../src/gate";
import { open, type Cutoffs } from "../src/vault";
import { makeVault, cleanupVaults, withEnv } from "./vault-fixture";

afterAll(cleanupVaults);

const embedder = new TokenOverlapEmbedder();
const CUTOFFS: Cutoffs = { distanceCeiling: 0.9, bm25Ceiling: 0 };

test("list/show/restore round-trip: a discard comes back through the gate", async () => {
  const root = makeVault({ "notes/rota.md": "# Support rota\n\nwho carries the pager\n" });
  let decision: Decision = { action: "discard" };
  const decider: Decider = async () => decision;
  const v = open(root, { embedder, gate: { decider, cutoffs: CUTOFFS } });
  await v.reindex();

  await v.propose({ title: "Pager duty", namespace: "notes", body: "Pager rotation is weekly.\n" });
  await v.propose({ title: "Second thought", namespace: "notes", body: "Another pager candidate.\n" });

  const { entries, malformed } = listDiscards(root);
  expect(malformed).toBe(0);
  // newest first: 1 is the entry most recently refused
  expect(entries.map((e) => [e.n, e.candidate.title])).toEqual([
    [1, "Second thought"],
    [2, "Pager duty"],
  ]);
  expect(entries[0]!.decision).toEqual({ action: "discard" });
  expect(entries[0]!.at).toMatch(/^20\d\d-/);

  // show: the full candidate, recoverable byte for byte
  const full = getDiscard(root, 2)!;
  expect(full.candidate).toEqual({
    title: "Pager duty",
    namespace: "notes",
    body: "Pager rotation is weekly.\n",
  });
  expect(getDiscard(root, 9)).toBeNull();

  // restore re-runs search+decide against *current* state — the decider now
  // says create, so the candidate lands as a brand-new note through the gate
  decision = { action: "create" };
  const created = await restoreDiscard(v, 2);
  expect(created).toMatchObject({ action: "create", path: "notes/pager-duty.md" });
  expect(readFileSync(join(root, "notes/pager-duty.md"), "utf8")).toContain(
    "Pager rotation is weekly.",
  );

  // ...and with the vault in a different state the same mechanism lands as an
  // update: current state decides, not the state at discard time
  decision = {
    action: "update",
    target: "notes/rota.md",
    body: "# Support rota\n\nwho carries the pager, restored detail\n",
  };
  const updated = await restoreDiscard(v, 1);
  expect(updated).toMatchObject({ action: "update", path: "notes/rota.md" });
  expect(readFileSync(join(root, "notes/rota.md"), "utf8")).toContain("restored detail");

  await expect(restoreDiscard(v, 99)).rejects.toThrow(/no entry 99/);
  v.close();
});

test("rotation at the cap: nothing deleted, numbering spans the rotated files", () => {
  const root = makeVault({});
  const cand = (title: string) => ({ title, body: "body\n" });
  // A 1-byte cap makes every write-after-first rotate: .discarded.log fills,
  // moves to .discarded.1.log, then .2 — oldest entries in the lowest number.
  logCandidate(root, cand("one"), { similar: [] }, 1);
  logCandidate(root, cand("two"), { similar: [] }, 1);
  logCandidate(root, cand("three"), { similar: [] }, 1);

  expect(readFileSync(join(root, ".discarded.1.log"), "utf8")).toContain("one");
  expect(readFileSync(join(root, ".discarded.2.log"), "utf8")).toContain("two");
  expect(readFileSync(discardLog(root), "utf8")).toContain("three");

  // every entry still readable, numbered newest-first across all files
  const { entries } = listDiscards(root);
  expect(entries.map((e) => e.candidate.title)).toEqual(["three", "two", "one"]);
  expect(getDiscard(root, 3)!.candidate.title).toBe("one");
});

test("first write to the log creates the gitignore entry exactly once", () => {
  const root = makeVault({});
  const gi = join(root, ".gitignore");
  logCandidate(root, { title: "a", body: "b\n" }, { similar: [] });
  expect(readFileSync(gi, "utf8")).toBe(".discarded.log*\n");
  logCandidate(root, { title: "b", body: "b\n" }, { similar: [] });
  expect(readFileSync(gi, "utf8")).toBe(".discarded.log*\n"); // byte-identical
});

test("an existing .gitignore is extended; a deliberately removed line stays removed", () => {
  const root = makeVault({});
  const gi = join(root, ".gitignore");
  writeFileSync(gi, "node_modules"); // no trailing newline: the append must not fuse lines
  logCandidate(root, { title: "a", body: "b\n" }, { similar: [] });
  expect(readFileSync(gi, "utf8")).toBe("node_modules\n.discarded.log*\n");

  // The user strips the line while the log exists: their choice, not drift to
  // repair — only a *first* write (no log on disk) ever touches .gitignore.
  writeFileSync(gi, "node_modules\n");
  logCandidate(root, { title: "b", body: "b\n" }, { similar: [] });
  expect(readFileSync(gi, "utf8")).toBe("node_modules\n");
});

test("a malformed log line is counted and skipped, never fatal", () => {
  const root = makeVault({});
  logCandidate(root, { title: "good", body: "b\n" }, { similar: [] });
  appendFileSync(discardLog(root), "not json at all\n");
  appendFileSync(discardLog(root), `{"at":"2026-01-01T00:00:00Z"}\n`); // json, but no candidate
  const { entries, malformed } = listDiscards(root);
  expect(entries.map((e) => e.candidate.title)).toEqual(["good"]);
  expect(malformed).toBe(2);
});

test("a symlink planted among the log files is refused, never read through", () => {
  // The write side refuses a symlinked log (O_NOFOLLOW); the read side must
  // match, or `discards list` becomes a way to print any readable file — and
  // `restore` a way to propose its content into the vault.
  const root = makeVault({});
  logCandidate(root, { title: "real", body: "b\n" }, { similar: [] });
  const elsewhere = makeVault({});
  writeFileSync(
    join(elsewhere, "secrets.log"),
    `${JSON.stringify({ at: "2026-01-01T00:00:00.000Z", candidate: { title: "stolen", body: "s" } })}\n`,
  );
  symlinkSync(join(elsewhere, "secrets.log"), join(root, ".discarded.1.log"));
  expect(() => listDiscards(root)).toThrow(/symlink/);
});

test("countDiscards splits total from recent", () => {
  const root = makeVault({});
  expect(countDiscards(root)).toEqual({ entries: 0, recent: 0 });
  const line = (at: string) => `${JSON.stringify({ at, candidate: { title: "t", body: "b" } })}\n`;
  writeFileSync(discardLog(root), line("2020-01-01T00:00:00.000Z") + line(new Date().toISOString()));
  expect(countDiscards(root)).toEqual({ entries: 2, recent: 1 });
});

test("fetchDecider: a chat model is mandatory, and the reply goes through parseDecision", async () => {
  // No model, no default: a chat model name cannot be guessed the way the
  // local embedding model can.
  await withEnv({}, async () => {
    expect(() => fetchDecider()).toThrow(/VAULT_DECIDE_MODEL/);
  });

  // The decider posts the gate prompt and parses one strict JSON object back.
  const seen: { url: string; body: string } = { url: "", body: "" };
  const decide = fetchDecider({
    model: "test-chat",
    fetch: (async (url: string, init: RequestInit) => {
      seen.url = String(url);
      seen.body = String(init.body);
      return new Response(
        JSON.stringify({ choices: [{ message: { content: `{"action":"create"}` } }] }),
      );
    }) as typeof fetch,
  });
  const decision = await decide({ candidate: { title: "T", body: "B\n" }, similar: [] });
  expect(decision).toEqual({ action: "create" });
  expect(seen.url).toContain("localhost:11434"); // unconfigured stays local
  expect(seen.body).toContain("test-chat");
  expect(seen.body).toContain("write gate"); // the gate prompt, not a bespoke one

  // a chatty or fenced reply is malformed, exactly as the library treats it
  const chatty = fetchDecider({
    model: "test-chat",
    fetch: (async () =>
      new Response(
        JSON.stringify({ choices: [{ message: { content: "Sure! {\"action\":\"create\"}" } }] }),
      )) as unknown as typeof fetch,
  });
  await expect(chatty({ candidate: { title: "T", body: "B\n" }, similar: [] })).rejects.toThrow(
    /write gate/,
  );

  // a remote endpoint without a key is refused at construction, like FetchEmbedder
  await withEnv({ VAULT_DECIDE_ENDPOINT: "https://api.example.invalid/v1/chat/completions" }, async () => {
    expect(() => fetchDecider({ model: "m" })).toThrow(/API key/);
  });
});

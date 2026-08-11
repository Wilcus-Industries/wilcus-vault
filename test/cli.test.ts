import { test, expect, afterAll, spyOn } from "bun:test";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { dbPath } from "../src/db";
import { main } from "../src/cli";
import { makeVault, cleanupVaults, withEnv } from "./vault-fixture";

afterAll(cleanupVaults);

const GRAPH = {
  "notes/acme.md": "---\ntype: customer\n---\n# Acme Corp\n\nrenewal terms, see [[globex]], [[ghost]], [[dup]]\n",
  "notes/globex.md": "# Globex\n\nno outgoing links\n",
  "notes/lonely.md": "# Lonely\n\nnothing here\n",
  "one/dup.md": "# Dup one\n",
  "two/dup.md": "# Dup two\n",
};

/** Run the CLI with console captured; returns the exit code and both streams. */
async function cli(...argv: string[]): Promise<{ code: number; out: string; err: string }> {
  const log = spyOn(console, "log").mockImplementation(() => {});
  const error = spyOn(console, "error").mockImplementation(() => {});
  try {
    const code = await main(argv);
    return {
      code,
      out: log.mock.calls.flat().join("\n"),
      err: error.mock.calls.flat().join("\n"),
    };
  } finally {
    log.mockRestore();
    error.mockRestore();
  }
}

test("cli: reindex, doctor and doctor --rebuild", async () => {
  const root = makeVault(GRAPH);
  expect(await cli("reindex", "--lexical", "--vault", root)).toMatchObject({ code: 0, err: "" });
  expect(existsSync(dbPath(root))).toBe(true);

  // GRAPH has a broken link, an ambiguous link and a duplicate stem: doctor
  // repaired what it could, so the exit code has to say work is left
  const doctored = await cli("doctor", "--lexical", "--vault", root);
  expect(doctored.code).toBe(1);
  expect(doctored.out).toContain("ghost");
  expect(doctored.err).toBe("");
  expect(await cli("doctor", "--rebuild", "--lexical", "--vault", root)).toMatchObject({ code: 1 });
});

test("cli: doctor names the candidates that would qualify an ambiguous link", async () => {
  const root = makeVault(GRAPH);
  const { code, out } = await cli("doctor", "--lexical", "--vault", root);
  expect(code).toBe(1);
  expect(out).toContain("broken link:    notes/acme.md -> [[ghost]]");
  // the line is the fix: copy one candidate into the note as written
  expect(out).toContain("ambiguous link: notes/acme.md -> [[dup]] (one/dup, two/dup)");
});

test("cli: control characters from a note never reach the terminal raw", async () => {
  // a link target and a filename are note-controlled strings; a bare \r or ESC
  // in one would rewrite the line the CLI just printed
  const root = makeVault({ "notes/evil.md": "# Evil\n\n[[gh\rostX]]\n" });
  const { out } = await cli("doctor", "--lexical", "--vault", root);
  expect(out).toContain("gh?ost?X");
  // any control character other than the newlines the CLI itself writes
  expect(out).not.toMatch(/[^\n\P{Cc}]/u);
});

test("cli: a clean vault exits 0; usage goes to stderr", async () => {
  const root = makeVault({ "a.md": "# A\n\n[[b]]\n", "b.md": "# B\n\n[[a]]\n" });
  expect(await cli("doctor", "--lexical", "--vault", root)).toMatchObject({ code: 0, err: "" });

  expect((await cli()).code).toBe(1);
  expect((await cli("nope")).code).toBe(1);
  expect((await cli("nope")).err).toContain("unknown command nope");
  // a flag swallowed as the vault path would index the wrong directory
  expect((await cli("doctor", "--vault", "--rebuild")).code).toBe(1);
  expect((await cli("doctor", "--vault")).code).toBe(1);
  expect((await cli("doctor", "--wat")).err).toContain("unknown flag --wat");
  expect((await cli()).err).toContain("vault <command>");
});

test("cli: --help documents every command and exits 0", async () => {
  for (const argv of [["--help"], ["-h"], ["search", "--help"], ["watch", "-h"], ["doctor", "--help"]]) {
    const { code, out, err } = await cli(...argv);
    expect(code).toBe(0); // help was asked for; answering it is not a failure
    expect(err).toBe("");
    for (const command of [
      "reindex",
      "doctor",
      "search",
      "watch",
      "consolidate",
      "discards",
      "--vault",
      "--lexical",
      "--ceiling",
    ]) {
      expect(out).toContain(command);
    }
  }
});

test("cli: consolidate reports the clusters it found and writes nothing", async () => {
  const twins = "# Renewal\n\nthe acme renewal closes in march\n";
  const root = makeVault({
    "notes/one.md": twins,
    "notes/two.md": twins,
    "notes/far.md": "# Zeppelin\n\ntorque values for the fuselage struts\n",
  });
  const { code, out, err } = await cli("consolidate", "--ceiling", "0.2", "--lexical", "--vault", root);
  expect(code).toBe(0);
  expect(err).toBe("");
  expect(out).toContain("notes/one.md notes/two.md");
  expect(out).not.toContain("notes/far.md");
  // Report-only: merging needs an injected merger (an LLM), which the CLI has
  // no way to wire — so nothing on disk may have moved.
  expect(readdirSync(join(root, "notes")).sort()).toEqual(["far.md", "one.md", "two.md"]);
  expect(readFileSync(join(root, "notes/one.md"), "utf8")).toBe(twins);

  // the ceiling is mandatory here too, and it is a cosine distance
  expect((await cli("consolidate", "--lexical", "--vault", root)).code).toBe(1);
  expect((await cli("consolidate", "--ceiling", "9", "--lexical", "--vault", root)).code).toBe(1);
  expect((await cli("consolidate", "--ceiling", "--lexical", "--vault", root)).code).toBe(1);
});

test("cli: search prints one readable line per hit — score, then path", async () => {
  const root = makeVault(GRAPH);
  await cli("reindex", "--lexical", "--vault", root);
  const { code, out, err } = await cli("search", "renewal", "terms", "--lexical", "--vault", root);
  expect(code).toBe(0);
  expect(err).toBe("");
  expect(out).toMatch(/^0\.\d{4}\s+notes\/acme\.md\b.*Acme Corp$/m);
  // every hit is on one line, scores first, so the ranking reads down the page
  expect(out.split("\n")).toHaveLength(5);
  // a query with nothing to search on is an answer, not an error: the CLI sets
  // no relevance cutoffs (none is meaningful for a bag-of-tokens embedder), so
  // this is the empty case — both signals had nothing to ask
  expect(await cli("search", "!!!", "--lexical", "--vault", root)).toMatchObject({
    code: 0,
    out: "no matches",
  });
  // and a query is required
  expect((await cli("search", "--lexical", "--vault", root)).code).toBe(1);
});

test("cli: searching an unindexed vault says so rather than reporting nothing", async () => {
  const root = makeVault(GRAPH);
  expect(await cli("search", "acme", "--lexical", "--vault", root)).toMatchObject({
    code: 0,
    out: "vault is not indexed (run vault reindex)",
  });
});

test("cli: discards list/show read the log; restore is gated on config", async () => {
  const root = makeVault(GRAPH);
  // an empty log is an answer, not an error
  expect(await cli("discards", "list", "--lexical", "--vault", root)).toMatchObject({
    code: 0,
    out: "no discards",
  });

  const entry = {
    at: "2026-01-01T00:00:00.000Z",
    candidate: { title: "Dropped thought", namespace: "notes", body: "body\n" },
    decision: { action: "discard" },
    similar: [],
  };
  writeFileSync(join(root, ".discarded.log"), `${JSON.stringify(entry)}\n`);

  const list = await cli("discards", "list", "--lexical", "--vault", root);
  expect(list.code).toBe(0);
  expect(list.out).toContain("Dropped thought");
  expect(list.out).toContain("discard");

  const show = await cli("discards", "show", "1", "--lexical", "--vault", root);
  expect(show.code).toBe(0);
  expect(JSON.parse(show.out)).toMatchObject({ candidate: { title: "Dropped thought" } });
  expect((await cli("discards", "show", "9", "--lexical", "--vault", root)).code).toBe(1);

  // restore needs a ceiling (like consolidate) and a configured chat model —
  // refused with the fix named, before any database is opened
  expect((await cli("discards", "restore", "1", "--lexical", "--vault", root)).code).toBe(1);
  const noModel = await withEnv({}, () =>
    cli("discards", "restore", "1", "--ceiling", "0.5", "--lexical", "--vault", root),
  );
  expect(noModel.code).toBe(1);
  expect(noModel.err).toContain("VAULT_DECIDE_MODEL");

  // a missing or malformed subcommand is usage, not a crash
  expect((await cli("discards", "--lexical", "--vault", root)).code).toBe(1);
  expect((await cli("discards", "show", "x", "--lexical", "--vault", root)).code).toBe(1);

  // doctor surfaces the log so normal maintenance sees it
  const doctored = await cli("doctor", "--lexical", "--vault", root);
  expect(doctored.out).toContain("discard log: 1 entries (0 recent)");
});

// A remote endpoint with no key that FetchEmbedder refuses to construct at all,
// so this reaches no network: it is offline proof of *which* embedder the CLI
// built, because TokenOverlapEmbedder never reads the environment.
const REMOTE = { VAULT_EMBED_ENDPOINT: "https://api.example.invalid/v1/embeddings" };

test("cli: the default embedder is the real one; --lexical is the offline escape hatch", async () => {
  const root = makeVault(GRAPH);
  for (const command of [["reindex"], ["doctor"], ["watch"], ["search", "acme"]]) {
    const { code, err } = await withEnv(REMOTE, () => cli(...command, "--vault", root));
    expect(code).toBe(1);
    expect(err).toBe("FetchEmbedder: no API key — pass apiKey or set VAULT_EMBED_API_KEY");
  }
  // same environment, --lexical: no provider is consulted, so the vault indexes
  expect(await withEnv(REMOTE, () => cli("reindex", "--lexical", "--vault", root))).toMatchObject({
    code: 0,
    err: "",
  });
});

test("cli: an unreachable embedder is one line on stderr, never a stack trace", async () => {
  const root = makeVault(GRAPH);
  // Nothing listens on port 1, and it is loopback: a refusal, not a network
  // call. (That a refused *default* endpoint says "no embedder configured:
  // start Ollama ..." is embed.test.ts's; this is that the CLI prints whatever
  // the embedder said as one sentence and exits 1.)
  const { code, out, err } = await withEnv({ VAULT_EMBED_ENDPOINT: "http://127.0.0.1:1/v1/embeddings" }, () =>
    cli("reindex", "--vault", root),
  );
  expect(code).toBe(1);
  expect(out).toBe("");
  expect(err).not.toBe("");
  expect(err).not.toContain("    at "); // a stack frame is not an error message
});

test("cli: an error message cannot rewrite the terminal it is printed to", async () => {
  // Everything on stderr was written by someone else: an argument, a note, and
  // now up to 200 bytes of whatever answered on :11434. A carriage return or an
  // ESC in any of them redraws the line the terminal has already printed, so
  // stderr gets the same scrub stdout has always had. Argv is the shortest
  // untrusted string that reaches the catch all of them land in — and the bytes
  // are built by code, so no editor or patch tool can quietly normalize them.
  const ESC = String.fromCharCode(27);
  const CR = String.fromCharCode(13);
  const control = (c: string) => (c.codePointAt(0) ?? 0) < 32 || c.codePointAt(0) === 127;
  for (const arg of [`--${ESC}[2K${CR}x`, `${ESC}[2K${CR}x`]) {
    const { code, err } = await cli(arg);
    expect(code).toBe(1);
    // every control character is gone but the newlines the CLI itself writes
    expect([...err].filter(control)).toEqual([...err].filter((c) => c === "\n"));
    expect(err).toContain("?[2K?x");
    expect(err).toContain("vault <command>"); // and a multi-line usage error stays readable
  }
});

test("cli: search with no query reports a broken embedder before it asks for one", async () => {
  const root = makeVault(GRAPH);
  // The embedder is built before the query is checked, so a misconfigured one
  // is what a bare `vault search` complains about. Pinned because the order is
  // arbitrary: swapping it would silently change this message.
  const { code, err } = await withEnv(REMOTE, () => cli("search", "--vault", root));
  expect(code).toBe(1);
  expect(err).toBe("FetchEmbedder: no API key — pass apiKey or set VAULT_EMBED_API_KEY");
});

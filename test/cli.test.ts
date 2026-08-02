import { test, expect, afterAll, spyOn } from "bun:test";
import { existsSync } from "node:fs";
import { dbPath } from "../src/db";
import { main } from "../src/cli";
import { makeVault, cleanupVaults } from "./vault-fixture";

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
  expect(await cli("reindex", "--vault", root)).toMatchObject({ code: 0, err: "" });
  expect(existsSync(dbPath(root))).toBe(true);

  // GRAPH has a broken link, an ambiguous link and a duplicate stem: doctor
  // repaired what it could, so the exit code has to say work is left
  const doctored = await cli("doctor", "--vault", root);
  expect(doctored.code).toBe(1);
  expect(doctored.out).toContain("ghost");
  expect(doctored.err).toBe("");
  expect(await cli("doctor", "--rebuild", "--vault", root)).toMatchObject({ code: 1 });
});

test("cli: doctor names the candidates that would qualify an ambiguous link", async () => {
  const root = makeVault(GRAPH);
  const { code, out } = await cli("doctor", "--vault", root);
  expect(code).toBe(1);
  expect(out).toContain("broken link:    notes/acme.md -> [[ghost]]");
  // the line is the fix: copy one candidate into the note as written
  expect(out).toContain("ambiguous link: notes/acme.md -> [[dup]] (one/dup, two/dup)");
});

test("cli: control characters from a note never reach the terminal raw", async () => {
  // a link target and a filename are note-controlled strings; a bare \r or ESC
  // in one would rewrite the line the CLI just printed
  const root = makeVault({ "notes/evil.md": "# Evil\n\n[[gh\rostX]]\n" });
  const { out } = await cli("doctor", "--vault", root);
  expect(out).toContain("gh?ost?X");
  // any control character other than the newlines the CLI itself writes
  expect(out).not.toMatch(/[^\n\P{Cc}]/u);
});

test("cli: a clean vault exits 0; usage goes to stderr", async () => {
  const root = makeVault({ "a.md": "# A\n\n[[b]]\n", "b.md": "# B\n\n[[a]]\n" });
  expect(await cli("doctor", "--vault", root)).toMatchObject({ code: 0, err: "" });

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
    for (const command of ["reindex", "doctor", "search", "watch", "--vault"]) {
      expect(out).toContain(command);
    }
  }
});

test("cli: search prints one readable line per hit — score, then path", async () => {
  const root = makeVault(GRAPH);
  await cli("reindex", "--vault", root);
  const { code, out, err } = await cli("search", "renewal", "terms", "--vault", root);
  expect(code).toBe(0);
  expect(err).toBe("");
  expect(out).toMatch(/^0\.\d{4}\s+notes\/acme\.md\b.*Acme Corp$/m);
  // every hit is on one line, scores first, so the ranking reads down the page
  expect(out.split("\n")).toHaveLength(5);
  // a query with nothing to search on is an answer, not an error: the CLI sets
  // no relevance cutoffs (none is meaningful for a bag-of-tokens embedder), so
  // this is the empty case — both signals had nothing to ask
  expect(await cli("search", "!!!", "--vault", root)).toMatchObject({
    code: 0,
    out: "no matches",
  });
  // and a query is required
  expect((await cli("search", "--vault", root)).code).toBe(1);
});

test("cli: searching an unindexed vault says so rather than reporting nothing", async () => {
  const root = makeVault(GRAPH);
  expect(await cli("search", "acme", "--vault", root)).toMatchObject({
    code: 0,
    out: "vault is not indexed (run vault reindex)",
  });
});

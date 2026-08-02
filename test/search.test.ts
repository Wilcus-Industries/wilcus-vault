// Retrieval evals (DESIGN.md § Testing): deterministic, seeded vaults, the
// TokenOverlapEmbedder standing in for semantics. No network.
import { test, expect, afterAll, spyOn } from "bun:test";
import type { Database } from "bun:sqlite";
import { openDb, dbPath } from "../src/db";
import { TokenOverlapEmbedder } from "../src/embed";
import { reindex } from "../src/indexer";
import { hybridSearch, ftsQuery, type SearchHit } from "../src/search";
import { makeVault, cleanupVaults, stubEmbedder } from "./vault-fixture";
import { main } from "../src/cli";

afterAll(cleanupVaults);

const embedder = new TokenOverlapEmbedder();

async function indexed(files: Record<string, string>): Promise<{ root: string; db: Database }> {
  const root = makeVault(files);
  const db = openDb(dbPath(root));
  await reindex(db, root, embedder);
  return { root, db };
}

const paths = (hits: SearchHit[]): string[] => hits.map((h) => h.path);

/**
 * The eval vault, seeded so the query "acme renewal pricing" has a *different*
 * top hit on each signal. "acme" and "renewal" sit in half of the ten notes, so
 * BM25 gives them no IDF and the keyword ranking turns entirely on the rare
 * term "pricing"; the vector ranking turns on token purity instead. That pulls
 * the two signals apart: `renewal-stub` is the vector top (nothing but query
 * tokens) and nowhere near the BM25 top, `pricing-model` is the BM25 top (the
 * rare term, three times) and nowhere near the vector top, and `acme-renewal` —
 * the note a human wants — is second on both and first on neither.
 */
const EVAL_VAULT: Record<string, string> = {
  "ledger/invoice-2031.md":
    "---\ntype: ledger\n---\n# Invoice INV-2031\n\nAnnual license invoice INV-2031 issued to Acme Corp, net 30 days.\n",
  "notes/account-history.md":
    "# Account history\n\nThe Acme account has run since 2019; the last renewal was signed without changes and the paperwork sits with legal.\n",
  "notes/acme-contacts.md":
    "# Acme contacts\n\nDay to day contacts at Acme: the procurement lead owns the Acme renewal thread.\n",
  "notes/acme-renewal.md":
    "---\ntype: customer\n---\n# Acme renewal\n\nThe Acme renewal closes in March. Renewal pricing is agreed; Acme signs the order form. See [[support-rota]].\n",
  "notes/discount-approvals.md":
    "# Discount approvals\n\nWho signs off a discount, the escalation ladder, the finance review window, and where an approved pricing exception is recorded afterwards.\n",
  "notes/globex.md":
    "# Globex\n\nVendor policy for the Globex account: procurement, security review, invoicing cadence.\n",
  "notes/pricing-model.md":
    "# Pricing model\n\nPricing bands, pricing exceptions, discount approval, procurement thresholds, escalation path, finance sign off, quarterly review cadence.\n",
  "notes/renewal-playbook.md":
    "# Renewal playbook\n\nThe renewal playbook: notice periods, uplift caps, and the renewal calendar.\n",
  "notes/renewal-stub.md": "# Acme renewal\n\nAcme renewal.\n",
  "notes/support-rota.md":
    "# Support rota\n\nWho carries the pager each week, and how the handover works.\n",
};

test("eval: an exact identifier is the top hit", async () => {
  const { db } = await indexed(EVAL_VAULT);
  const hits = await hybridSearch(db, embedder, "INV-2031");
  expect(hits[0]!.path).toBe("ledger/invoice-2031.md");
  db.close();
});

test("eval: a reworded query finds the note through the vector path", async () => {
  const { db } = await indexed({
    "notes/onboarding.md":
      "# Customer onboarding\n\nSend the welcome email, create the shared workspace, then schedule a call to kick off the work.\n",
    "notes/email-policy.md": "# Email policy\n\nRetention rules for email archives.\n",
    "notes/globex.md": EVAL_VAULT["notes/globex.md"]!,
  });

  // reordered and padded — no phrase in common, plenty of tokens in common
  const hits = await hybridSearch(db, embedder, "how do we kick off a new customer: workspace, welcome call, email");
  expect(hits[0]!.path).toBe("notes/onboarding.md");

  // A hyphenated term is one FTS phrase ("kick off" adjacent, in that order)
  // and two embedder tokens, so this query has no keyword match at all: only
  // the vector side can carry it.
  const vecOnly = await hybridSearch(db, embedder, "customer-workspace");
  expect(vecOnly[0]).toMatchObject({ path: "notes/onboarding.md", ftsRank: null, vecRank: 1 });
  db.close();
});

test("eval: fusion beats either signal alone", async () => {
  const { db } = await indexed(EVAL_VAULT);
  // n=1 ⇒ each signal over-fetches 3, so a note outside a signal's top 3 is
  // absent from it entirely — exactly what a single-signal search would miss.
  const hits = await hybridSearch(db, embedder, "acme renewal pricing", { n: 1 });
  expect(paths(hits)).toEqual(["notes/acme-renewal.md"]);

  const both = await hybridSearch(db, embedder, "acme renewal pricing", { n: 5 });
  const target = both.find((h) => h.path === "notes/acme-renewal.md")!;
  // second-best on both signals: neither ranking alone puts it first
  expect(target.vecRank).toBe(2);
  expect(target.ftsRank).toBe(2);
  expect(both.find((h) => h.vecRank === 1)!.path).toBe("notes/renewal-stub.md");
  expect(both.find((h) => h.ftsRank === 1)!.path).toBe("notes/pricing-model.md");
  // and it still wins the fusion, by a clear margin over both single-signal tops
  expect(both[0]!.path).toBe("notes/acme-renewal.md");
  expect(target.score).toBeGreaterThan(both[1]!.score);
  db.close();
});

test("eval: a superseded note is filtered out of both signals", async () => {
  const { db } = await indexed({
    "notes/old-terms.md":
      "---\nsuperseded_by: notes/new-terms.md\n---\n# Acme payment terms\n\nAcme pays net 60 on the annual invoice.\n",
    "notes/new-terms.md": "# Acme payment terms 2026\n\nAcme pays net 30 on the annual invoice.\n",
  });
  const hits = await hybridSearch(db, embedder, "acme payment terms net");
  expect(paths(hits)).toEqual(["notes/new-terms.md"]);
  db.close();
});

test("eval: cutoffs make a garbage query return nothing", async () => {
  const { db } = await indexed(EVAL_VAULT);
  const garbage = "zzqqxx wwvvuu";
  // without cutoffs the vector side still returns its k nearest, however far
  expect((await hybridSearch(db, embedder, garbage)).length).toBeGreaterThan(0);
  // with them, nothing survives — and an empty result is a valid outcome
  expect(
    await hybridSearch(db, embedder, garbage, {
      cutoffs: { distanceCeiling: 0.5, bm25Ceiling: -1 },
    }),
  ).toEqual([]);
  db.close();
});

test("each cutoff narrows its own signal, in the direction it says", async () => {
  const { db } = await indexed(EVAL_VAULT);
  const q = "acme renewal pricing";

  // BM25 alone: FTS5's rank is negative and lower is better, so -1 keeps only
  // the strongest keyword hit. A reversed comparison would keep the rest.
  const keyword = await hybridSearch(db, embedder, q, { n: 10, cutoffs: { bm25Ceiling: -1 } });
  expect(keyword.filter((h) => h.ftsRank !== null).map((h) => h.path)).toEqual([
    "notes/pricing-model.md",
  ]);
  expect(keyword.filter((h) => h.vecRank !== null).length).toBeGreaterThan(1); // vector side untouched

  // Cosine distance alone: 0.3 keeps the two nearest (0.184, 0.265) and drops
  // the rest, while FTS keeps returning rows.
  const vector = await hybridSearch(db, embedder, q, { n: 10, cutoffs: { distanceCeiling: 0.3 } });
  expect(vector.filter((h) => h.vecRank !== null).map((h) => h.path).sort()).toEqual([
    "notes/acme-renewal.md",
    "notes/renewal-stub.md",
  ]);
  expect(vector.filter((h) => h.ftsRank !== null).length).toBeGreaterThan(2); // FTS side untouched
  db.close();
});

test("eval: FTS5 syntax in the query is text, not syntax", async () => {
  const { db } = await indexed(EVAL_VAULT);
  const hostile = [
    `"a OR b"`,
    `NEAR(acme renewal, 2)`,
    `acme AND NOT globex`,
    `acme*`,
    `body:acme`,
    `^acme`,
    `"""`,
    `((((`,
    `-`,
    `{acme}`,
    `acme" OR "renewal`,
  ];
  for (const q of hostile) {
    expect(Array.isArray(await hybridSearch(db, embedder, q))).toBe(true);
  }
  // every term is quoted, embedded quotes are doubled, and operators are terms
  expect(ftsQuery(`acme OR renewal`)).toBe(`"acme" OR "OR" OR "renewal"`);
  expect(ftsQuery(`a"b`)).toBe(`"a""b"`);
  expect(ftsQuery(`  ***  `)).toBeNull();
  // a whole note body is a legitimate query (the write gate passes one), so the
  // keyword side is capped at the first 32 distinct terms
  const long = ftsQuery(Array.from({ length: 200 }, (_, i) => `w${i} w${i}`).join(" "))!;
  expect(long.split(" OR ")).toHaveLength(32);
  expect(long.startsWith(`"w0" OR "w1" OR "w2"`)).toBe(true);
});

test("an FTS5 error drops the keyword signal instead of failing the search", async () => {
  const { db } = await indexed(EVAL_VAULT);
  // The quoting should make this unreachable; the fallback is what keeps a
  // tokenizer/Unicode skew from taking the whole search down.
  const wedged = new Proxy(db, {
    get(target, prop, receiver) {
      if (prop !== "query") return Reflect.get(target, prop, receiver);
      return (sql: string) => {
        if (sql.includes("notes_fts match ?")) throw new Error(`fts5: syntax error near "x"`);
        return target.query(sql);
      };
    },
  });
  const hits = await hybridSearch(wedged, embedder, "acme renewal pricing");
  expect(hits.length).toBeGreaterThan(0);
  expect(hits.every((h) => h.ftsRank === null)).toBe(true);
  db.close();
});

test("a token-less note has no vector row and stays findable through FTS", async () => {
  const { db } = await indexed({
    "notes/cjk.md": "# 圏点\n\n日本語 🙂\n",
    "notes/globex.md": EVAL_VAULT["notes/globex.md"]!,
  });
  expect(db.query("select count(*) as c from vectors").get()).toEqual({ c: 1 });
  // FTS-only note, and a query the embedder tokenizes to nothing: no vector
  // signal on either end, no NaN distances, still a hit
  const hits = await hybridSearch(db, embedder, "日本語");
  expect(paths(hits)).toEqual(["notes/cjk.md"]);
  expect(hits[0]).toMatchObject({ vecRank: null, ftsRank: 1 });
  db.close();
});

test("expandLinks appends one-hop neighbours below every direct hit", async () => {
  const { db } = await indexed({
    "notes/acme-renewal.md": EVAL_VAULT["notes/acme-renewal.md"]!, // links to support-rota
    "notes/support-rota.md": EVAL_VAULT["notes/support-rota.md"]!,
    "notes/globex.md": EVAL_VAULT["notes/globex.md"]!,
  });
  const plain = await hybridSearch(db, embedder, "acme renewal march", { n: 1 });
  expect(paths(plain)).toEqual(["notes/acme-renewal.md"]);

  const expanded = await hybridSearch(db, embedder, "acme renewal march", { n: 1, expandLinks: true });
  expect(paths(expanded)).toEqual(["notes/acme-renewal.md", "notes/support-rota.md"]);
  expect(expanded[1]).toMatchObject({ expansion: true, score: 0, vecRank: null, ftsRank: null });
  // an expansion never outranks a direct hit
  expect(expanded.filter((h) => !h.expansion).at(-1)!.score).toBeGreaterThan(expanded[1]!.score);
  db.close();
});

test("expandLinks walks a path-qualified link, past a stem two notes share", async () => {
  // the neighbour is reached by exact path — a bare [[globex]] here would be
  // ambiguous, resolve to nothing, and expand to nothing (the test above is
  // the bare-stem half of this)
  const { db } = await indexed({
    "notes/acme-renewal.md":
      "---\ntype: customer\n---\n# Acme renewal\n\nThe Acme renewal closes in March. Renewal pricing is agreed; Acme signs the order form. See [[vendors/globex]].\n",
    "vendors/globex.md": EVAL_VAULT["notes/globex.md"]!,
    "customers/globex.md": "# Globex the customer\n\nA different account with the same stem.\n",
  });
  const expanded = await hybridSearch(db, embedder, "acme renewal march", {
    n: 1,
    expandLinks: true,
  });
  expect(paths(expanded)).toEqual(["notes/acme-renewal.md", "vendors/globex.md"]);
  expect(expanded[1]).toMatchObject({ expansion: true });
  db.close();
});

test("n caps the result and an empty vault returns nothing", async () => {
  const { db } = await indexed(EVAL_VAULT);
  expect(await hybridSearch(db, embedder, "acme", { n: 2 })).toHaveLength(2);
  await expect(hybridSearch(db, embedder, "acme", { n: 0 })).rejects.toThrow(/positive integer/);
  db.close();

  const { db: empty } = await indexed({});
  expect(await hybridSearch(empty, embedder, "acme")).toEqual([]);
  empty.close();
});

test("searching with a different embedder than the index was built with is refused", async () => {
  const { db } = await indexed(EVAL_VAULT);
  await expect(hybridSearch(db, stubEmbedder("other-model", 256), "acme")).rejects.toThrow(
    /token-overlap-v1.*other-model.*doctor/s,
  );
  db.close();
});

test("vault search prints ranked hits", async () => {
  const { root, db } = await indexed(EVAL_VAULT);
  db.close();
  const log = spyOn(console, "log").mockImplementation(() => {});
  const err = spyOn(console, "error").mockImplementation(() => {});
  const said = () => log.mock.calls.flat().join("\n");
  try {
    // --lexical throughout: it is the embedder `indexed` built these vectors
    // with, and it keeps the suite off any provider
    expect(await main(["search", "INV-2031", "--lexical", "--vault", root])).toBe(0);
    expect(said()).toContain("ledger/invoice-2031.md");

    // the flag and its value are consumed wherever they sit, and never become
    // query words — this must find the same note as the line above
    log.mockClear();
    expect(await main(["search", "--vault", root, "--lexical", "INV-2031"])).toBe(0);
    expect(said()).toContain("ledger/invoice-2031.md");

    log.mockClear();
    expect(await main(["search", "🙂", "--lexical", "--vault", root])).toBe(0); // no signal either side
    expect(said()).toBe("no matches");

    // an unindexed vault is not the same answer as "nothing matched"
    log.mockClear();
    const { root: emptyRoot } = await indexed({});
    expect(await main(["search", "acme", "--lexical", "--vault", emptyRoot])).toBe(0);
    expect(said()).toMatch(/not indexed.*reindex/);

    expect(await main(["search", "--lexical", "--vault", root])).toBe(1); // no query at all
    // an error is a message, not a stack trace
    err.mockClear();
    expect(await main(["search", "acme", "--sideways", "--lexical", "--vault", root])).toBe(1);
    expect(err.mock.calls.flat().join("\n")).toContain("--sideways");
  } finally {
    log.mockRestore();
    err.mockRestore();
  }
});

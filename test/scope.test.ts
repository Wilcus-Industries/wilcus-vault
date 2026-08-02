// Per-agent scopes (DESIGN.md § Scopes and context). Policy is fixed at
// `open()`, identity travels per call, and every enforcement point lives in the
// library so no caller re-implements one. Deterministic embedder, fake
// deciders, no network.
import { test, expect, afterAll } from "bun:test";
import { TokenOverlapEmbedder } from "../src/embed";
import { compileScopes, scopeFor, type Permission, type ScopePolicy } from "../src/scope";
import { gatePrompt, type DeciderInput } from "../src/gate";
import { open, type Cutoffs, type Decider, type Vault } from "../src/vault";
import { makeVault, cleanupVaults } from "./vault-fixture";

const embedder = new TokenOverlapEmbedder();
const CUTOFFS: Cutoffs = { distanceCeiling: 0.9, bm25Ceiling: 0 };

const opened: Vault[] = [];
// Handles first, fixture directories second — removing a vault out from under a
// live sqlite handle is how the other suites stay clean.
afterAll(() => {
  for (const v of opened.splice(0)) v.close();
  cleanupVaults();
});

/**
 * The policy DESIGN.md § Scopes and context prints: reads everywhere, writes
 * everywhere except `ledger/`, which stays readable.
 */
const DESIGN_EXAMPLE: ScopePolicy = {
  "core/scheduler": [
    { prefix: "", read: true, write: true },
    { prefix: "ledger/", write: false },
  ],
};

/** `notes/` is this agent's own namespace; `ledger/` it may only read. */
const POLICY: ScopePolicy = {
  ...DESIGN_EXAMPLE,
  "core/notes": [
    { prefix: "notes/", read: true, write: true },
    { prefix: "ledger/", read: true },
  ],
};

const NOTES = { agent: "core/notes", source: "task-42" };
const SCHEDULER = { agent: "core/scheduler" };

const VAULT: Record<string, string> = {
  "notes/acme-renewal.md":
    "---\ntype: customer\n---\n# Acme renewal\n\nThe Acme renewal closes in March at the agreed renewal pricing.\n",
  "notes/support-rota.md": "# Support rota\n\nWho carries the pager each week.\n",
  "notes/hub.md":
    "# Renewal hub\n\nThe renewal hub: [[secret/plans]] and [[notes/support-rota]].\n",
  "ledger/q3.md": "# Q3 ledger\n\nQ3 revenue by account, including the Acme renewal line.\n",
  "ledger-archive/q3.md": "# Archived Q3\n\nSuperseded Acme renewal numbers.\n",
  "secret/plans.md": "# Secret plans\n\nThe Acme renewal pricing nobody else may read.\n",
  "root-note.md": "# Root note\n\nAt the vault root, about the Acme renewal.\n",
};

const CANDIDATE = {
  title: "Acme renewal 2026",
  type: "customer",
  namespace: "notes",
  body: "The Acme renewal closes in March 2026 at the agreed renewal pricing.\n",
};

async function openVault(
  scopes?: ScopePolicy,
  decider: Decider = async () => ({ action: "create" }),
  files: Record<string, string> = VAULT,
): Promise<Vault> {
  const v = open(makeVault(files), { embedder, gate: { decider, cutoffs: CUTOFFS }, scopes });
  opened.push(v);
  await v.reindex();
  return v;
}

/** The resolver itself, without a vault around it. */
function checker(policy: ScopePolicy, agent: string): (perm: Permission, path: string) => boolean {
  const scope = scopeFor(compileScopes(policy), { agent });
  return (perm, path) => scope.may(perm, path);
}

const paths = (hits: { path: string }[]): string[] => hits.map((h) => h.path);

test("no policy is allow-all: every call works, with a context or without one", async () => {
  const v = await openVault(undefined);

  expect(paths(await v.search("acme renewal"))).toContain("secret/plans.md");
  expect(paths(await v.search("acme renewal", { ctx: { agent: "anyone" } }))).toContain(
    "secret/plans.md",
  );
  expect((await v.get("secret/plans.md"))?.title).toBe("Secret plans");
  expect((await v.get("secret/plans.md", { agent: "anyone" }))?.title).toBe("Secret plans");
  expect(v.list()).toContain("secret/plans.md");
  expect(v.list("secret", { agent: "anyone" })).toEqual(["secret/plans.md"]);
  // an agent no policy ever named may still write, because there is no policy
  const r = await v.propose({ ...CANDIDATE, namespace: "ledger" }, { agent: "anyone" });
  expect(r).toMatchObject({ action: "create", path: "ledger/acme-renewal-2026.md" });
});

test("open refuses a policy that contradicts itself", async () => {
  const root = makeVault({});
  const refuse = (scopes: ScopePolicy): (() => Vault) => () => open(root, { embedder, scopes });

  // `ledger` and `ledger/` are one prefix once normalized, so these two rules
  // are two answers to the same question — pick a winner and the policy means
  // whatever the array order happened to be.
  expect(refuse({ a: [{ prefix: "ledger", read: true }, { prefix: "ledger/", read: false }] })).toThrow(
    /both specify read/,
  );
  expect(
    refuse({ a: [{ prefix: "", read: true, write: true }, { prefix: "/", write: true }] }),
  ).toThrow(/both specify write/);
  // Same prefix, different permissions, is not a contradiction.
  expect(refuse({ a: [{ prefix: "x/", read: true }, { prefix: "x/", write: false }] })).not.toThrow();

  // A write-blind agent never sees its own notes as `similar`, so every propose
  // lands as `create`: a duplicate factory, not a scope.
  expect(refuse({ a: [{ prefix: "notes/", write: true }] })).toThrow(/writable but not readable/);
  // ...including when the write is inherited from a shorter rule.
  expect(
    refuse({ a: [{ prefix: "", read: true, write: true }, { prefix: "ledger/", read: false }] }),
  ).toThrow(/writable but not readable/);
  // Read-only is fine in the other direction: `ledger/` above is exactly that.
  expect(refuse(POLICY)).not.toThrow();
});

test("a prefix is a namespace, matched on segment boundaries only", () => {
  const may = checker(DESIGN_EXAMPLE, "core/scheduler");
  // `ledger/` is the write-denied subtree; its sibling namespace is not.
  expect(may("write", "ledger/q3.md")).toBe(false);
  expect(may("write", "ledger/2026/q3.md")).toBe(false);
  expect(may("write", "ledger-archive/q3.md")).toBe(true);
  expect(may("write", "ledgers/q3.md")).toBe(true);
  // `""` is the root rule: it matches every note, and a root-level note (no `/`
  // in its path) matches only it.
  expect(may("write", "root-note.md")).toBe(true);
  expect(checker({ a: [{ prefix: "ledger/", read: true, write: true }] }, "a")("read", "q3.md")).toBe(
    false,
  );
});

test("resolution is per permission, longest prefix wins, unspecified defers", () => {
  const may = checker(DESIGN_EXAMPLE, "core/scheduler");
  // The DESIGN example: reads everywhere, writes everywhere but `ledger/`,
  // which stays readable because the longer rule specifies only `write`.
  expect(may("read", "ledger/q3.md")).toBe(true);
  expect(may("write", "ledger/q3.md")).toBe(false);
  expect(may("read", "secret/plans.md")).toBe(true);
  expect(may("write", "secret/plans.md")).toBe(true);

  // Three deep, the middle rule specifying one half of the answer and
  // deferring the other to the root rule.
  const deep = checker(
    {
      a: [
        { prefix: "", read: true, write: false },
        { prefix: "a/", write: true },
        { prefix: "a/b/", read: false, write: false },
      ],
    },
    "a",
  );
  expect([deep("read", "elsewhere.md"), deep("write", "elsewhere.md")]).toEqual([true, false]);
  expect([deep("read", "a/x.md"), deep("write", "a/x.md")]).toEqual([true, true]);
  expect([deep("read", "a/b/x.md"), deep("write", "a/b/x.md")]).toEqual([false, false]);

  // Nothing specifies ⇒ denied, both ways.
  const thin = checker({ a: [{ prefix: "notes/", read: true }] }, "a");
  expect(thin("write", "notes/x.md")).toBe(false);
  expect(thin("read", "other/x.md")).toBe(false);
});

test("a policy in force fails closed: no context, an unknown agent, or `{}`", async () => {
  const v = await openVault(POLICY);
  // Silence is how an orchestrator typo makes an agent re-create the memory it
  // thinks it lost — so every one of these throws instead.
  await expect(v.search("acme")).rejects.toThrow(/VaultContext/);
  await expect(v.get("notes/hub.md")).rejects.toThrow(/VaultContext/);
  expect(() => v.list()).toThrow(/VaultContext/);
  await expect(v.propose(CANDIDATE)).rejects.toThrow(/VaultContext/);

  await expect(v.search("acme", { ctx: { agent: "core/typo" } })).rejects.toThrow(/has no scope/);
  await expect(v.get("notes/hub.md", { agent: "core/typo" })).rejects.toThrow(/has no scope/);
  expect(() => v.list(undefined, { agent: "core/typo" })).toThrow(/has no scope/);
  await expect(v.propose(CANDIDATE, { agent: "core/typo" })).rejects.toThrow(/has no scope/);

  // `{}` and `undefined` sit on opposite sides of the fail-closed line.
  const empty = await openVault({});
  await expect(empty.get("notes/hub.md", NOTES)).rejects.toThrow(/has no scope/);
});

test("search returns only readable hits, filtered over the over-fetched set", async () => {
  const v = await openVault(POLICY);
  const query = "acme renewal pricing";

  // Unscoped, the vault answers with notes from every namespace.
  expect(paths(await v.search(query, { ctx: SCHEDULER, n: 10 }))).toEqual(
    expect.arrayContaining(["secret/plans.md", "ledger-archive/q3.md", "root-note.md"]),
  );

  const scoped = await v.search(query, { ctx: NOTES, n: 10 });
  expect(paths(scoped).length).toBeGreaterThan(0);
  for (const path of paths(scoped)) expect(path).toMatch(/^(notes|ledger)\//);
  expect(paths(scoped)).toContain("ledger/q3.md"); // readable, just not writable
});

test("the scope filter runs before the cap, so N unreadable hits do not eat the answer", async () => {
  // Three short, dense notes the agent cannot read outrank the one it can on
  // both signals; the over-fetch (3×N) is what still leaves room for it.
  const files: Record<string, string> = {
    "notes/acme-renewal.md":
      "# Acme renewal\n\nThe Acme renewal closes in March. Contract, notice periods, uplift caps,\n" +
      "escalation ladder, finance sign off, procurement thresholds, quarterly review cadence,\n" +
      "support handover, pager rota, and where an approved pricing exception is recorded.\n",
  };
  for (const n of [1, 2, 3]) files[`secret/plan-${n}.md`] = "# Acme renewal pricing\n\nAcme renewal pricing.\n";
  const v = await openVault(POLICY, undefined, files);

  const query = "acme renewal pricing";
  expect(paths(await v.search(query, { ctx: SCHEDULER, n: 1 }))).toEqual(["secret/plan-1.md"]);
  // "up to N": the readable note is outside the 3×1 over-fetch, so a scoped
  // agent sees fewer hits — accepted, and documented.
  expect(await v.search(query, { ctx: NOTES, n: 1 })).toEqual([]);
  // Widen N and the over-fetch reaches past the crowd to the note it may read.
  expect(paths(await v.search(query, { ctx: NOTES, n: 3 }))).toEqual(["notes/acme-renewal.md"]);
});

test("expandLinks neighbours pass the same read filter", async () => {
  const v = await openVault(POLICY);
  // Cutoffs keep the direct set to the hub note itself, so both of its
  // neighbours are inside the expansion's own cap of N.
  const expansions = async (ctx: { agent: string }): Promise<string[]> =>
    paths(
      (
        await v.search("renewal hub", { ctx, n: 2, cutoffs: CUTOFFS, expandLinks: true })
      ).filter((h) => h.expansion),
    );

  // A forbidden note is one wikilink away from a hit the agent may read.
  expect(await expansions(SCHEDULER)).toContain("secret/plans.md");
  const scoped = await expansions(NOTES);
  expect(scoped).toContain("notes/support-rota.md");
  expect(scoped).not.toContain("secret/plans.md");
});

test("get: an unreadable note is null, exactly like an absent one", async () => {
  const v = await openVault(POLICY);
  // A scope is not an existence oracle: these two answers must be identical.
  expect(await v.get("secret/plans.md", NOTES)).toBeNull();
  expect(await v.get("secret/nothing-here.md", NOTES)).toBeNull();
  expect((await v.get("secret/plans.md", SCHEDULER))?.title).toBe("Secret plans");
  expect((await v.get("ledger/q3.md", NOTES))?.title).toBe("Q3 ledger"); // readable, unwritable
});

test("list is filtered to what the agent may read", async () => {
  const v = await openVault(POLICY);
  expect(v.list(undefined, NOTES)).toEqual([
    "ledger/q3.md",
    "notes/acme-renewal.md",
    "notes/hub.md",
    "notes/support-rota.md",
  ]);
  expect(v.list("secret", NOTES)).toEqual([]);
  expect(v.list(undefined, SCHEDULER)).toContain("secret/plans.md");
});

test("propose refuses an unwritable namespace before the decider runs", async () => {
  const seen: DeciderInput[] = [];
  const v = await openVault(POLICY, async (input) => {
    seen.push(input);
    return { action: "create" };
  });

  // Fail fast: no model spend on a doomed write.
  await expect(v.propose({ ...CANDIDATE, namespace: "ledger" }, NOTES)).rejects.toThrow(
    /may not write/,
  );
  await expect(v.propose({ ...CANDIDATE, namespace: undefined }, NOTES)).rejects.toThrow(
    /may not write/,
  );
  expect(seen).toEqual([]);

  expect(await v.propose(CANDIDATE, NOTES)).toMatchObject({
    action: "create",
    path: "notes/acme-renewal-2026.md",
  });
  expect(seen.length).toBe(1);
});

test("only readable notes feed the decider, and unwritable ones are marked read-only", async () => {
  let input: DeciderInput | undefined;
  const v = await openVault(POLICY, async (seen) => {
    input = seen;
    return { action: "discard" };
  });
  await v.propose(CANDIDATE, NOTES);

  const similar = input!.similar;
  expect(similar.length).toBeGreaterThan(1);
  // An agent must not have another agent's note bodies quoted back to it.
  expect(paths(similar.map((s) => s.note))).not.toContain("secret/plans.md");
  expect(paths(similar.map((s) => s.note))).toContain("ledger/q3.md");
  expect(similar.find((s) => s.note.path === "ledger/q3.md")?.readOnly).toBe(true);
  expect(similar.find((s) => s.note.path === "notes/acme-renewal.md")?.readOnly).toBeUndefined();

  const prompt = gatePrompt(input!);
  expect(prompt).toMatch(/ledger\/q3\.md.*read-only/);
  expect(prompt).not.toMatch(/notes\/acme-renewal\.md.*read-only/);
  expect(prompt).not.toContain("nobody else may read");
});

test("a decision targeting an unwritable note falls back to create", async () => {
  const v = await openVault(POLICY, async () => ({
    action: "update",
    target: "ledger/q3.md",
    body: "# Q3 ledger\n\nRewritten by an agent that may not write here.\n",
  }));
  const before = (await v.get("ledger/q3.md", NOTES))!.hash;

  // Like a target that failed check-and-write twice: the candidate always lands.
  const r = await v.propose(CANDIDATE, NOTES);
  expect(r).toMatchObject({ action: "create", path: "notes/acme-renewal-2026.md", fellBack: true });
  expect((await v.get("ledger/q3.md", NOTES))!.hash).toBe(before);
});

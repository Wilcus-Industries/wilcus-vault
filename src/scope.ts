// Per-agent scopes (DESIGN.md § Scopes and context). Policy is fixed at
// `open()`, identity travels per call. One resolver lives here and every
// enforcement point — search, get, list, the write gate — asks it, so a
// namespace prefix cannot come to mean two different things in two files.
//
// Scopes are advisory containment at the library API, not security: any process
// with filesystem access can read or edit the notes directly. That is the
// files-are-truth contract, not a hole in it.
import type { VaultContext } from "./gate";
import { safe } from "./term";

/**
 * A namespace subtree and what it grants. `prefix` matches on **segment
 * boundaries only** (`ledger/` matches `ledger/q3.md`, never
 * `ledger-archive/q3.md`); `""` is the root rule and matches every note. A
 * permission left undefined is not "denied" — it defers to the next-shorter
 * matching rule.
 */
export type ScopeRule = { prefix: string; read?: boolean; write?: boolean };

/** agent name → its rules. Absent from `VaultOptions` means allow-all. */
export type ScopePolicy = Record<string, ScopeRule[]>;

export type Permission = "read" | "write";

/** A validated policy: agent → normalized rules, longest prefix first. */
export type CompiledPolicy = Map<string, ScopeRule[]>;

/**
 * What one call may touch, resolved once from the policy and the caller's
 * context — which it carries, because the write gate needs the same object for
 * provenance.
 */
export type Scope = {
  readonly ctx?: VaultContext;
  may(permission: Permission, path: string): boolean;
  /**
   * The read check as a SQL boolean over `n.path`, for the two filters that
   * have to run inside the query (search's over-fetched set and its link
   * expansion). Generated from the same normalized rules as `may`, so the two
   * cannot drift — `test/scope.test.ts` holds them to the same answers.
   */
  readonly readSql: { sql: string; params: (string | number)[] };
};

/** No policy: everything allowed, and the query filters compile away to `1`. */
export const ALLOW_ALL: Scope = {
  may: () => true,
  readSql: { sql: "1", params: [] },
};

/**
 * `ledger`, `/ledger/` and `ledger/` all name the subtree `ledger/`; `""`, `/`
 * and undefined name the vault root. Slashes at either end are decoration —
 * stored paths carry neither — and the trailing one is what makes a prefix
 * match whole segments instead of half a namespace name.
 */
export function normalizePrefix(prefix?: string): string {
  const ns = prefix?.replace(/^\/+|\/+$/g, "") ?? "";
  return ns === "" ? "" : `${ns}/`;
}

/**
 * Validate and normalize a policy at `open()`, or null when there is none.
 * Two rules that answer the same question two ways are refused rather than
 * resolved by array order, and so is a subtree an agent could write but not
 * read — a write-blind agent never sees its own notes as `similar`, so every
 * propose lands as `create`: a duplicate factory, not a scope.
 */
export function compileScopes(policy?: ScopePolicy): CompiledPolicy | null {
  if (policy === undefined) return null;
  const compiled: CompiledPolicy = new Map();
  for (const [agent, given] of Object.entries(policy)) {
    // Annotated on the variable, not the arrow: that is what lets TypeScript
    // read a call to it as "control flow ends here" and narrow what follows.
    const bad: (why: string) => never = (why) => {
      throw new Error(`vault: scope policy for ${JSON.stringify(safe(agent))}: ${why}`);
    };
    // A policy is operator configuration — it arrives from a config file, an
    // orchestrator, another process's JSON — so it is checked, not trusted to
    // be what its type says. `read: "false"` is *truthy*: unchecked, a policy
    // that meant to deny would grant, which is the one direction an allowlist
    // must never fail in.
    if (!Array.isArray(given)) bad("must be a list of rules");
    for (const rule of given as unknown[]) {
      if (rule === null || typeof rule !== "object") {
        bad(`a rule must be {prefix, read?, write?}, got ${JSON.stringify(rule)?.slice(0, 60)}`);
      }
      const r = rule as Partial<ScopeRule>;
      if (typeof r.prefix !== "string") bad("a rule's prefix must be a string");
      for (const permission of PERMISSIONS) {
        if (r[permission] !== undefined && typeof r[permission] !== "boolean") {
          bad(
            `${permission} must be true, false or absent, in rule ` +
              JSON.stringify(safe(r.prefix)),
          );
        }
      }
      // Stored paths are canonical, so a prefix that is not matches nothing —
      // and a deny rule that matches nothing is a deny that never fires.
      // Refused rather than normalized: `ledger/../x` is not a typo we get to
      // guess the intent of.
      const segments = r.prefix.replace(/^\/+|\/+$/g, "").split("/");
      const canonical = segments.every(
        (s, i) => s !== "." && s !== ".." && (s !== "" || i === 0),
      );
      if (r.prefix !== "" && !canonical) {
        bad(`${JSON.stringify(safe(r.prefix))} is not a canonical namespace`);
      }
    }
    const rules = given
      .map((r) => ({ ...r, prefix: normalizePrefix(r.prefix) }))
      // Longest first: resolution is "longest matching prefix that specifies
      // this permission wins", which is a first-match scan over this order.
      .sort((a, b) => b.prefix.length - a.prefix.length);

    const seen = new Map<string, ScopeRule>();
    for (const rule of rules) {
      const prior = seen.get(rule.prefix);
      for (const permission of PERMISSIONS) {
        if (rule[permission] !== undefined && prior?.[permission] !== undefined) {
          bad(`two rules for ${JSON.stringify(safe(rule.prefix))} both specify ${permission}`);
        }
      }
      seen.set(rule.prefix, {
        prefix: rule.prefix,
        read: rule.read ?? prior?.read,
        write: rule.write ?? prior?.write,
      });
    }
    // Resolution only changes at a rule's own prefix, so checking each one
    // covers every path under it.
    for (const { prefix } of rules) {
      if (resolve(rules, "write", prefix) && !resolve(rules, "read", prefix)) {
        bad(
          `${JSON.stringify(safe(prefix))} is writable but not readable — an agent that cannot ` +
            `see its own notes re-creates them on every propose`,
        );
      }
    }
    compiled.set(agent, rules);
  }
  return compiled;
}

/**
 * The scope in force for one call. A policy fails closed: a call with no
 * `VaultContext`, or one naming an agent the policy never mentions, throws
 * rather than returning a silently empty result — silence is how an
 * orchestrator typo makes an agent re-create the memory it thinks it lost.
 */
export function scopeFor(policy: CompiledPolicy | null, ctx?: VaultContext): Scope {
  if (policy === null) return { ...ALLOW_ALL, ctx };
  if (ctx === undefined) {
    throw new Error(
      "vault: this vault has a scope policy, so every call needs a VaultContext — {agent: '<name>'}",
    );
  }
  const rules = policy.get(ctx.agent);
  if (rules === undefined) {
    throw new Error(`vault: agent ${JSON.stringify(safe(ctx.agent))} has no scope in this vault`);
  }
  return {
    ctx,
    may: (permission, path) => resolve(rules, permission, path),
    readSql: readSql(rules),
  };
}

const PERMISSIONS: readonly Permission[] = ["read", "write"];

/** Longest matching prefix that specifies this permission decides; none ⇒ denied. */
function resolve(rules: ScopeRule[], permission: Permission, path: string): boolean {
  for (const rule of rules) {
    if (!path.startsWith(rule.prefix)) continue;
    const allowed = rule[permission];
    if (allowed !== undefined) return allowed;
  }
  return false;
}

/**
 * `resolve(rules, "read", n.path)` as SQL: a CASE over the rules in the same
 * longest-first order, so the first prefix that matches and specifies `read`
 * decides. `substr` rather than `like`, whose `%` and `_` would have to be
 * escaped out of a namespace name — and SQLite measures the prefix itself
 * (`length(?)`), because it counts characters where JS counts UTF-16 code
 * units: a JS length for `🔒secret/` is one too many, the substr never equals
 * the prefix, and a deny rule quietly stops denying.
 */
function readSql(rules: ScopeRule[]): Scope["readSql"] {
  const params: (string | number)[] = [];
  const whens = rules
    .filter((r) => r.read !== undefined)
    .map((r) => {
      params.push(r.prefix, r.prefix);
      return `when substr(n.path, 1, length(?)) = ? then ${r.read ? 1 : 0}`;
    });
  // No rule mentions `read` at all, so nothing is readable.
  if (whens.length === 0) return { sql: "0", params: [] };
  return { sql: `case ${whens.join(" ")} else 0 end`, params };
}

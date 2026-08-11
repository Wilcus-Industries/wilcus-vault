#!/usr/bin/env bun
import { resolve } from "node:path";
import { openDb, dbPath } from "./db";
import { clusters } from "./consolidate";
import { fetchDecider } from "./decide";
import { getDiscard, listDiscards, restoreDiscard } from "./discards";
import { reindex, type IndexStats } from "./indexer";
import { doctor, type DoctorReport } from "./doctor";
import { FetchEmbedder, TokenOverlapEmbedder } from "./embed";
import { hybridSearch } from "./search";
import { open } from "./vault";
import { watch } from "./watch";
import { printable, safe } from "./term";

const USAGE = `vault <command> [options]

  reindex             index new and changed notes
  doctor [--rebuild]  check and repair the index (--rebuild: from scratch)
  search <query>      hybrid search: one line per hit — score, path, title
  watch               index every change as it is saved, until interrupted
  consolidate         report near-duplicate clusters (needs --ceiling)
  discards list       the discard log, newest first — one line per entry
  discards show <n>   one entry in full, as JSON
  discards restore <n>  re-propose entry n through the write gate

  --vault <dir>       vault root (default: the current directory)
  --lexical           embed offline, without a provider (see below)
  --ceiling <d>       cosine distance two notes must be within to cluster
  --help, -h          this text
  --                  end of flags, so a search query may start with a dash

vault consolidate is report-only: one line per cluster — widest internal
distance, then the member paths, with the ones that span namespaces flagged
(those are never merged). Merging itself needs an injected merger, so it
lives in the library API; this is the pass that tells you which ceiling your
embedder calls a duplicate.

vault discards reviews <root>/.discarded.log — every candidate the write
gate refused, kept whole. Entries are numbered newest-first across the
size-rotated .discarded.N.log files, so a rotation can renumber between a
list and a restore: show <n> before restoring. restore feeds the candidate
back through the gate against the current vault, which needs --ceiling and
a chat model: set VAULT_DECIDE_MODEL (and VAULT_DECIDE_ENDPOINT /
VAULT_DECIDE_API_KEY for a provider that is not the local Ollama). The
log's first write adds .discarded.log* to the vault's .gitignore, once,
so refused note bodies never ride into git history.

Notes embed on a local Ollama unless VAULT_EMBED_* names another
OpenAI-compatible provider: run \`ollama pull all-minilm\` once, or pass
--lexical for a deterministic bag-of-tokens embedder that needs nothing
running — no network, and no semantics either. They are different vector
spaces: switching costs a full re-embed, which reindex, doctor and watch do
on their next pass and search refuses to go without.

Exit code 0 on success, 1 on error — and 1 from doctor when it found links
only a human can fix: broken (nothing to point at) or ambiguous (a bare
[[stem]] several notes answer to — qualify it as [[folder/stem]]).`;

export async function main(argv: string[]): Promise<number> {
  try {
    return await run(argv);
  } catch (e) {
    // A stack trace is not an error message. Anything thrown from here down
    // (a bad flag, an index built by another embedder, a provider's response
    // body quoted back) reaches the user as the sentence it was written as —
    // and only as that, never as escape codes the terminal would act on.
    console.error(printable(e));
    return 1;
  }
}

/**
 * Flags are consumed with their values wherever they appear — before or after
 * the command — and everything else is a positional word, the first of which is
 * the command. `--` ends flag parsing, so a query may start with dashes.
 */
function parseArgs(argv: string[]): {
  root: string;
  words: string[];
  rebuild: boolean;
  lexical: boolean;
  ceiling: number | undefined;
  help: boolean;
} {
  const words: string[] = [];
  let root = process.cwd();
  let rebuild = false;
  let lexical = false;
  let ceiling: number | undefined;
  let help = false;
  let literal = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (literal || (!arg.startsWith("--") && arg !== "-h")) {
      words.push(arg);
    } else if (arg === "--") {
      literal = true;
    } else if (arg === "--help" || arg === "-h") {
      help = true;
    } else if (arg === "--rebuild") {
      rebuild = true;
    } else if (arg === "--lexical") {
      lexical = true;
    } else if (arg === "--ceiling") {
      const value = argv[++i];
      ceiling = Number(value);
      // A missing value would swallow the next flag; a nonsense one would reach
      // the query as NaN, where every comparison is false and the pass would
      // report "nothing to merge" about a vault it never really looked at.
      if (
        value === undefined ||
        value.startsWith("--") ||
        !Number.isFinite(ceiling) ||
        ceiling < 0 ||
        ceiling > 2
      ) {
        throw new Error(`--ceiling needs a cosine distance in 0..2\n\n${USAGE}`);
      }
    } else if (arg !== "--vault") {
      throw new Error(`unknown flag ${arg}\n\n${USAGE}`);
    } else {
      const dir = argv[++i];
      // A missing value would otherwise swallow the next flag and index
      // whatever directory that names — or the cwd.
      if (dir === undefined || dir.startsWith("--")) {
        throw new Error(`--vault needs a directory\n\n${USAGE}`);
      }
      root = dir;
    }
  }
  return { root, words, rebuild, lexical, ceiling, help };
}

const COMMANDS = ["reindex", "doctor", "search", "watch", "consolidate", "discards"] as const;

async function run(argv: string[]): Promise<number> {
  const { root, words, rebuild, lexical, ceiling, help } = parseArgs(argv);
  // Help was asked for, so answering it is a success — `vault --help | less`
  // reads it, and a script checking the exit code is not told it failed.
  if (help) {
    console.log(USAGE);
    return 0;
  }
  const [command, ...rest] = words;
  if (!COMMANDS.includes(command as (typeof COMMANDS)[number])) {
    console.error(command === undefined ? USAGE : printable(`unknown command ${command}\n\n${USAGE}`));
    return 1;
  }
  // The CLI embeds in-process against the same embedder the library documents:
  // FetchEmbedder resolves its own configuration (VAULT_EMBED_*, else a local
  // Ollama), so no note text leaves the machine unless the environment points it
  // at a remote provider. `--lexical` is the escape hatch for a machine with no
  // daemon — deterministic, offline, and lexical only. Whichever it is, a bad
  // configuration throws here, before a database is opened, and `main` prints it.
  const embedder = lexical ? new TokenOverlapEmbedder() : new FetchEmbedder();

  if (command === "search") {
    const query = rest.join(" ");
    if (query === "") throw new Error(`search needs a query\n\n${USAGE}`);
    const db = openDb(dbPath(root));
    try {
      // "nothing here" and "nothing matched" are different answers, and only
      // one of them is the user's fault.
      if (db.query(`select 1 from notes limit 1`).get() === null) {
        console.log("vault is not indexed (run vault reindex)");
        return 0;
      }
      // No relevance cutoffs here: the ceiling that means "irrelevant" is a
      // property of the embedder — unknowable for whichever provider is
      // configured, and meaningless under --lexical (a one-word query is far
      // from every long note by construction) — so the CLI shows the ranking
      // and lets the reader judge. Library callers pass their own.
      const hits = await hybridSearch(db, embedder, query);
      console.log(
        hits.length === 0
          ? "no matches"
          // score first so the ranking reads down the page, then the note's
          // identity, then what it is called
          : hits.map((h) => safe(`${h.score.toFixed(4)}  ${h.path} — ${h.title}`)).join("\n"),
      );
    } finally {
      db.close();
    }
    return 0;
  }

  if (command === "reindex") {
    const db = openDb(dbPath(root));
    try {
      console.log(`indexed ${summary(await reindex(db, root, embedder))}`);
    } finally {
      db.close();
    }
    return 0;
  }

  if (command === "consolidate") {
    // Report-only, deliberately: a merge needs an injected merger (an LLM), and
    // the CLI wires no model — the same reason there is no `vault propose`. The
    // library API is the write path; this is the pass you run first, because the
    // ceiling that means "duplicate" is a property of your embedder.
    if (ceiling === undefined) throw new Error(`consolidate needs --ceiling\n\n${USAGE}`);
    const db = openDb(dbPath(root));
    try {
      // Discovery reads the index, so bring it up to the files first — and say
      // so, like `watch` does, since a first run may be doing the whole vault.
      console.log(`indexed ${summary(await reindex(db, root, embedder))}`);
      const found = clusters(db, ceiling);
      console.log(
        found.length === 0
          ? "no clusters under that ceiling"
          : found
              .map((c) =>
                // distance first so the closest duplicates read down the page,
                // then the flag for a cluster no merge would touch, then the
                // notes themselves
                safe(
                  `${c.distance.toFixed(4)}  ` +
                    `${c.namespace === null ? "cross-namespace  " : ""}${c.members.join(" ")}`,
                ),
              )
              .join("\n"),
      );
    } finally {
      db.close();
    }
    return 0;
  }

  if (command === "discards") {
    const [sub, arg] = rest;
    if (sub === "list") {
      const { entries, malformed } = listDiscards(root);
      const lines = entries.map((e) =>
        // number first (it is what show/restore take), then when, then what,
        // then why it is in this log at all
        safe(`[${e.n}] ${e.at}  ${e.candidate.title} — ${e.decision?.action ?? e.reason ?? "?"}`),
      );
      // said, not silently skipped: a truncated history reading as complete is
      // how a discarded note gets forgotten twice
      if (malformed > 0) lines.push(`(${malformed} unreadable line${malformed === 1 ? "" : "s"} skipped)`);
      console.log(lines.length === 0 ? "no discards" : lines.join("\n"));
      return 0;
    }
    const n = Number(arg);
    if ((sub !== "show" && sub !== "restore") || !Number.isInteger(n) || n < 1) {
      throw new Error(`discards needs list, show <n> or restore <n>\n\n${USAGE}`);
    }
    if (sub === "show") {
      const entry = getDiscard(root, n);
      if (entry === null) throw new Error(`discards: no entry ${n} — run vault discards list`);
      // JSON escaping keeps note-controlled control characters off the terminal
      console.log(JSON.stringify(entry, null, 2));
      return 0;
    }
    // restore: back through the gate — the only write door — against current
    // vault state. The ceiling is mandatory for the same reason propose's
    // cutoffs are: without one, "most similar" is only "least unrelated".
    if (ceiling === undefined) throw new Error(`discards restore needs --ceiling\n\n${USAGE}`);
    // Both constructed before the database is touched, so a missing
    // VAULT_DECIDE_MODEL is one sentence, not a half-opened restore.
    const decider = fetchDecider();
    const v = open(root, { embedder, gate: { decider, cutoffs: { distanceCeiling: ceiling } } });
    try {
      // The gate searches the index, so bring it up to the files first — and
      // say so, like consolidate does.
      console.log(`indexed ${summary(await v.reindex())}`);
      const r = await restoreDiscard(v, n);
      console.log(
        safe(`${r.action}${r.path === undefined ? "" : `  ${r.path}`}${r.fellBack ? " (fell back)" : ""}`),
      );
    } finally {
      v.close();
    }
    return 0;
  }

  if (command === "watch") {
    const db = openDb(dbPath(root));
    try {
      // Start from a current index: edits made while nothing was watching are
      // picked up here rather than silently missed.
      console.log(`indexed ${summary(await reindex(db, root, embedder))}`);
      const watcher = watch(db, root, embedder, {
        onChange: (paths, stats) => {
          // An editor rewriting identical bytes is not news; a change is.
          if (stats.added || stats.updated || stats.removed) {
            console.log(safe(`${paths.join(" ")} — ${summary(stats)}`));
          }
        },
      });
      console.log(`watching ${resolve(root)} — press ctrl-c to stop`);
      await interrupted();
      // The pass in flight still owns the database handle `finally` closes.
      await watcher.close();
    } finally {
      db.close();
    }
    return 0;
  }

  const report = await doctor(root, embedder, { rebuild });
  print(report);
  // Drift is repaired; links a human has to fix are not, so say so in the
  // exit code. (Orphans and malformed frontmatter are notes, not damage.)
  const unrepaired = report.brokenLinks.length + report.ambiguousLinks.length;
  return unrepaired === 0 ? 0 : 1;
}

const summary = (s: IndexStats): string =>
  `${s.added} new, ${s.updated} changed, ${s.removed} removed, ${s.unchanged} unchanged` +
  (s.reembedded ? " (re-embedded: model or dims changed)" : "") +
  // a stem collision this pass auto-qualified (or could not). Single-line and
  // `safe`d here: the stem and target are filenames, not our text, and the
  // watcher wraps whole lines in `safe`, which turns a newline into `?`.
  s.qualified
    .map((q) =>
      safe(
        q.target === null
          ? `; [[${q.stem}]] went ambiguous (root incumbent — ${q.skipped.length} linking note${q.skipped.length === 1 ? "" : "s"} left for doctor)`
          : `; [[${q.stem}]] qualified to [[${q.target}]] in ${q.rewritten.length} note${q.rewritten.length === 1 ? "" : "s"}` +
              (q.skipped.length > 0 ? ` (${q.skipped.length} skipped — see doctor)` : ""),
      ),
    )
    .join("");

/** `vault watch` runs until it is stopped; this is what "until" means. */
function interrupted(): Promise<void> {
  return new Promise((done) => {
    process.once("SIGINT", () => done());
    process.once("SIGTERM", () => done());
  });
}

function print(r: DoctorReport): void {
  const lines = [
    `reindexed ${r.stale.length} stale, purged ${r.missing.length} deleted` +
      (r.reembedded ? ", re-embedded all notes (model or dims changed)" : ""),
    ...(r.migratedDiscardLog ? ["moved .vault/discarded.log -> .discarded.log"] : []),
    // surfaced by normal maintenance, so the log is not write-only memory; an
    // empty one earns no line
    ...(r.discards.entries > 0
      ? [`discard log: ${r.discards.entries} entries (${r.discards.recent} recent)`]
      : []),
    ...r.brokenLinks.map((l) => `broken link:    ${l.from} -> [[${l.slug}]]`),
    // the candidates are the fix: qualify the link with one of these paths
    ...r.ambiguousLinks.map(
      (l) => `ambiguous link: ${l.from} -> [[${l.slug}]] (${l.candidates.join(", ")})`,
    ),
    ...r.malformed.map((p) => `malformed frontmatter: ${p}`),
    ...r.orphans.map((p) => `orphan: ${p}`),
  ];
  console.log(lines.map(safe).join("\n"));
}

if (import.meta.main) process.exit(await main(Bun.argv.slice(2)));

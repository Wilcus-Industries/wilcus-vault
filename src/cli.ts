#!/usr/bin/env bun
import { resolve } from "node:path";
import { openDb, dbPath } from "./db";
import { reindex, type IndexStats } from "./indexer";
import { doctor, type DoctorReport } from "./doctor";
import { TokenOverlapEmbedder } from "./embed";
import { hybridSearch } from "./search";
import { watch } from "./watch";

const USAGE = `vault <command> [options]

  reindex             index new and changed notes
  doctor [--rebuild]  check and repair the index (--rebuild: from scratch)
  search <query>      hybrid search: one line per hit — score, path, title
  watch               index every change as it is saved, until interrupted

  --vault <dir>       vault root (default: the current directory)
  --help, -h          this text
  --                  end of flags, so a search query may start with a dash

Exit code 0 on success, 1 on error — and 1 from doctor when it found links
only a human can fix: broken (nothing to point at) or ambiguous (a bare
[[stem]] several notes answer to — qualify it as [[folder/stem]]).`;

export async function main(argv: string[]): Promise<number> {
  try {
    return await run(argv);
  } catch (e) {
    // A stack trace is not an error message. Anything thrown from here down
    // (a bad flag, an index built by another embedder) reaches the user as
    // the sentence it was written as.
    console.error(e instanceof Error ? e.message : String(e));
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
  help: boolean;
} {
  const words: string[] = [];
  let root = process.cwd();
  let rebuild = false;
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
  return { root, words, rebuild, help };
}

const COMMANDS = ["reindex", "doctor", "search", "watch"] as const;

async function run(argv: string[]): Promise<number> {
  const { root, words, rebuild, help } = parseArgs(argv);
  // Help was asked for, so answering it is a success — `vault --help | less`
  // reads it, and a script checking the exit code is not told it failed.
  if (help) {
    console.log(USAGE);
    return 0;
  }
  const [command, ...rest] = words;
  if (!COMMANDS.includes(command as (typeof COMMANDS)[number])) {
    console.error(command === undefined ? USAGE : `unknown command ${command}\n\n${USAGE}`);
    return 1;
  }
  // The CLI embeds in-process: no daemon to start, and no note text leaves the
  // machine unless a library caller points FetchEmbedder at a remote provider.
  const embedder = new TokenOverlapEmbedder();

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
      // No relevance cutoffs here: against a bag-of-tokens embedder no fixed
      // cosine ceiling is meaningful (a one-word query is far from every long
      // note by construction), so the CLI shows the ranking and lets the
      // reader judge. Library callers with a real embedder pass their own.
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
  (s.reembedded ? " (re-embedded: model or dims changed)" : "");

/** `vault watch` runs until it is stopped; this is what "until" means. */
function interrupted(): Promise<void> {
  return new Promise((done) => {
    process.once("SIGINT", () => done());
    process.once("SIGTERM", () => done());
  });
}

/**
 * Every string the CLI echoes is note-controlled — a filename, a title, a link
 * target a human (or an LLM) wrote. A bare `\r` or an ESC sequence in one of
 * them would rewrite the line the terminal has already drawn, so control
 * characters print as `?`; the newlines the CLI itself writes are added after.
 */
const safe = (line: string): string => line.replace(/\p{Cc}/gu, "?");

function print(r: DoctorReport): void {
  const lines = [
    `reindexed ${r.stale.length} stale, purged ${r.missing.length} deleted` +
      (r.reembedded ? ", re-embedded all notes (model or dims changed)" : ""),
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

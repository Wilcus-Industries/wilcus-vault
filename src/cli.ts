#!/usr/bin/env bun
import { openDb, dbPath } from "./db";
import { reindex } from "./indexer";
import { doctor, type DoctorReport } from "./doctor";
import { TokenOverlapEmbedder } from "./embed";
import { hybridSearch } from "./search";

const USAGE = `vault <command> [options]

  reindex             index new and changed notes
  doctor [--rebuild]  check and repair the index (--rebuild: from scratch)
  search <query>      hybrid search over the indexed notes

  --vault <dir>       vault root (default: cwd)`;

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
 * Flags are consumed with their values wherever they appear; everything else is
 * a positional word. `--` ends flag parsing, so a query may start with dashes.
 */
function parseArgs(argv: string[]): { root: string; words: string[]; rebuild: boolean } {
  const words: string[] = [];
  let root = process.cwd();
  let rebuild = false;
  let literal = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (literal || !arg.startsWith("--")) {
      words.push(arg);
    } else if (arg === "--") {
      literal = true;
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
  return { root, words, rebuild };
}

async function run(argv: string[]): Promise<number> {
  const command = argv[0];
  if (command !== "reindex" && command !== "doctor" && command !== "search") {
    console.error(USAGE);
    return 1;
  }
  const { root, words, rebuild } = parseArgs(argv.slice(1));
  // The CLI embeds locally: no note text leaves the machine unless a library
  // caller wires up FetchEmbedder.
  const embedder = new TokenOverlapEmbedder();

  if (command === "search") {
    const query = words.join(" ");
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
          : hits
              .map(
                (h) =>
                  `${h.score.toFixed(4)}  ${h.path}${h.expansion ? " (link)" : ""} — ${h.title}`,
              )
              .join("\n"),
      );
    } finally {
      db.close();
    }
    return 0;
  }

  if (command === "reindex") {
    const db = openDb(dbPath(root));
    try {
      const s = await reindex(db, root, embedder);
      console.log(
        `indexed ${s.added} new, ${s.updated} changed, ${s.removed} removed, ${s.unchanged} unchanged` +
          (s.reembedded ? " (re-embedded: model or dims changed)" : ""),
      );
    } finally {
      db.close();
    }
    return 0;
  }

  const report = await doctor(root, embedder, { rebuild });
  print(report);
  // Drift is repaired; links a human has to fix are not, so say so in the
  // exit code. (Orphans and malformed frontmatter are notes, not damage.)
  const unrepaired =
    report.brokenLinks.length + report.ambiguousLinks.length + report.duplicateStems.length;
  return unrepaired === 0 ? 0 : 1;
}

function print(r: DoctorReport): void {
  const lines = [
    `reindexed ${r.stale.length} stale, purged ${r.missing.length} deleted` +
      (r.reembedded ? ", re-embedded all notes (model or dims changed)" : ""),
    ...r.brokenLinks.map((l) => `broken link:    ${l.from} -> [[${l.slug}]]`),
    ...r.ambiguousLinks.map((l) => `ambiguous link: ${l.from} -> [[${l.slug}]]`),
    ...r.duplicateStems.map((d) => `duplicate stem: ${d.slug} (${d.paths.join(", ")})`),
    ...r.malformed.map((p) => `malformed frontmatter: ${p}`),
    ...r.orphans.map((p) => `orphan: ${p}`),
  ];
  console.log(lines.join("\n"));
}

if (import.meta.main) process.exit(await main(Bun.argv.slice(2)));

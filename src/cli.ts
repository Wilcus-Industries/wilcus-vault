#!/usr/bin/env bun
import { openDb, dbPath } from "./db";
import { reindex } from "./indexer";
import { doctor, type DoctorReport } from "./doctor";
import { TokenOverlapEmbedder } from "./embed";

const USAGE = `vault <command> [options]

  reindex             index new and changed notes
  doctor [--rebuild]  check and repair the index (--rebuild: from scratch)

  --vault <dir>       vault root (default: cwd)`;

export async function main(argv: string[]): Promise<number> {
  const command = argv[0];
  if (command !== "reindex" && command !== "doctor") {
    console.log(USAGE);
    return 1;
  }
  const flag = argv.indexOf("--vault");
  const root = flag === -1 ? process.cwd() : argv[flag + 1];
  if (root === undefined) {
    console.log(USAGE);
    return 1;
  }
  // Only the deterministic embedder ships so far; the API one lands with search.
  const embedder = new TokenOverlapEmbedder();

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

  print(await doctor(root, embedder, { rebuild: argv.includes("--rebuild") }));
  return 0;
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

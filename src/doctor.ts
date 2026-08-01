// doctor is the safety net for "files are truth": it compares the index to the
// files, repairs what drifted, and reports what only a human can fix.
import type { Database } from "bun:sqlite";
import { renameSync, rmSync } from "node:fs";
import { openDb, dbPath } from "./db";
import { reindex, readNote, scanVault } from "./indexer";
import type { Embedder } from "./embed";

export type LinkProblem = { from: string; slug: string };

export type DoctorReport = {
  /** files whose indexed hash is wrong or missing (reindexed when repairing) */
  stale: string[];
  /** indexed rows whose file is gone (purged when repairing) */
  missing: string[];
  /** `[[slug]]` matching no note */
  brokenLinks: LinkProblem[];
  /** `[[slug]]` matching several notes — none of them wins */
  ambiguousLinks: LinkProblem[];
  /** notes with no *resolved* link in and no link out at all */
  orphans: string[];
  /** filename stems shared by several notes: every link to them is ambiguous */
  duplicateStems: { slug: string; paths: string[] }[];
  /** frontmatter the parser could not use — indexed anyway, whole file as body */
  malformed: string[];
  /** every note was re-embedded: the model or dims changed, or this was a rebuild */
  reembedded: boolean;
};

export type DoctorOptions = {
  /** reindex stale notes and purge deleted ones (default true) */
  repair?: boolean;
  /** index into a temp DB and rename it over index.db (instead of repairing) */
  rebuild?: boolean;
};

export async function doctor(
  root: string,
  embedder: Embedder,
  { repair = true, rebuild = false }: DoctorOptions = {},
): Promise<DoctorReport> {
  const path = dbPath(root);
  // Drift is measured before any repair, so the report says what was wrong.
  const drift = await withDb(path, (db) => diskDrift(db, root));
  let reembedded = rebuild; // a rebuild embeds every note from scratch
  if (rebuild) await rebuildIndex(root, embedder);
  else if (repair) {
    reembedded = (await withDb(path, (db) => reindex(db, root, embedder))).reembedded;
  }
  return { ...drift, ...(await withDb(path, graphReport)), reembedded };
}

async function withDb<T>(path: string, fn: (db: Database) => T | Promise<T>): Promise<T> {
  const db = openDb(path);
  try {
    return await fn(db);
  } finally {
    db.close();
  }
}

/**
 * What the files say that the index does not.
 * ponytail: re-reads and re-hashes every note, and a repair then reads them
 * again. Fine for a hand-written vault; gate on mtime if one gets huge.
 */
async function diskDrift(
  db: Database,
  root: string,
): Promise<{ stale: string[]; missing: string[] }> {
  const indexed = new Map(
    (db.query(`select path, hash from notes`).all() as { path: string; hash: string }[]).map(
      (r) => [r.path, r.hash],
    ),
  );
  const stale: string[] = [];
  for (const rel of scanVault(root)) {
    const note = await readNote(root, rel);
    if (note === null) continue; // deleted while we looked: it counts as missing
    if (indexed.get(rel) !== note.hash) stale.push(rel);
    indexed.delete(rel);
  }
  return { stale, missing: [...indexed.keys()].sort() };
}

function graphReport(db: Database): Omit<DoctorReport, "stale" | "missing" | "reembedded"> {
  const unresolved = db
    .query(
      `select n.path as "from", e.to_slug as slug,
              (select count(*) from notes m where m.slug = e.to_slug) as matches
       from edges e join notes n on n.id = e.from_id
       where e.to_id is null
       order by n.path, e.to_slug`,
    )
    .all() as (LinkProblem & { matches: number })[];
  const strip = ({ from, slug }: LinkProblem): LinkProblem => ({ from, slug });

  const duplicateStems: DoctorReport["duplicateStems"] = [];
  const dupes = db
    .query(
      `select slug, path from notes
       where slug in (select slug from notes group by slug having count(*) > 1)
       order by slug, path`,
    )
    .all() as { slug: string; path: string }[];
  for (const { slug, path } of dupes) {
    const last = duplicateStems.at(-1);
    if (last?.slug === slug) last.paths.push(path);
    else duplicateStems.push({ slug, paths: [path] });
  }

  const paths = (sql: string): string[] =>
    (db.query(sql).all() as { path: string }[]).map((r) => r.path);

  return {
    brokenLinks: unresolved.filter((l) => l.matches === 0).map(strip),
    ambiguousLinks: unresolved.filter((l) => l.matches > 1).map(strip),
    orphans: paths(`select path from notes n
       where not exists (select 1 from edges e where e.from_id = n.id)
         and not exists (select 1 from edges e where e.to_id = n.id)
       order by path`),
    duplicateStems,
    malformed: paths(`select path from notes where malformed = 1 order by path`),
  };
}

/**
 * Rebuild from the files into a temp DB, then rename it over index.db — a
 * half-finished rebuild can never become the live index, and a failed one
 * leaves nothing behind.
 * ponytail: two races remain, both narrowed rather than closed. A watcher
 * holding the old file keeps writing to the replaced inode: those writes are
 * lost, not corrupting, and the next doctor run picks them back up off disk.
 * And a watcher that recreates the old file's -wal between our unlink and the
 * rename leaves a stale journal beside the fresh index, which SQLite would
 * then try to apply. A lock file is the upgrade path for both.
 */
async function rebuildIndex(root: string, embedder: Embedder): Promise<void> {
  const target = dbPath(root);
  const tmp = `${target}.rebuild-${Bun.randomUUIDv7()}`;
  try {
    await withDb(tmp, async (db) => {
      await reindex(db, root, embedder);
      db.run("pragma wal_checkpoint(truncate)"); // fold the WAL in before the rename
    });
  } catch (e) {
    rmDb(tmp); // a dead temp index must not accumulate in .vault/
    throw e;
  }
  // The replaced file's journal describes a database that is about to stop
  // existing; drop it first so it is never seen next to the new one.
  rmDb(target, { journalsOnly: true });
  renameSync(tmp, target);
}

function rmDb(path: string, { journalsOnly = false } = {}): void {
  for (const suffix of ["-wal", "-shm"]) rmSync(`${path}${suffix}`, { force: true });
  if (!journalsOnly) rmSync(path, { force: true });
}

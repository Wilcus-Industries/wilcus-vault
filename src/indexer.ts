// Files are truth: the scan is the input, the DB is the output. Dirtiness is
// decided by content hash, never by the DB's own bookkeeping.
import type { Database } from "bun:sqlite";
import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { parseNote, type Note } from "./note";
import { ensureVectors } from "./db";
import { l2normalize, type Embedder } from "./embed";

export type IndexStats = {
  added: number;
  updated: number;
  removed: number;
  unchanged: number;
  /** the embedding model or dims changed, so every note was re-embedded */
  reembedded: boolean;
};

/**
 * Vault-relative paths of every `.md` file, sorted. Dot-directories (`.vault`,
 * `.git`, `.obsidian`, …) are skipped and symlinks are never followed — a link
 * out of the vault is not a note, and a link back in is not a second one.
 */
export function scanVault(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith(".") || entry.isSymbolicLink()) continue;
      const abs = join(dir, entry.name);
      if (entry.isDirectory()) walk(abs);
      else if (entry.isFile() && entry.name.endsWith(".md")) out.push(relative(root, abs));
    }
  };
  walk(root);
  return out.sort();
}

export async function readNote(root: string, rel: string): Promise<Note> {
  return parseNote(await Bun.file(join(root, rel)).text(), rel);
}

/** Hash-diff the vault against the index and write only what changed. */
export async function reindex(
  db: Database,
  root: string,
  embedder: Embedder,
): Promise<IndexStats> {
  const reembedded = ensureVectors(db, embedder);
  const rows = db.query(`select id, path, hash from notes`).all() as {
    id: number;
    path: string;
    hash: string;
  }[];
  const stale = new Map(rows.map((r) => [r.path, r]));
  const embedded = new Set(
    (db.query(`select note_id from vector_meta`).all() as { note_id: number }[]).map(
      (r) => r.note_id,
    ),
  );

  const dirty: { note: Note; mtime: number; id: number | undefined }[] = [];
  let unchanged = 0;
  for (const rel of scanVault(root)) {
    const note = await readNote(root, rel);
    const row = stale.get(rel);
    stale.delete(rel);
    // A row with no vector is half-indexed (interrupted run) — redo it.
    if (row && row.hash === note.hash && !reembedded && embedded.has(row.id)) {
      unchanged++;
      continue;
    }
    dirty.push({ note, mtime: Math.floor(statSync(join(root, rel)).mtimeMs), id: row?.id });
  }

  const vectors = dirty.length
    ? await embedder.embed(dirty.map((d) => `${d.note.title}\n\n${d.note.body}`))
    : [];
  if (vectors.length !== dirty.length) {
    throw new Error(
      `embedder ${embedder.model} returned ${vectors.length} vectors for ${dirty.length} texts`,
    );
  }

  const upsert = db.query(`insert into notes
      (path, slug, title, type, hash, frontmatter, superseded_by, mtime, malformed)
      values (?, ?, ?, ?, ?, ?, ?, ?, ?)
    on conflict(path) do update set
      slug = excluded.slug, title = excluded.title, type = excluded.type,
      hash = excluded.hash, frontmatter = excluded.frontmatter,
      superseded_by = excluded.superseded_by, mtime = excluded.mtime,
      malformed = excluded.malformed
    returning id`);

  db.transaction(() => {
    for (const [i, { note, mtime }] of dirty.entries()) {
      const supersededBy = note.frontmatter["superseded_by"];
      const { id } = upsert.get(
        note.path,
        note.slug,
        note.title,
        note.type ?? null,
        note.hash,
        JSON.stringify(note.frontmatter),
        typeof supersededBy === "string" ? supersededBy : null,
        mtime,
        note.malformedFrontmatter ? 1 : 0,
      ) as { id: number };

      db.run(`delete from notes_fts where rowid = ?`, [id]);
      db.run(`insert into notes_fts (rowid, title, body) values (?, ?, ?)`, [
        id,
        note.title,
        note.body,
      ]);
      db.run(`delete from edges where from_id = ?`, [id]);
      for (const slug of note.links) {
        db.run(`insert or ignore into edges (from_id, to_slug) values (?, ?)`, [id, slug]);
      }
      db.run(`delete from vectors where note_id = ?`, [id]);
      db.run(`insert into vectors (note_id, emb) values (?, ?)`, [id, l2normalize(vectors[i]!)]);
      db.run(`insert or replace into vector_meta (note_id, model, dims) values (?, ?, ?)`, [
        id,
        embedder.model,
        embedder.dims,
      ]);
    }
    for (const row of stale.values()) purgeNote(db, row.id);
    // Resolution depends only on the note set, so an unchanged pass is a no-op
    // — and must stay one, or "nothing changed" would still dirty the file.
    if (dirty.length || stale.size) resolveEdges(db);
  })();

  return {
    added: dirty.filter((d) => d.id === undefined).length,
    updated: dirty.filter((d) => d.id !== undefined).length,
    removed: stale.size,
    unchanged,
    reembedded,
  };
}

/** Drop every derived row for a note whose file is gone. */
export function purgeNote(db: Database, id: number): void {
  db.run(`delete from notes where id = ?`, [id]);
  db.run(`delete from notes_fts where rowid = ?`, [id]);
  db.run(`delete from edges where from_id = ?`, [id]);
  db.run(`delete from vectors where note_id = ?`, [id]);
  db.run(`delete from vector_meta where note_id = ?`, [id]);
}

/**
 * Resolve every wikilink by exact stem match — a slug is never path-joined.
 * Exactly one match wins; zero or several leave to_id null for doctor to
 * report. Recomputed wholesale because adding or removing *any* note can flip
 * a link either way, including links in notes that did not themselves change.
 * ponytail: rewrites every edge on any change — scope it to the touched slugs
 * if a vault ever gets big enough for that to show up.
 */
export function resolveEdges(db: Database): void {
  db.run(`update edges set to_id = (
    select min(n.id) from notes n where n.slug = edges.to_slug having count(*) = 1
  )`);
}

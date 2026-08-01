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

/**
 * Is this vault-relative path one `scanVault` would index — a `.md` file with
 * no dot-directory (or dot-file) in it? The same rule, applied to a path we
 * were handed rather than one we walked to: `fs.watch` reports paths, not
 * directory entries, so the watcher cannot check for symlinks the way the scan
 * does (one that slips in is purged by the next `doctor`).
 */
export function isNotePath(rel: string): boolean {
  return rel.endsWith(".md") && !rel.split(/[/\\]/).some((segment) => segment.startsWith("."));
}

/**
 * Read and parse one note, or null if it is no longer there. A human can
 * delete a file between the scan and the read; that is an ordinary event in a
 * files-are-truth vault, not a reason to abort the run. Any other read error
 * (permissions, I/O) still throws.
 */
export async function readNote(root: string, rel: string): Promise<Note | null> {
  const raw = await readRaw(root, rel);
  return raw === null ? null : parseNote(raw, rel);
}

/** The file's text, or null if it is no longer there (see `readNote`). */
export async function readRaw(root: string, rel: string): Promise<string | null> {
  try {
    return await Bun.file(join(root, rel)).text();
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw e;
  }
}

/** Hash-diff the whole vault against the index and write only what changed. */
export async function reindex(
  db: Database,
  root: string,
  embedder: Embedder,
): Promise<IndexStats> {
  // Every path the files know about plus every path the index knows about: the
  // ones only the index has are deletions, and `indexPaths` purges them.
  const indexed = (db.query(`select path from notes`).all() as { path: string }[]).map(
    (r) => r.path,
  );
  return indexPaths(db, root, embedder, [...scanVault(root), ...indexed]);
}

/**
 * Hash-diff exactly these vault-relative paths and write only what changed; a
 * path whose file is gone is purged. `reindex` passes the whole vault, the
 * watcher passes the handful of paths that just changed — one code path, so a
 * watched vault and a rebuilt one cannot end up with different rows.
 * ponytail: reads the whole `notes` table even for one path (three columns of a
 * hand-written vault). Restrict it to the paths asked for if that ever shows up.
 */
export async function indexPaths(
  db: Database,
  root: string,
  embedder: Embedder,
  rels: Iterable<string>,
): Promise<IndexStats> {
  const reembedded = ensureVectors(db, embedder);
  const rows = db.query(`select id, path, hash from notes`).all() as {
    id: number;
    path: string;
    hash: string;
  }[];
  const indexed = new Map(rows.map((r) => [r.path, r]));
  const embedded = new Set(
    (db.query(`select note_id from vector_meta`).all() as { note_id: number }[]).map(
      (r) => r.note_id,
    ),
  );

  const dirty: { note: Note; mtime: number; id: number | undefined }[] = [];
  // Rows whose file is no longer on disk — a deletion, or a file removed
  // between the scan and the read, which is the same thing by the time we look.
  const gone: number[] = [];
  let unchanged = 0;
  for (const rel of new Set(rels)) {
    const row = indexed.get(rel);
    const note = await readNote(root, rel);
    if (note === null) {
      if (row) gone.push(row.id);
      continue;
    }
    // A row with no vector is half-indexed (interrupted run) — redo it.
    if (row && row.hash === note.hash && !reembedded && embedded.has(row.id)) {
      unchanged++;
      continue;
    }
    const mtime = statSync(join(root, rel), { throwIfNoEntry: false })?.mtimeMs ?? 0;
    dirty.push({ note, mtime: Math.floor(mtime), id: row?.id });
  }

  const vectors = dirty.length
    ? await embedder.embed(dirty.map((d) => `${d.note.title}\n\n${d.note.body}`))
    : [];
  // Catch a misbehaving provider here, not as a raw sqlite-vec failure halfway
  // through the write transaction.
  if (vectors.length !== dirty.length) {
    throw new Error(
      `embedder ${embedder.model} returned ${vectors.length} vectors for ${dirty.length} texts`,
    );
  }
  const wrong = vectors.find((v) => v.length !== embedder.dims);
  if (wrong) {
    throw new Error(
      `embedder ${embedder.model} returned a vector of width ${wrong.length}, expected ${embedder.dims}`,
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
      // A note with no tokens the embedder recognises (CJK, emoji, empty)
      // yields an all-zero vector, which has no direction — cosine distance
      // against it is NaN and would poison KNN. Skip the row; the note stays
      // findable through FTS. The meta row still records the attempt, so the
      // note does not look half-indexed on every later pass.
      const emb = l2normalize(vectors[i]!);
      if (emb.some((x) => x !== 0)) {
        db.run(`insert into vectors (note_id, emb) values (?, ?)`, [id, emb]);
      }
      db.run(`insert or replace into vector_meta (note_id, model, dims) values (?, ?, ?)`, [
        id,
        embedder.model,
        embedder.dims,
      ]);
    }
    for (const id of gone) purgeNote(db, id);
    // Resolution depends only on the note set, so an unchanged pass is a no-op
    // — and must stay one, or "nothing changed" would still dirty the file.
    if (dirty.length || gone.length) resolveEdges(db);
  })();

  return {
    added: dirty.filter((d) => d.id === undefined).length,
    updated: dirty.filter((d) => d.id !== undefined).length,
    removed: gone.length,
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

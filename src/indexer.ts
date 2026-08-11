// Files are truth: the scan is the input, the DB is the output. Dirtiness is
// decided by content hash, never by the DB's own bookkeeping.
import type { Database } from "bun:sqlite";
import { lstatSync, readdirSync, type Stats } from "node:fs";
import { join, relative } from "node:path";
import { linkTarget, parseNote, qualifyLinks, type Note } from "./note";
import { confinedPath, writeAtomic } from "./gate";
import { resetVectors, vectorsStale } from "./db";
import { l2normalize, type Embedder } from "./embed";

/**
 * One stem collision's auto-qualify outcome (DESIGN.md § Data model): the bare
 * `[[stem]]` links that still unambiguously meant the incumbent, rewritten to
 * its path-qualified form. `target` is null when the incumbent sits at the
 * vault root — it has no qualified form, so nothing is rewritten and every
 * linker is reported. `skipped` also carries linkers edited mid-flight (hash
 * mismatch — never clobbered) and the remainder past the cap; all of them fall
 * back to doctor's ambiguous report. The invariant is *never guesses*, not
 * *never ambiguous*.
 */
export type Qualified = {
  stem: string;
  /** the incumbent's path-qualified link target, or null for a root incumbent */
  target: string | null;
  /** linking notes whose bare links were rewritten */
  rewritten: string[];
  /** linking notes left alone: root incumbent, edited mid-flight, or over the cap */
  skipped: string[];
};

export type IndexStats = {
  added: number;
  updated: number;
  removed: number;
  unchanged: number;
  /** the embedding model or dims changed, so every note was re-embedded */
  reembedded: boolean;
  /** stem collisions this pass created, and the bare links qualified for them */
  qualified: Qualified[];
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
      // Separators are normalized to `/`: a stored path is compared against
      // `/`-written wikilink targets, and `relative` yields `\` on Windows.
      else if (entry.isFile() && entry.name.endsWith(".md")) {
        out.push(relative(root, abs).replaceAll("\\", "/"));
      }
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
 * The `lstat` of a path that really holds a note, or null. **Only a regular
 * file is a note** (DESIGN.md § Layout): a path that is gone, has become a
 * directory, or has become a symlink — which the scan skips, and which can
 * point clean out of the vault — holds no note. Decided *before* the read and
 * shared, so the indexer's deletion rule and `get`'s null cannot drift apart.
 */
export function noteEntry(root: string, rel: string): Stats | null {
  const entry = lstatSync(join(root, rel), { throwIfNoEntry: false });
  return entry?.isFile() ? entry : null;
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
  const stats = await indexPaths(db, root, embedder, [...scanVault(root), ...indexed]);
  // Unconditionally, unlike the pass inside `indexPaths`: an index written by
  // an older version holds `to_id` values *that* rule produced, and no file has
  // to change for them to be wrong. The whole-vault entry point is where a
  // changed rule gets applied — otherwise a link that only the new rule can
  // resolve stays broken until someone edits a note or runs `--rebuild`. The
  // watcher's no-op pass stays writeless because it does not come through here.
  resolveEdges(db);
  return stats;
}

/**
 * Hash-diff exactly these vault-relative paths and write only what changed; a
 * path whose file is gone is purged. `reindex` passes the whole vault, the
 * watcher passes the handful of paths that just changed — one code path, so a
 * watched vault and a rebuilt one cannot end up with different rows.
 * ponytail: reads the whole `notes` table even for one path (three columns of a
 * hand-written vault). Restrict it to the paths asked for if that ever shows up.
 *
 * When this pass *creates* a stem collision — a newly indexed note's stem
 * matches exactly one note the index already knew about — every bare
 * `[[stem]]` link in the vault still unambiguously means that incumbent, and
 * only this moment can know it: once both notes are indexed, nothing records
 * which was first. Those links are rewritten to the incumbent's path-qualified
 * form after the write transaction commits (no file I/O inside it — a rollback
 * cannot unwrite a file), check-and-write gated, capped at `qualifyCap` per
 * collision, and reported in `stats.qualified`. The rewritten paths re-enter
 * this function once; their stems do not change, so no further collisions and
 * no recursion.
 */
export async function indexPaths(
  db: Database,
  root: string,
  embedder: Embedder,
  rels: Iterable<string>,
  qualifyCap = QUALIFY_CAP,
): Promise<IndexStats> {
  // Only asked here, never acted on: the vectors this invalidates are the ones
  // the vault searches with until the replacements exist, and `embed` below is
  // a network call that fails for ordinary reasons. The drop happens in the
  // write transaction, with the new vectors in hand.
  const reembedded = vectorsStale(db, embedder);
  const rows = db.query(`select id, path, hash, slug from notes`).all() as {
    id: number;
    path: string;
    hash: string;
    slug: string;
  }[];
  const indexed = new Map(rows.map((r) => [r.path, r]));
  const embedded = new Set(
    (db.query(`select note_id from vector_meta`).all() as { note_id: number }[]).map(
      (r) => r.note_id,
    ),
  );

  const dirty: { note: Note; mtime: number; id: number | undefined }[] = [];
  // Rows whose path is no longer a note on disk — see the entry check below.
  const gone: number[] = [];
  let unchanged = 0;
  for (const rel of new Set(rels)) {
    const row = indexed.get(rel);
    // What is at the path decides, and it is decided *before* the read: only a
    // regular file is a note. A path that is gone, has become a directory, or
    // has become a symlink — which the scan skips, and which can point clean
    // out of the vault — is a deletion. Reading first would index a symlink's
    // target from outside the vault into a row nothing can ever purge (the
    // scan never lists it again, so doctor calls it missing forever), and would
    // throw EISDIR out of both reindex and doctor on a path turned directory,
    // leaving no way to repair the vault at all.
    const entry = noteEntry(root, rel);
    // Still null-checked after the read: a human may delete the file between
    // the two syscalls, which is an ordinary event, not a failure.
    const note = entry === null ? null : await readNote(root, rel);
    if (note === null) {
      if (row) gone.push(row.id);
      continue;
    }
    // A row with no vector is half-indexed (interrupted run) — redo it.
    if (row && row.hash === note.hash && !reembedded && embedded.has(row.id)) {
      unchanged++;
      continue;
    }
    dirty.push({ note, mtime: Math.floor(entry?.mtimeMs ?? 0), id: row?.id });
  }

  // Detect, before the write transaction, from the snapshot above: a new note
  // (`id === undefined`) whose stem exactly one pre-existing note still holds —
  // "still": a row in `gone` is a rename or deletion, not an incumbent. Zero
  // holders is no collision; two or more means the stem was already ambiguous
  // and there is nothing left to protect. Deduped per stem: two newcomers
  // colliding with the same incumbent are one rewrite.
  const goneSet = new Set(gone);
  const collisions = new Map<string, string>(); // stem → incumbent path
  for (const d of dirty) {
    if (d.id !== undefined) continue;
    const holders = rows.filter((r) => r.slug === d.note.slug && !goneSet.has(r.id));
    if (holders.length === 1) collisions.set(d.note.slug, holders[0]!.path);
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
    // Now: the replacements are embedded and validated, and a failure from here
    // on rolls the drop back with everything else.
    resetVectors(db, embedder, reembedded);
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
      // `to_slug` holds the link as written — a bare stem or a path; see
      // `resolveEdges` for how each is resolved.
      for (const target of note.links) {
        db.run(`insert or ignore into edges (from_id, to_slug) values (?, ?)`, [id, target]);
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

  // Rewrite after commit: the moment both notes are indexed is the only one
  // that knows which was the incumbent, and the only file writes `indexPaths`
  // makes. Between the commit above and the re-entry below, the qualified
  // edges briefly read `to_id = null` — the documented window.
  const qualified: Qualified[] = [];
  const rewritten: string[] = [];
  // A bare link *inside* a colliding newcomer has no settled meaning — it was
  // written when its own note was the collision — so it is left to doctor.
  const newPaths = new Set(dirty.filter((d) => d.id === undefined).map((d) => d.note.path));
  for (const [stem, incumbent] of collisions) {
    // A root incumbent's path minus `.md` *is* the bare stem: no qualified
    // form, so its linkers are reported, never rewritten (DESIGN.md § Layout).
    const target = incumbent.includes("/") ? linkTarget(incumbent) : null;
    const linkers = db
      .query(
        `select n.path, n.hash from edges e join notes n on n.id = e.from_id
         where e.to_slug = ? order by n.path`,
      )
      .all(stem) as { path: string; hash: string }[];
    const entry: Qualified = { stem, target, rewritten: [], skipped: [] };
    for (const { path, hash } of linkers) {
      if (newPaths.has(path)) continue;
      if (target === null || entry.rewritten.length >= qualifyCap) {
        entry.skipped.push(path);
        continue;
      }
      const abs = confinedPath(root, path); // the index is derived data, not a trusted path source
      const raw = await readRaw(root, path);
      if (raw === null) continue; // indexed but gone: files are truth, the row is stale
      // Check-and-write: the file must still be what the index read it at — a
      // human's mid-flight edit is skipped, never clobbered, like supersede's
      // `unmarked`.
      if (parseNote(raw, path).hash !== hash) {
        entry.skipped.push(path);
        continue;
      }
      // In-place over the raw text (body region only): every byte this rewrite
      // does not mean — frontmatter, CRLF, a BOM — survives verbatim.
      const out = qualifyLinks(raw, stem, target);
      if (out === raw) continue; // nothing the parser calls a link actually matched
      await writeAtomic(abs, out);
      entry.rewritten.push(path);
      rewritten.push(path);
    }
    qualified.push(entry);
  }
  // Re-enter on just the rewritten paths so the index never lags a write we
  // made ourselves. Bounded structurally: none of them is new to the index, so
  // the pass detects no collision and recurses no further.
  if (rewritten.length > 0) await indexPaths(db, root, embedder, rewritten);

  return {
    added: dirty.filter((d) => d.id === undefined).length,
    updated: dirty.filter((d) => d.id !== undefined).length,
    removed: gone.length,
    unchanged,
    reembedded,
    qualified,
  };
}

/**
 * Rewrites per collision before the remainder is reported instead (#29): the
 * rewrite is deterministic, so this is a generous fuse against a pathological
 * vault, not a tuning knob. Injectable via `indexPaths` for the tests.
 */
export const QUALIFY_CAP = 500;

/** Drop every derived row for a note whose file is gone. */
export function purgeNote(db: Database, id: number): void {
  db.run(`delete from notes where id = ?`, [id]);
  db.run(`delete from notes_fts where rowid = ?`, [id]);
  db.run(`delete from edges where from_id = ?`, [id]);
  db.run(`delete from vectors where note_id = ?`, [id]);
  db.run(`delete from vector_meta where note_id = ?`, [id]);
}

/**
 * Resolve every wikilink, namespace-aware (DESIGN.md § Data model). A target
 * carrying a `/` is a **path**: it matches one note's vault-relative path minus
 * its `.md`, exactly, so `[[customers/acme]]` and `[[vendors/acme]]` are two
 * different notes. A target without one is a **bare stem**: it resolves only
 * when exactly one note in the vault carries that filename stem — several
 * candidates leave `to_id` null (ambiguous) rather than picking a winner, which
 * is the only answer that cannot silently point at the wrong note.
 *
 * Both branches are SQL against paths the scan already found: a link is never
 * joined onto the filesystem, so `[[../../etc/passwd]]` is not a traversal, it
 * is a string that matches no row.
 *
 * Recomputed wholesale because adding or removing *any* note can flip a link
 * either way, including links in notes that did not themselves change.
 * ponytail: rewrites every edge on any change — scope it to the touched targets
 * if a vault ever gets big enough for that to show up.
 */
export function resolveEdges(db: Database): void {
  db.run(`update edges set to_id = case
    when instr(to_slug, '/') > 0
      then (select n.id from notes n where n.path = edges.to_slug || '.md')
      else (select min(n.id) from notes n where n.slug = edges.to_slug having count(*) = 1)
  end`);
}

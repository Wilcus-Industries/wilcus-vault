// The public API. Everything a library caller needs is here; the modules below
// it are the implementation. Embedder and decider are injected — the vault
// never hardcodes a provider.
import type { Database } from "bun:sqlite";
import { dirname, relative, resolve } from "node:path";
import { openDb, dbPath } from "./db";
import {
  isNotePath,
  noteEntry,
  readNote,
  reindex as reindexVault,
  type IndexStats,
} from "./indexer";
import { doctor as runDoctor, type DoctorOptions, type DoctorReport } from "./doctor";
import {
  confinedPath,
  propose as runGate,
  type Candidate,
  type GateOptions,
  type GateResult,
  type VaultContext,
} from "./gate";
import { hybridSearch, type SearchHit, type SearchOptions } from "./search";
import { watch as watchVault, type WatchOptions, type Watcher } from "./watch";
import type { Embedder } from "./embed";
import type { Note } from "./note";

export type VaultOptions = {
  embedder: Embedder;
  /** required by `propose`: the write gate needs a decider *and* explicit cutoffs */
  gate?: GateOptions;
};

export type Vault = {
  /** absolute vault root */
  readonly root: string;
  /** hybrid search over the index (DESIGN.md § Retrieval) */
  search(query: string, options?: SearchOptions): Promise<SearchHit[]>;
  /**
   * One note by its identity — the vault-relative path, `.md` and all
   * (`ledger/q3.md`). Read from the **file**, so a stale or missing index row
   * cannot change the answer. The path is canonicalized first (`./x.md`,
   * `a//x.md` and an absolute path inside the vault all name one note), so the
   * returned `path` is that identity however the caller spelled it. Null when
   * nothing is there; throws when the path escapes the vault, or runs through a
   * dot-directory or a symlinked one.
   */
  get(path: string): Promise<Note | null>;
  /**
   * Vault-relative paths of every note, sorted. `prefix` scopes it to one
   * namespace and matches on segment boundaries: `ledger` and `ledger/` both
   * match `ledger/q3.md`, neither matches `ledger-archive/q3.md`.
   */
  list(prefix?: string): string[];
  /**
   * The write gate: search → decide → apply with check-and-write + confinement.
   * `ctx` names the calling agent for this one call; given, its provenance is
   * stamped on every note the gate authors (DESIGN.md § Scopes and context).
   */
  propose(candidate: Candidate, ctx?: VaultContext): Promise<GateResult>;
  /** hash-diff the files into the index */
  reindex(): Promise<IndexStats>;
  /** report and repair index drift; `--rebuild` reindexes from scratch */
  doctor(options?: DoctorOptions): Promise<DoctorReport>;
  /** keep the index up to date as the files change, until `close()` */
  watch(options?: WatchOptions): Watcher;
  close(): void;
};

/** Open a vault. The index is created on demand; the files are the truth. */
export function open(root: string, { embedder, gate }: VaultOptions): Vault {
  const dir = resolve(root);
  let db: Database = openDb(dbPath(dir));
  return {
    root: dir,
    search: (query, options) => hybridSearch(db, embedder, query, options),
    // Files are truth, so `get` reads the file and parses it — the index is
    // never the thing served. Identity is the path *including* `.md`
    // (DESIGN.md § Data model): `ledger/q3` names no note. A path that had no
    // business being built — one that escapes the root, or runs through a
    // dot-directory or a symlinked one — throws, like the write gate's; a path
    // that is merely absent, or holds a directory or a symlink, is null. Async
    // so both answers reach the caller the same way.
    async get(rel) {
      // Canonicalized once, into exactly the form the scan stores: `./x.md`,
      // `a//x.md`, `a/../a/x.md`, an absolute path inside the vault and a
      // Windows-joined `a\x.md` all name one note, and `path` is the identity a
      // caller keeps — it must not vary with how the caller spelled it. An
      // escape becomes a leading `../`, which `confinedPath` refuses below.
      const norm = relative(dir, resolve(dir, rel)).replaceAll("\\", "/");
      // No filename holds a NUL, so there is no note here — and saying so is
      // better than the raw TypeError `lstat` would throw at the caller.
      if (norm.includes("\0")) return null;
      // The *parent* is confined, not the leaf: `noteEntry` already refuses a
      // symlink where the note should be (it is not a note, like a directory),
      // while a symlinked or hidden directory on the way down, and any escape,
      // still throw — a path cannot escape through its last segment alone.
      confinedPath(dir, dirname(norm));
      return isNotePath(norm) && noteEntry(dir, norm) ? readNote(dir, norm) : null;
    },
    // Derived data is fine here: these are paths, and every one of them is
    // rebuildable from the files. A prefix names a namespace and matches on
    // segment boundaries — raw `startsWith` would sweep in `ledger-archive/`
    // (DESIGN.md § Scopes and context, same rule). Superseded notes are
    // listed: this is the note set, not the search set.
    // ponytail: filters in JS rather than in SQL, where `like`'s wildcards
    // would need escaping. Push it into the query if a vault ever gets big.
    list(prefix) {
      // Slashes at either end are decoration: `/`, `` and `ledger/` say root,
      // root and `ledger/`. Stored paths carry neither.
      const ns = prefix?.replace(/^\/+|\/+$/g, "") ?? "";
      const under = ns === "" ? "" : `${ns}/`;
      return (db.query(`select path from notes order by path`).all() as { path: string }[])
        .map((r) => r.path)
        .filter((path) => path.startsWith(under));
    },
    // async, so a missing gate is a rejected promise like every other failure
    // here rather than a synchronous throw a caller's .catch() would miss
    async propose(candidate, ctx) {
      if (gate === undefined) {
        throw new Error("vault: propose needs a gate — open() with {gate: {decider, cutoffs}}");
      }
      return runGate(db, dir, embedder, candidate, gate, ctx);
    },
    reindex: () => reindexVault(db, dir, embedder),
    async doctor(options) {
      const report = await runDoctor(dir, embedder, options);
      // A rebuild renames a fresh index over the old file; our handle still
      // points at the replaced inode, so take the new one.
      if (options?.rebuild) {
        db.close();
        db = openDb(dbPath(dir));
      }
      return report;
    },
    // `db` is re-opened by a rebuild, so the watcher is handed the handle that
    // is current when it starts, not the one this closure was built with.
    watch: (options) => watchVault(db, dir, embedder, options),
    close: () => db.close(),
  };
}

export type { Candidate, GateResult, GateOptions, VaultContext } from "./gate";
export type { SearchHit, SearchOptions, Cutoffs } from "./search";
export type { Embedder } from "./embed";
export type { DoctorReport, DoctorOptions } from "./doctor";
export type { WatchOptions, Watcher } from "./watch";
export type { IndexStats } from "./indexer";
export type { Note } from "./note";
export { gatePrompt, parseDecision, slugify } from "./gate";
export type { Decider, DeciderInput, Decision, SimilarNote, Action } from "./gate";
export { TokenOverlapEmbedder, FetchEmbedder } from "./embed";

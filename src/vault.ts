// The public API. Everything a library caller needs is here; the modules below
// it are the implementation. Embedder and decider are injected — the vault
// never hardcodes a provider.
import type { Database } from "bun:sqlite";
import { resolve } from "node:path";
import { openDb, dbPath } from "./db";
import { reindex as reindexVault, type IndexStats } from "./indexer";
import { doctor as runDoctor, type DoctorOptions, type DoctorReport } from "./doctor";
import { propose as runGate, type Candidate, type GateOptions, type GateResult } from "./gate";
import { hybridSearch, type SearchHit, type SearchOptions } from "./search";
import { watch as watchVault, type WatchOptions, type Watcher } from "./watch";
import type { Embedder } from "./embed";

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
  /** the write gate: search → decide → apply with check-and-write + confinement */
  propose(candidate: Candidate): Promise<GateResult>;
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
    // async, so a missing gate is a rejected promise like every other failure
    // here rather than a synchronous throw a caller's .catch() would miss
    async propose(candidate) {
      if (gate === undefined) {
        throw new Error("vault: propose needs a gate — open() with {gate: {decider, cutoffs}}");
      }
      return runGate(db, dir, embedder, candidate, gate);
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

export type { Candidate, GateResult, GateOptions } from "./gate";
export type { SearchHit, SearchOptions, Cutoffs } from "./search";
export type { Embedder } from "./embed";
export type { DoctorReport, DoctorOptions } from "./doctor";
export type { WatchOptions, Watcher } from "./watch";
export type { IndexStats } from "./indexer";
export type { Note } from "./note";
export { gatePrompt, parseDecision, slugify } from "./gate";
export type { Decider, DeciderInput, Decision, SimilarNote, Action } from "./gate";
export { TokenOverlapEmbedder, FetchEmbedder } from "./embed";

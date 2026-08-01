// The watcher keeps the index warm while a human edits the vault. It is a
// convenience, never a source of truth (DESIGN.md § Doctor / watcher): every
// pass is the same hash-diff `reindex` runs, and anything the watcher misses or
// drops on the floor — a missed event, a failed pass, a crash — is found again
// by `vault doctor`. So nothing here may throw at the caller mid-run.
import type { Database } from "bun:sqlite";
import { watch as fsWatch } from "node:fs";
import { indexPaths, isNotePath, reindex, type IndexStats } from "./indexer";
import type { Embedder } from "./embed";

export type WatchOptions = {
  /** quiet period per path before it is indexed (default 250ms) */
  debounceMs?: number;
  /** after every pass, with the paths it covered — including no-op passes */
  onChange?: (paths: string[], stats: IndexStats) => void;
  /** transient fs or indexing failures; logged and swallowed by default */
  onError?: (error: unknown) => void;
};

export type Watcher = {
  /**
   * Feed in a vault-relative path as if `fs.watch` had reported it. The event
   * handler is this call, so the debounce-and-index core can be driven directly
   * instead of through the filesystem's timing.
   */
  touch(rel: string): void;
  /** resolves once every debounced path has been indexed and nothing is in flight */
  idle(): Promise<void>;
  /** stop watching; pending debounced paths are dropped (doctor finds them) */
  close(): void;
};

/**
 * Watch the vault and index what changes. `fs.watch` is recursive from the
 * root; only `.md` paths outside dot-directories are acted on, so the index's
 * own writes under `.vault/` cannot feed the watcher its own tail.
 *
 * Debounce is **per path**: an editor writing one file continuously delays that
 * file only, never every other change behind it. Paths that come due together
 * are indexed together, and while a pass is in flight new ones queue up behind
 * it rather than running concurrently against the same database.
 */
export function watch(
  db: Database,
  root: string,
  embedder: Embedder,
  { debounceMs = 250, onChange, onError = logError }: WatchOptions = {},
): Watcher {
  const timers = new Map<string, ReturnType<typeof setTimeout>>();
  const ready = new Set<string>();
  let running: Promise<void> | null = null;
  let closed = false;

  const flush = async (): Promise<void> => {
    // Yield one timer tick first: paths whose debounce expired in the same tick
    // (a save-all, a `git checkout`) then join this pass instead of queueing a
    // pass each.
    await Bun.sleep(0);
    while (ready.size > 0) {
      const paths = [...ready].sort();
      ready.clear();
      try {
        const stats = await indexPaths(db, root, embedder, paths);
        // A model/dims swap wipes every vector; the paths above were re-embedded
        // with it, so re-embed the rest rather than leave the vault half-indexed.
        if (stats.reembedded) await reindex(db, root, embedder);
        onChange?.(paths, stats);
      } catch (e) {
        // A malformed note, a provider blip, a locked database: report it and
        // keep watching. The files are still the truth, and doctor still fixes
        // whatever this pass could not.
        onError(e);
      }
    }
    running = null;
  };

  const touch = (rel: string): void => {
    if (closed) return;
    clearTimeout(timers.get(rel));
    timers.set(
      rel,
      setTimeout(() => {
        timers.delete(rel);
        ready.add(rel);
        running ??= flush();
      }, debounceMs),
    );
  };

  const watcher = fsWatch(root, { recursive: true }, (_event, filename) => {
    // A directory rename reports only the directory, so the notes moved inside
    // it are missed until the next doctor run — the documented ceiling of a
    // watcher that is not the source of truth.
    if (typeof filename === "string" && isNotePath(filename)) touch(filename);
  });
  // Without a listener an 'error' event throws out of the event loop and takes
  // the process with it. Watching stops mattering long before losing the vault.
  watcher.on("error", onError);

  return {
    touch,
    // ponytail: polls instead of keeping a waiter list — it is used at the end
    // of a run and in tests, not on the hot path.
    async idle() {
      while (timers.size > 0 || ready.size > 0 || running !== null) await Bun.sleep(1);
    },
    close() {
      closed = true;
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
      watcher.close();
    },
  };
}

function logError(error: unknown): void {
  console.error(`vault watch: ${error instanceof Error ? error.message : String(error)}`);
}

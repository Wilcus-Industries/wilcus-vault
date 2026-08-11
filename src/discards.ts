// The read side of the discard log (#33). The write gate appends
// (`logCandidate` in gate.ts); this module lists, fetches and restores. Files
// are truth here too: the log is JSONL on disk, read whole on every call — no
// index, no cache.
// ponytail: a full parse per call, bounded by the 5 MiB rotation cap per file.
// Stream it if a vault ever accumulates enough rotations for this to show up.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { discardLog, type Candidate, type Decision, type GateResult, type VaultContext } from "./gate";

/** One member of the similar set the decider saw, as logged. */
export type DiscardedSimilar = { path: string; hash: string; score: number };

export type DiscardEntry = {
  /**
   * 1 = most recent, numbered across the live log and every rotation of it.
   * A rotation between a `list` and a `restore` can shift numbers — `show <n>`
   * first is the check. Assigned at read time, never stored.
   */
  n: number;
  at: string;
  candidate: Candidate;
  /** the decider's verdict; absent when the line is a placement failure */
  decision?: Decision;
  /** why the gate could not place the note; absent on a decided discard */
  reason?: string;
  /** what the decider judged against; [] when nothing was similar — and for lines logged before #33 */
  similar: DiscardedSimilar[];
};

/** Every log file, oldest first: `.discarded.1.log`, `.2`, …, then the live log. */
function logFiles(root: string): string[] {
  const rotated = readdirSync(root)
    .map((f) => f.match(/^\.discarded\.(\d+)\.log$/))
    .filter((m): m is RegExpMatchArray => m !== null)
    .sort((a, b) => Number(a[1]) - Number(b[1]))
    .map((m) => join(root, m[0]));
  return [...rotated, discardLog(root)];
}

/**
 * Every entry, newest first. Malformed lines are counted and skipped, never
 * fatal: the log is append-only history, and one bad line must not brick the
 * review of every good one.
 */
export function listDiscards(root: string): { entries: DiscardEntry[]; malformed: number } {
  const parsed: Omit<DiscardEntry, "n">[] = [];
  let malformed = 0;
  for (const file of logFiles(root)) {
    let text: string;
    try {
      text = readFileSync(file, "utf8");
    } catch (e) {
      // The live log may simply not exist yet; a rotated file came from
      // readdir a moment ago, but files-are-truth means disk wins either way.
      if ((e as NodeJS.ErrnoException).code === "ENOENT") continue;
      throw e;
    }
    for (const line of text.split("\n")) {
      if (line.trim() === "") continue;
      try {
        const e = JSON.parse(line) as Partial<DiscardEntry>;
        if (typeof e?.at !== "string" || e.candidate === undefined) {
          malformed++;
          continue;
        }
        parsed.push({
          at: e.at,
          candidate: e.candidate,
          ...(e.decision !== undefined && { decision: e.decision }),
          ...(e.reason !== undefined && { reason: e.reason }),
          similar: Array.isArray(e.similar) ? e.similar : [],
        });
      } catch {
        malformed++;
      }
    }
  }
  parsed.reverse(); // files are oldest-first, appends land at the bottom
  return { entries: parsed.map((e, i) => ({ ...e, n: i + 1 })), malformed };
}

/** One entry in full, or null — `n` as `listDiscards` just numbered it. */
export function getDiscard(root: string, n: number): DiscardEntry | null {
  return listDiscards(root).entries.find((e) => e.n === n) ?? null;
}

/**
 * Feed a discarded candidate back through the write gate — the only write
 * door. Re-admission re-runs search+decide against the *current* vault, so a
 * restore lands as create or update by what the vault holds today, not what it
 * held when the entry was refused. The entry itself stays in the log: history,
 * not a queue.
 */
export async function restoreDiscard(
  vault: { readonly root: string; propose(c: Candidate, ctx?: VaultContext): Promise<GateResult> },
  n: number,
  ctx?: VaultContext,
): Promise<GateResult> {
  const entry = getDiscard(vault.root, n);
  if (entry === null) throw new Error(`discards: no entry ${n} — run vault discards list`);
  return vault.propose(entry.candidate, ctx);
}

/** Entries younger than this are "recent" in doctor's report. */
const RECENT_MS = 7 * 24 * 60 * 60 * 1000;

/** What doctor reports: how much history there is, and how much of it is fresh. */
export function countDiscards(root: string): { entries: number; recent: number } {
  const { entries } = listDiscards(root);
  const cutoff = Date.now() - RECENT_MS;
  // An unparseable `at` is NaN, and NaN >= cutoff is false: not recent, still counted.
  return { entries: entries.length, recent: entries.filter((e) => Date.parse(e.at) >= cutoff).length };
}

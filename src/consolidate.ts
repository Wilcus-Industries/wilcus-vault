// The consolidation pass (DESIGN.md § Consolidation pass): the deliberate,
// occasional merge of near-duplicates a vault accretes — the gate only ever
// sees top-k similar at write time, and humans add notes behind its back.
// Manually triggered, never a daemon. Discovery is embedding distance under a
// caller-set ceiling; merging reuses the gate's own rails (create for the
// merged note, the supersede marking for each member) with an injected merger
// in place of the decider. Dry-run by default, and nothing is ever deleted.
import type { Database } from "bun:sqlite";
import { resolve } from "node:path";
import { confinedPath, create, markSuperseded, fence, type VaultContext } from "./gate";
import { readRaw, reindex } from "./indexer";
import { parseNote, type Note } from "./note";
import { printable } from "./term";
import type { Embedder } from "./embed";

/** The one note a cluster becomes. Shaped like a `Candidate` minus its
 * namespace — the cluster decides where it lands, never the merger. */
export type MergedNote = {
  title: string;
  body: string;
  type?: string;
};

/** One cluster's notes, bodies as they are on disk right now. */
export type MergeInput = { notes: Note[] };

/**
 * Injected — the caller wires an LLM (with `mergePrompt` + `parseMerged`),
 * tests wire a fake. Like the gate's decider, and for the same reason: the
 * vault never hardcodes a provider.
 */
export type Merger = (input: MergeInput) => Promise<MergedNote>;

/** Fixed at `open()`, like `GateOptions` — the wiring, not the run. */
export type ConsolidateOptions = { merger: Merger };

/** One run's policy. `ceiling` has no default; see `consolidate`. */
export type ConsolidateRun = {
  /**
   * Cosine-distance ceiling for two notes to be near-duplicates (0 identical,
   * 1 orthogonal). **Mandatory**, like the gate's cutoffs and for the same
   * reason: there is no universal number for "duplicate", it is a property of
   * the embedder, and a default here would merge notes on the library's guess.
   */
  ceiling: number;
  /** clusters this run may merge before it stops and reports the rest (default 5) */
  cap?: number;
  /** actually write (default false — a run reports what it would do) */
  write?: boolean;
  /**
   * Who is running the pass; its provenance is stamped on the merged notes.
   * Consolidation is an operator operation like `doctor` — unscoped — so this
   * is provenance only, never a permission check.
   */
  ctx?: VaultContext;
};

/** Notes that are near-duplicates of each other, every internal pair included. */
export type Cluster = {
  /** vault-relative paths, sorted */
  members: string[];
  /**
   * The namespace every member shares, canonical and `/`-terminated (`""` is
   * the vault root) — where a merge would write. **Null when the members do
   * not share one**: a cluster spanning namespaces is reported and never
   * merged, because collapsing that boundary is a human call.
   */
  namespace: string | null;
  /** the widest pair inside the cluster — complete linkage, so this is under the ceiling */
  distance: number;
};

/** One cluster's merge: proposed on a dry run, applied on a write run. */
export type Merge = {
  cluster: Cluster;
  /** what the merger answered — reported whether or not it was written */
  candidate: MergedNote;
  /** vault-relative path of the merged note; absent on a dry run */
  path?: string;
  /** members marked `superseded_by` the merged note; absent on a dry run */
  superseded?: string[];
  /**
   * Members that could **not** be marked: each changed on disk between being
   * read for the merger and being marked, and a human's edit does not get
   * overwritten to tidy up our bookkeeping. The merged note stands and these
   * are untouched and still live in search — the operator reconciles.
   */
  unmarked?: string[];
};

export type ConsolidateReport = {
  /** nothing was written; `write: true` is what makes a run act */
  dryRun: boolean;
  merges: Merge[];
  /** clusters spanning namespaces: reported, never merged */
  crossNamespace: Cluster[];
  /** clusters left over once the cap was reached, or whose members moved under us */
  remaining: Cluster[];
  /**
   * Clusters whose merge threw mid-pass on a **write** run (merger error, no
   * free filename): collected here so the merges that already landed are still
   * reported instead of discarded behind one exception. `path` is the merged
   * note when the throw landed after `create` wrote it — the file exists and is
   * live in search, and without this the report would never name it. Always
   * empty on a dry run — there a throw propagates, because nothing has landed
   * that a report would need to account for. A member path failing confinement
   * still throws even on a write run: that is a tampered index, not a cluster
   * error.
   */
  errors: { cluster: Cluster; error: string; path?: string }[];
  /**
   * The closing reindex failed after ≥1 merge landed: the index lags the files
   * until the next pass (files are truth — a later reindex or doctor recovers
   * it). Reported rather than thrown, because a throw here would discard the
   * report of everything that landed.
   */
  indexError?: string;
};

/** Single digits, per DESIGN.md: a pass that rewrites half the vault is a wrong ceiling. */
const DEFAULT_CAP = 5;

/**
 * Every live note pair under `ceiling`, agglomerated into clusters under
 * **complete linkage**: a note joins only when it is under the ceiling from
 * *every* note already in the cluster. Single linkage would chain A~B~C and
 * merge A with a C it does not resemble.
 *
 * The scan is all-pairs, O(n²), and accepted: notes embed whole, so n is the
 * note count — thousands at most (DESIGN.md § Consolidation pass).
 * ponytail: one statement over the whole vault. Batch it per namespace if a
 * vault ever gets big enough for the self-join to show up.
 *
 * Superseded notes are excluded on both sides — a retired note is not a
 * duplicate of anything — and a note with no vector row never clusters at all:
 * the indexer stores none for text its embedder has no tokens for (CJK, emoji),
 * so consolidation cannot see those notes. They stay findable through FTS.
 */
export function clusters(db: Database, ceiling: number): Cluster[] {
  const pairs = db
    .query(
      `select na.path as a, nb.path as b, vec_distance_cosine(va.emb, vb.emb) as d
       from vectors va join vectors vb on va.note_id < vb.note_id
       join notes na on na.id = va.note_id
       join notes nb on nb.id = vb.note_id
       where na.superseded_by is null and nb.superseded_by is null
         and vec_distance_cosine(va.emb, vb.emb) <= ?
       order by d, a, b`,
    )
    .all(ceiling) as { a: string; b: string; d: number }[];

  // No path holds a NUL (`confinedPath` refuses one), so this cannot collide.
  const key = (x: string, y: string): string => (x < y ? `${x}\0${y}` : `${y}\0${x}`);
  const under = new Map(pairs.map((p) => [key(p.a, p.b), p.d]));
  // path → the member array of the cluster it belongs to, shared by identity.
  const owner = new Map<string, string[]>();
  // Ascending pair distance, so the closest notes cluster first and the answer
  // does not depend on which pair the query happened to return first.
  for (const { a, b } of pairs) {
    const left = owner.get(a) ?? [a];
    const right = owner.get(b) ?? [b];
    if (left === right) continue;
    // Complete linkage: every *cross* pair has to be under the ceiling too.
    if (!left.every((x) => right.every((y) => under.has(key(x, y))))) continue;
    const merged = [...left, ...right].sort();
    for (const path of merged) owner.set(path, merged);
  }

  return [...new Set(owner.values())]
    .map((members) => {
      // A stored path is canonical, so its directory is already the
      // `normalizePrefix` form `create` writes into: `notes/`, or `""` at the
      // root. Members that do not share one span namespaces.
      const dirs = members.map((p) => p.slice(0, p.lastIndexOf("/") + 1));
      // Every internal pair is in `under` — that is what complete linkage
      // means — so the widest of them is the cluster's own distance. A running
      // max, not a spread: n·(n-1)/2 arguments would hit the engine's argument
      // limit on a cluster a legal wide ceiling can produce.
      let distance = 0;
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          const d = under.get(key(members[i]!, members[j]!))!;
          if (d > distance) distance = d;
        }
      }
      return {
        members,
        namespace: dirs.every((d) => d === dirs[0]) ? dirs[0]! : null,
        distance,
      };
    })
    .sort((x, y) => x.distance - y.distance || x.members[0]!.localeCompare(y.members[0]!));
}

/**
 * Validate a merger's answer. Malformed output throws — the caller sees the
 * error. Like the gate's `checkDecision`: a wrong guess about what a model
 * meant writes the wrong file.
 */
export function checkMerged(value: unknown): MergedNote {
  const bad: (why: string) => never = (why) => {
    throw new Error(`consolidate: merger returned ${why}: ${JSON.stringify(value)?.slice(0, 200)}`);
  };
  if (value === null || typeof value !== "object") bad("a non-object");
  const { title, body, type } = value as Partial<MergedNote>;
  // An empty title or body is a model that lost the text, not a note: the
  // merge would retire N real notes behind a blank one.
  if (typeof title !== "string" || title.trim() === "") bad("no usable title");
  if (typeof body !== "string" || body.trim() === "") bad("no usable body");
  if (type !== undefined && typeof type !== "string") bad("a non-string type");
  // Rebuilt rather than passed through: unknown keys from a model do not
  // travel — a `namespace` it invented would decide where the note lands.
  return { title, body, ...(type !== undefined && { type }) };
}

/** Parse an LLM's reply. Strictly one JSON object — a fenced or chatty answer is malformed. */
export function parseMerged(text: string): MergedNote {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(`consolidate: merger did not return JSON: ${text.slice(0, 200)}`);
  }
  return checkMerged(value);
}

/** The prompt an LLM merger gets. `parseMerged` reads what it asks for. */
export function mergePrompt({ notes }: MergeInput): string {
  const listed = notes
    .map(
      (n, i) =>
        `[${i + 1}] path: ${n.path}\n` +
        `    title: ${n.title}\n` +
        `--- begin note ---\n${fence(n.body)}\n--- end note ---`,
    )
    .join("\n\n");
  return `You are the consolidation pass of a markdown memory vault. The notes below are near-duplicates of each other. Write the single note that replaces all of them.

NOTES TO MERGE
${listed}

Rules:
- Keep every fact from every note. Drop only repetition — two notes saying the same thing become one sentence.
- Where they disagree, keep both readings and say which note each came from; do not decide.
- Invent nothing. If it is not in a note above, it does not belong in the merged note.

Reply with one JSON object and nothing else — no prose, no markdown fence:
{"title": "<title for the merged note>", "type": "<optional type, as the notes above use it>", "body": "<full markdown body>"}`;
}

/**
 * Find near-duplicate clusters and merge them. **Dry-run by default**: a run
 * reports the clusters and each would-be merge, and writing takes `write:
 * true`. A wrong ceiling discovered in a report costs nothing; discovered in
 * the files, it costs an afternoon.
 *
 * A merge is the gate's rails with the merger in place of the decider — no
 * decider runs, because `propose`'s own search could neither see the cluster
 * nor supersede more than one note. The merged note is written through
 * `create` (slug confinement, collision suffixing, provenance from `ctx`) into
 * the cluster's namespace, then *each* member is marked `superseded_by` it
 * through `markSuperseded` (textual patch, check-and-write per member).
 * Nothing is ever deleted: the originals stay on disk, marked, out of search.
 */
export async function consolidate(
  db: Database,
  root: string,
  embedder: Embedder,
  { merger }: ConsolidateOptions,
  { ceiling, cap = DEFAULT_CAP, write = false, ctx }: ConsolidateRun,
): Promise<ConsolidateReport> {
  // No default, and checked rather than trusted to its type: an undefined or
  // NaN ceiling would reach the SQL and quietly match nothing (or everything).
  if (typeof ceiling !== "number" || !Number.isFinite(ceiling) || ceiling < 0 || ceiling > 2) {
    throw new Error(
      "consolidate: ceiling must be a cosine distance in 0..2 — there is no universal " +
        'number for "duplicate", it is a property of the embedder',
    );
  }
  if (!Number.isInteger(cap) || cap < 1) {
    throw new Error(`consolidate: cap must be a positive integer, got ${cap}`);
  }
  // Same rule as the gate's: a blank agent is a caller that meant to identify
  // itself and did not, and it would be stamped into files as an empty string.
  if (ctx !== undefined && ctx.agent.trim() === "") {
    throw new Error("consolidate: ctx.agent must name the calling agent");
  }
  const base = resolve(root);
  // Files are truth, so discovery runs against a current index rather than
  // whatever the last pass left: a note edited or deleted behind the vault's
  // back must not be merged as it used to read. (It also re-embeds after a
  // model swap, so every distance below is measured in one vector space.)
  await reindex(db, base, embedder);

  const found = clusters(db, ceiling);
  // Confinement up front, before anything is written: a member path that
  // escapes the vault root is a tampered or corrupt index, and that aborts the
  // run loudly — never after merges have landed, where the throw would discard
  // their report and skip the closing reindex.
  for (const cluster of found) {
    for (const path of cluster.members) {
      confinedPath(base, path); // the index is derived data, not a trusted path source
    }
  }
  const crossNamespace = found.filter((c) => c.namespace === null);
  const merges: Merge[] = [];
  const remaining: Cluster[] = [];
  const errors: ConsolidateReport["errors"] = [];
  let wrote = false;
  // The cap rations model calls, so it counts clusters whose merger *ran* —
  // merged or errored after the call. An error before the call (an unreadable
  // member) spent nothing and burns no slot: otherwise ≥cap broken clusters
  // would starve every real one, run after run.
  // ponytail: pre-call errors are unbounded — a chmod'd subtree yields one
  // entry per cluster per run. Bound them if a thousand-entry report shows up.
  let acted = 0;
  for (const cluster of found.filter((c) => c.namespace !== null)) {
    if (acted >= cap) {
      remaining.push(cluster);
      continue;
    }
    let created: string | undefined;
    try {
      const notes: Note[] = [];
      for (const path of cluster.members) {
        // Re-read from disk, like everything the vault shows a model — and the
        // hash each note is read at is what its marking is checked against.
        const raw = await readRaw(base, path);
        if (raw === null) break; // indexed but gone: files are truth, the row is stale
        notes.push(parseNote(raw, path));
      }
      // A member vanished under us, so this is no longer the cluster the merger
      // would be asked about. Reported, not guessed at.
      if (notes.length !== cluster.members.length) {
        remaining.push(cluster);
        continue;
      }
      acted++;
      const candidate = checkMerged(await merger({ notes }));
      if (!write) {
        merges.push({ cluster, candidate });
        continue;
      }
      created = (await create(db, base, candidate, cluster.namespace!, undefined, ctx)).path;
      wrote = true; // the file exists from here on, whatever the marking does
      const superseded: string[] = [];
      const unmarked: string[] = [];
      for (const note of notes) {
        const marked = await markSuperseded(base, note, created!);
        if ("superseded" in marked) superseded.push(marked.superseded);
        else unmarked.push(marked.unmarked);
      }
      merges.push({ cluster, candidate, path: created, superseded, unmarked });
    } catch (e) {
      // On a dry run nothing has landed, so a throw stays a throw. On a write
      // run one bad cluster must not discard the report of the merges that
      // already landed, nor skip the closing reindex below. The merged note's
      // path travels on the entry when `create` got that far: the file is live
      // in search after the reindex, and this is the only place that names it.
      if (!write) throw e;
      errors.push({ cluster, error: printable(e), ...(created !== undefined && { path: created }) });
    }
  }
  // The index never lags a write we made ourselves — and when the reindex
  // itself fails (the embedder is a network call), that must not discard the
  // report of everything that landed: reported, recovered by the next pass.
  let indexError: string | undefined;
  if (wrote) {
    try {
      await reindex(db, base, embedder);
    } catch (e) {
      indexError = printable(e);
    }
  }
  return {
    dryRun: !write,
    merges,
    crossNamespace,
    remaining,
    errors,
    ...(indexError !== undefined && { indexError }),
  };
}

// The write gate (DESIGN.md § Write gate): every programmatic write goes
// through `propose`. Search for what already exists, ask an injected decider
// what to do, then apply it behind two rails — check-and-write (nothing a human
// touched mid-flight is clobbered) and path confinement (an LLM-derived string
// never names a raw filesystem path).
import type { Database } from "bun:sqlite";
import { appendFileSync, existsSync, lstatSync, mkdirSync, renameSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import {
  linkTarget,
  parseNote,
  patchFrontmatter,
  replaceBody,
  serializeNote,
  type Note,
} from "./note";
import { readRaw, reindex } from "./indexer";
import { hybridSearch, type Cutoffs } from "./search";
import type { Embedder } from "./embed";

/** A note something wants to write. `namespace` is a directory under the root. */
export type Candidate = {
  title: string;
  body: string;
  type?: string;
  namespace?: string;
};

/** An existing note the candidate might belong in, as the decider sees it. */
export type SimilarNote = {
  note: Note;
  /** fused RRF score from the hybrid search */
  score: number;
  /**
   * The note's content hash *at search time*. The apply step re-hashes the file
   * and refuses to write if this no longer matches — see `propose`.
   */
  hash: string;
};

export type DeciderInput = { candidate: Candidate; similar: SimilarNote[] };

export type Action = "update" | "supersede" | "create" | "discard";

export type Decision = {
  action: Action;
  /** vault-relative path of the note to update or supersede; that action only */
  target?: string;
  /** body to write instead of the candidate's, if the decider merged one */
  body?: string;
};

/**
 * Injected — the caller wires an LLM (with `gatePrompt` + `parseDecision`),
 * tests wire a fake. The vault never hardcodes a provider.
 */
export type Decider = (input: DeciderInput) => Promise<Decision>;

export type GateOptions = {
  decider: Decider;
  /**
   * Mandatory, per DESIGN.md § Write gate: without cutoffs the search always
   * returns *something* and "most similar note" degrades into "least unrelated
   * note" — the gate would update a stranger instead of creating a new note.
   * An empty result is the expected outcome for a genuinely new note.
   */
  cutoffs: Cutoffs;
  /** similar notes to show the decider (default 5) */
  n?: number;
};

export type GateResult = {
  /** what was actually applied — `create` when the decider's action fell back */
  action: Action;
  /** vault-relative path written; absent for `discard` */
  path?: string;
  /** the note marked `superseded_by`, for `supersede` */
  superseded?: string;
  /**
   * `supersede` wrote the successor but could **not** mark this note: it
   * changed again in the window between the check and the patch, and a human's
   * edit does not get overwritten to tidy up our bookkeeping. The successor
   * stands, this note is untouched and still live in search — the caller (or a
   * later `propose`) reconciles.
   */
  unmarked?: string;
  /** the decision was abandoned: the target changed under two gate runs */
  fellBack: boolean;
};

const ACTIONS: readonly Action[] = ["update", "supersede", "create", "discard"];
/** Long enough to stay readable as a filename, short enough for every FS. */
const MAX_SLUG = 80;
/** Distinct notes may share a title; give up rather than loop forever. */
const MAX_SLUG_TRIES = 50;

/**
 * A single filename segment, `[a-z0-9-]+`. Everything else — separators, dots,
 * accents, punctuation — collapses to a hyphen, so a title can never smuggle a
 * path in (`../../evil` is the note `evil`). Null when nothing survives, which
 * a Japanese or Russian title does legitimately: the caller names the file
 * some other way rather than dropping the note.
 * ponytail: ASCII-only slugs. Transliterate if non-Latin vaults become normal.
 */
export function slugify(title: string): string | null {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, MAX_SLUG)
    .replace(/^-+|-+$/g, "");
  return slug === "" ? null : slug;
}

/**
 * Resolve a vault-relative path and prove it is really inside the vault: no
 * `..` escape, no absolute path, no symlinked directory or file on the way
 * down. Callers pass strings that came from an LLM, a caller's namespace, or a
 * stale index — none of them are trusted to be a safe path.
 */
export function confinedPath(root: string, rel: string): string {
  const base = resolve(root);
  const abs = resolve(base, rel);
  if (abs !== base && !abs.startsWith(base + sep)) {
    throw new Error(`write gate: ${rel} resolves outside the vault`);
  }
  let walk = base;
  for (const segment of relative(base, abs).split(sep)) {
    // The scan skips dot-directories, so a note written into one could never
    // be indexed — an invisible write is not a successful one.
    if (segment.startsWith(".")) {
      throw new Error(`write gate: ${rel} passes through a hidden directory`);
    }
    walk = join(walk, segment);
    // A symlink is a second name for a file that may not be a note at all —
    // the scan skips them too, so the gate must not write through one either.
    if (lstatSync(walk, { throwIfNoEntry: false })?.isSymbolicLink()) {
      throw new Error(`write gate: ${rel} passes through a symlink`);
    }
  }
  return abs;
}

/**
 * Validate a decider's answer. Malformed output throws — the caller sees the
 * error. The gate never guesses what an LLM meant: a wrong guess writes to the
 * wrong file.
 */
export function checkDecision(value: unknown): Decision {
  const bad: (why: string) => never = (why) => {
    throw new Error(`write gate: decider returned ${why}: ${JSON.stringify(value)?.slice(0, 200)}`);
  };
  if (value === null || typeof value !== "object") bad("a non-object");
  const d = value as Partial<Decision>;
  if (!ACTIONS.includes(d.action as Action)) bad("an unknown action");
  const { action, target, body } = d;
  if (target !== undefined && typeof target !== "string") bad("a non-string target");
  if (body !== undefined && typeof body !== "string") bad("a non-string body");
  // An empty body is not an instruction to blank a note — it is a model that
  // lost the text. `update` would truncate the target with it.
  if (body !== undefined && body.trim() === "") bad("an empty body");

  const needsTarget = action === "update" || action === "supersede";
  if (needsTarget && !target) bad(`${action} without a target`);
  // A target on create/discard means the model was confused about the action;
  // silently dropping it is the guess we refuse to make.
  if (!needsTarget && target !== undefined) bad(`${action} with a target`);
  // Rebuilt rather than passed through: unknown keys from a model do not travel.
  return {
    action: action as Action,
    ...(target !== undefined && { target }),
    ...(body !== undefined && { body }),
  };
}

/** Parse an LLM's reply. Strictly one JSON object — a fenced or chatty answer is malformed. */
export function parseDecision(text: string): Decision {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(`write gate: decider did not return JSON: ${text.slice(0, 200)}`);
  }
  return checkDecision(value);
}

/**
 * Note text is data, not instruction. Anything that could pass for one of the
 * delimiters below is indented one space, so a note body cannot close its own
 * fence and address the model directly. The text still reads normally.
 */
const fence = (text: string): string => text.trim().replace(/^---[ \t]*(begin|end)/gim, " $&");

/** The prompt an LLM decider gets. `parseDecision` reads what it asks for. */
export function gatePrompt({ candidate, similar }: DeciderInput): string {
  const notes =
    similar.length === 0
      ? "(none — the vault has nothing similar)"
      : similar
          .map(
            (s, i) =>
              `[${i + 1}] path: ${s.note.path}\n    title: ${s.note.title}\n` +
              `--- begin note ---\n${fence(s.note.body)}\n--- end note ---`,
          )
          .join("\n\n");
  return `You are the write gate of a markdown memory vault. Decide what should happen to one candidate note.

CANDIDATE
title: ${candidate.title}
type: ${candidate.type ?? "(none)"}
--- begin candidate ---
${fence(candidate.body)}
--- end candidate ---

EXISTING NOTES THAT LOOK SIMILAR
${notes}

Choose exactly one action:
- update: the candidate is a correction or addition to one existing note. Give that note's path as "target" and the complete new body as "body".
- supersede: an existing note is now wrong or outdated and the candidate replaces it. Give the old note's path as "target"; it will be marked superseded and linked forward.
- create: nothing above covers this. Do not give a target.
- discard: the candidate adds nothing that is not already written down. Do not give a target.

Reply with one JSON object and nothing else — no prose, no markdown fence:
{"action": "update" | "supersede" | "create" | "discard", "target": "<path exactly as listed above, update and supersede only>", "body": "<full markdown body, optional>"}`;
}

/**
 * Search, decide, apply. The two rails:
 *
 * - **Check-and-write.** Each similar note's hash is captured at search time
 *   and re-checked immediately before the file is touched. A human edit in
 *   between aborts the apply and re-runs the whole gate once against fresh
 *   state; a second mismatch falls back to `create`, so a busy file is never
 *   clobbered and the information is never dropped.
 * - **Path confinement.** New paths are `<namespace>/<slug>.md` with a
 *   single-segment slug, resolved and asserted under the root; an existing
 *   target must be one of the notes the search actually returned.
 *
 * Touched files are reindexed before this returns, so the index never lags a
 * write we made ourselves.
 */
export async function propose(
  db: Database,
  root: string,
  embedder: Embedder,
  candidate: Candidate,
  { decider, cutoffs, n = 5 }: GateOptions,
): Promise<GateResult> {
  // `{}` type-checks, which would make the mandate above cosmetic: with no
  // ceiling at all the search returns the least unrelated note and the gate
  // acts on it. A caller who wants everything says so with a wide ceiling.
  if (cutoffs.distanceCeiling === undefined && cutoffs.bm25Ceiling === undefined) {
    throw new Error(
      "write gate: cutoffs must set distanceCeiling or bm25Ceiling — without one, " +
        "'most similar note' is only 'least unrelated note'",
    );
  }
  const base = resolve(root);
  let applied: Applied | null = null;
  let fellBack = false;
  for (let attempt = 0; attempt < 2 && applied === null; attempt++) {
    // Re-running the gate means re-running it against *fresh* state: the edit
    // that aborted the first attempt is on disk but not yet in the index.
    if (attempt > 0) await reindex(db, base, embedder);
    const similar = await findSimilar(db, base, embedder, candidate, cutoffs, n);
    const decision = checkDecision(await decider({ candidate, similar }));
    if (decision.target !== undefined && !similar.some((s) => s.note.path === decision.target)) {
      // The only paths the gate will touch are ones it just read itself.
      throw new Error(
        `write gate: decider targeted ${decision.target}, which was not among the similar notes`,
      );
    }
    applied = await apply(db, base, candidate, decision, similar);
  }
  if (applied === null) {
    applied = await create(db, base, candidate);
    fellBack = true;
  }
  // The index never lags a write we made ourselves.
  // ponytail: a whole-vault hash-diff pass — it only re-embeds what changed, so
  // it is a directory scan, not a re-embed. Scope it to the touched paths if a
  // vault ever gets big enough for the scan to show up.
  await reindex(db, base, embedder);
  return { ...applied, fellBack };
}

type Applied = Omit<GateResult, "fellBack">;

/** Top-k similar notes, read from disk so their hashes are current. */
async function findSimilar(
  db: Database,
  root: string,
  embedder: Embedder,
  candidate: Candidate,
  cutoffs: Cutoffs,
  n: number,
): Promise<SimilarNote[]> {
  // Same shape the indexer embeds a note with, so like is compared with like.
  const hits = await hybridSearch(db, embedder, `${candidate.title}\n\n${candidate.body}`, {
    n,
    cutoffs,
  });
  const similar: SimilarNote[] = [];
  for (const hit of hits) {
    confinedPath(root, hit.path); // the index is derived data, not a trusted path source
    const raw = await readRaw(root, hit.path);
    if (raw === null) continue; // indexed but gone: files are truth, the row is stale
    const note = parseNote(raw, hit.path);
    similar.push({ note, score: hit.score, hash: note.hash });
  }
  return similar;
}

/** Apply one decision, or return null if the target changed under us. */
async function apply(
  db: Database,
  root: string,
  candidate: Candidate,
  decision: Decision,
  similar: SimilarNote[],
): Promise<Applied | null> {
  if (decision.action === "discard") {
    logCandidate(root, candidate, { decision });
    return { action: "discard" };
  }
  if (decision.action === "create") return await create(db, root, candidate, decision.body);

  const hit = similar.find((s) => s.note.path === decision.target)!; // checked in propose
  const rel = hit.note.path;
  const abs = confinedPath(root, rel);
  // Check-and-write: the file must still be byte-for-byte what the decider saw.
  const raw = await readRaw(root, rel);
  if (raw === null || parseNote(raw, rel).hash !== hit.hash) return null;

  if (decision.action === "update") {
    // Frontmatter is the note's identity — keep it verbatim, bump `updated`.
    const bumped = patchFrontmatter(raw, "updated", now());
    await writeAtomic(abs, replaceBody(bumped, decision.body ?? candidate.body));
    return { action: "update", path: rel };
  }

  // supersede: the successor is written first, then the old note is marked and
  // linked forward to it.
  const created = await create(db, root, candidate, decision.body);
  // Writing the successor took time, and the old note belongs to a human. If it
  // moved in that window the successor stands and the caller is told the old
  // note is unmarked — the alternative is overwriting an edit for bookkeeping.
  const current = await readRaw(root, rel);
  if (current === null || parseNote(current, rel).hash !== hit.hash) {
    return { action: "supersede", path: created.path, unmarked: rel };
  }
  const marked = patchFrontmatter(current, "superseded_by", created.path!);
  // The gate knows the exact path, so it links by path rather than by stem: a
  // namespaced successor cannot go ambiguous later. (A successor written to the
  // vault root has no qualified form, so its link is a bare stem and *can* —
  // DESIGN.md § Data model.)
  const link = `Superseded by [[${linkTarget(created.path!)}]].\n`;
  await writeAtomic(abs, marked.endsWith("\n") ? `${marked}\n${link}` : `${marked}\n\n${link}`);
  return { action: "supersede", path: created.path, superseded: rel };
}

/**
 * Write through a temp file and rename it over the target. A reader never sees
 * a half-written note, a crash never truncates one, and — since rename replaces
 * a symlink instead of following it — nothing can be swapped in between
 * `confinedPath` and the write. The temp name is not `.md`, so a crashed write
 * leaves nothing the scan would index.
 */
async function writeAtomic(abs: string, text: string): Promise<void> {
  const tmp = `${abs}.tmp-${process.pid}-${Bun.randomUUIDv7().slice(-8)}`;
  await Bun.write(tmp, text);
  renameSync(tmp, abs);
}

/** Write a note we authored ourselves — the one place `serializeNote` is used. */
async function create(
  db: Database,
  root: string,
  candidate: Candidate,
  body?: string,
): Promise<Applied> {
  const { rel, abs } = freePath(db, root, candidate);
  const at = now();
  const text = serializeNote({
    frontmatter: {
      title: candidate.title,
      ...(candidate.type !== undefined && { type: candidate.type }),
      created: at,
      updated: at,
    },
    body: body ?? candidate.body,
  });
  mkdirSync(dirname(abs), { recursive: true });
  await writeAtomic(abs, text);
  return { action: "create", path: rel };
}

/**
 * `<namespace>/<slug>.md`, confined to the vault and not already taken —
 * neither by a file nor by another note's stem. A collision suffixes rather
 * than overwrites: the gate does not get to lose someone else's note to a
 * shared title. Stems no longer *have* to be unique (`customers/acme` and
 * `vendors/acme` are two legitimate notes), but the gate still keeps its own
 * unique, because creating a second `acme` is what turns every human's
 * `[[acme]]` ambiguous.
 */
function freePath(db: Database, root: string, candidate: Candidate): { rel: string; abs: string } {
  // A title of nothing but CJK, Cyrillic or emoji slugifies to nothing — name
  // the file after the candidate's own content rather than losing the note.
  const base =
    slugify(candidate.title) ??
    `note-${new Bun.CryptoHasher("sha256")
      .update(`${candidate.title}\n\n${candidate.body}`)
      .digest("hex")
      .slice(0, 8)}`;
  const taken = db.query(`select 1 from notes where slug = ?`);
  for (let i = 1; i <= MAX_SLUG_TRIES; i++) {
    const slug = i === 1 ? base : `${base}-${i}`;
    const rel = candidate.namespace ? `${candidate.namespace}/${slug}.md` : `${slug}.md`;
    const abs = confinedPath(root, rel);
    if (!existsSync(abs) && taken.get(slug) === null) return { rel, abs };
  }
  // Nowhere to put it is still not a reason to drop it on the floor.
  const reason = `write gate: no free filename for ${JSON.stringify(candidate.title)}`;
  logCandidate(root, candidate, { reason });
  throw new Error(reason);
}

/**
 * Append a candidate to `.vault/discarded.log` as JSONL, whole: a wrong decider
 * call — or a gate that cannot write the file — costs a line in a log, never
 * the information itself.
 */
function logCandidate(root: string, candidate: Candidate, extra: Record<string, unknown>): void {
  const dir = join(root, ".vault");
  mkdirSync(dir, { recursive: true });
  appendFileSync(join(dir, "discarded.log"), `${JSON.stringify({ at: now(), candidate, ...extra })}\n`);
}

const now = (): string => new Date().toISOString();

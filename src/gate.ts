// The write gate (DESIGN.md § Write gate): every programmatic write goes
// through `propose`. Search for what already exists, ask an injected decider
// what to do, then apply it behind two rails — check-and-write (nothing a human
// touched mid-flight is clobbered) and path confinement (an LLM-derived string
// never names a raw filesystem path).
import type { Database } from "bun:sqlite";
import {
  closeSync,
  constants,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  writeSync,
} from "node:fs";
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
import { ALLOW_ALL, normalizePrefix, type Scope } from "./scope";
import { safe } from "./term";
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
  /**
   * The calling agent may read this note but not write it (§ Scopes and
   * context). Set only when true, and only under a scope policy: a decision
   * that targets one anyway falls back to `create`, and `gatePrompt` marks it
   * so the model is not asked to guess.
   */
  readOnly?: boolean;
};

/**
 * Who is calling, per call (DESIGN.md § Scopes and context). `agent` names the
 * caller (`core/scheduler`); `source` optionally records what prompted the call
 * — a conversation id, a task id, freeform. Identity travels per call rather
 * than being frozen at `open()`, because one process holds one vault handle on
 * behalf of many agents. Optional — unless the vault was opened with a
 * `ScopePolicy`, which has nothing to key on without it and refuses the call.
 */
export type VaultContext = { agent: string; source?: string };

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
 * stale index — none of them are trusted to be a safe path. The gate's rail,
 * and the facade's `get` reads through it too, so the paths a caller may read
 * and the paths it may write are decided by one function; the errors say
 * "vault", not "write gate", for that reason.
 */
export function confinedPath(root: string, rel: string): string {
  // No path holds a NUL, and `lstat` answers one with a raw TypeError at
  // whoever called us — a caller's namespace reaches here, so the refusal is
  // the vault's own error like every other bad path below.
  if (rel.includes("\0")) throw new Error(`vault: ${safe(rel)} contains a NUL byte`);
  const base = resolve(root);
  const abs = resolve(base, rel);
  if (abs !== base && !abs.startsWith(base + sep)) {
    throw new Error(`vault: ${rel} resolves outside the vault`);
  }
  let walk = base;
  // The root itself is not walked (`relative` gives one empty segment when the
  // path *is* the root): the caller named it, and the scan does not lstat it
  // either — a vault opened through a symlinked root is still a vault.
  for (const segment of relative(base, abs).split(sep).filter((s) => s !== "")) {
    // The scan skips dot-directories, so a note written into one could never
    // be indexed — an invisible write is not a successful one — and nothing in
    // one is a note to read back either.
    if (segment.startsWith(".")) {
      throw new Error(`vault: ${rel} passes through a hidden directory`);
    }
    walk = join(walk, segment);
    // A symlink is a second name for a file that may not be a note at all —
    // the scan skips them too, so the gate must not write through one either.
    if (lstatSync(walk, { throwIfNoEntry: false })?.isSymbolicLink()) {
      throw new Error(`vault: ${rel} passes through a symlink`);
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
 * Exported because the consolidation pass quotes note bodies into a prompt of
 * its own, and one escaping rule is the only kind that cannot go stale.
 */
export const fence = (text: string): string =>
  text.trim().replace(/^---[ \t]*(begin|end)/gim, " $&");

/** The prompt an LLM decider gets. `parseDecision` reads what it asks for. */
export function gatePrompt({ candidate, similar }: DeciderInput): string {
  const notes =
    similar.length === 0
      ? "(none — the vault has nothing similar)"
      : similar
          .map(
            (s, i) =>
              `[${i + 1}] path: ${s.note.path}${s.readOnly ? " (read-only)" : ""}\n` +
              `    title: ${s.note.title}\n` +
              `--- begin note ---\n${fence(s.note.body)}\n--- end note ---`,
          )
          .join("\n\n");
  // A hint to the model, never the enforcement: a path or title is unfenced
  // text, so a note could write ` (read-only)` into its own title, and nothing
  // stops a decider from targeting a marked note anyway. The rail is the write
  // check `propose` makes on `decision.target` — which is why neither is
  // exploitable. Said only when it applies: a scope-free vault should not have
  // the model reading a rule about markings it will never see.
  const readOnly = similar.some((s) => s.readOnly)
    ? "\nA note marked read-only is outside the calling agent's write scope: do not name it as a target — choose create instead.\n"
    : "";
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
${readOnly}
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
 *
 * `scope` carries the caller's identity for this one call — given, its
 * provenance is stamped into every note the gate *authors* — and what that
 * caller may touch: the candidate's namespace is write-checked before the
 * decider runs, only readable notes are shown to it, and a decision targeting
 * a note the agent may not write falls back to `create` (§ Scopes and context).
 */
export async function propose(
  db: Database,
  root: string,
  embedder: Embedder,
  candidate: Candidate,
  { decider, cutoffs, n = 5 }: GateOptions,
  scope: Scope = ALLOW_ALL,
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
  const ctx = scope.ctx;
  // A blank agent is a caller that meant to identify itself and did not. It
  // would be stamped into files as an empty string and match no scope rule —
  // refuse it here, before any write.
  if (ctx !== undefined && ctx.agent.trim() === "") {
    throw new Error("write gate: ctx.agent must name the calling agent");
  }
  const base = resolve(root);
  // Both refusals a doomed write can earn, before the decider: no model spend
  // on a namespace the gate would not write into anyway. Confinement first, so
  // a namespace that is not a path at all is named as that rather than as a
  // scope decision.
  //
  // Canonicalized *through* that rail and used from here on, because checking
  // one spelling and writing another is a scope bypass: `notes/../ledger` is
  // inside the vault and starts with `notes/`, so a raw check passes — and the
  // file lands in `ledger/`. Scope rules end at a segment boundary, so the
  // namespace alone decides the answer for every filename the gate could pick
  // inside it.
  const namespace = normalizePrefix(
    relative(base, confinedPath(base, candidate.namespace ?? "")).replaceAll("\\", "/"),
  );
  if (!scope.may("write", namespace)) {
    throw new Error(
      `write gate: ${JSON.stringify(safe(ctx?.agent ?? ""))} may not write to ` +
        (namespace === "" ? "the vault root" : safe(namespace)),
    );
  }
  let applied: Applied | null = null;
  let fellBack = false;
  for (let attempt = 0; attempt < 2 && applied === null; attempt++) {
    // Re-running the gate means re-running it against *fresh* state: the edit
    // that aborted the first attempt is on disk but not yet in the index.
    if (attempt > 0) await reindex(db, base, embedder);
    const similar = await findSimilar(db, base, embedder, candidate, cutoffs, n, scope);
    const decision = checkDecision(await decider({ candidate, similar }));
    if (decision.target !== undefined && !similar.some((s) => s.note.path === decision.target)) {
      // The only paths the gate will touch are ones it just read itself.
      throw new Error(
        `write gate: decider targeted ${decision.target}, which was not among the similar notes`,
      );
    }
    // A target the agent may read but not write falls back to `create` below,
    // like one that failed check-and-write twice — and re-asking the decider
    // would only spend another model call on the same forbidden note.
    if (decision.target !== undefined && !scope.may("write", decision.target)) break;
    applied = await apply(db, base, candidate, namespace, decision, similar, ctx);
  }
  if (applied === null) {
    applied = await create(db, base, candidate, namespace, undefined, ctx);
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

/**
 * Top-k similar notes, read from disk so their hashes are current. The search
 * is scoped, so an agent never has another agent's note bodies quoted back to
 * it by the prompt; one it may read but not write is flagged read-only.
 */
async function findSimilar(
  db: Database,
  root: string,
  embedder: Embedder,
  candidate: Candidate,
  cutoffs: Cutoffs,
  n: number,
  scope: Scope,
): Promise<SimilarNote[]> {
  // Same shape the indexer embeds a note with, so like is compared with like.
  const hits = await hybridSearch(
    db,
    embedder,
    `${candidate.title}\n\n${candidate.body}`,
    { n, cutoffs },
    scope,
  );
  const similar: SimilarNote[] = [];
  for (const hit of hits) {
    // The search already filtered these in SQL. Checked again here because this
    // is where note *bodies* leave the vault and enter a prompt: the highest
    // consequence of a filter that ever goes wrong, and the cheapest place to
    // not depend on one.
    if (!scope.may("read", hit.path)) continue;
    confinedPath(root, hit.path); // the index is derived data, not a trusted path source
    const raw = await readRaw(root, hit.path);
    if (raw === null) continue; // indexed but gone: files are truth, the row is stale
    const note = parseNote(raw, hit.path);
    similar.push({
      note,
      score: hit.score,
      hash: note.hash,
      ...(!scope.may("write", hit.path) && { readOnly: true }),
    });
  }
  return similar;
}

/**
 * Apply one decision, or return null if the target changed under us.
 * `namespace` is the canonical, already-write-checked one from `propose` — the
 * candidate's own string is never joined into a path again.
 */
async function apply(
  db: Database,
  root: string,
  candidate: Candidate,
  namespace: string,
  decision: Decision,
  similar: SimilarNote[],
  ctx?: VaultContext,
): Promise<Applied | null> {
  if (decision.action === "discard") {
    // The justification rides with the entry: which notes the decider judged
    // this candidate against, at which hash and score. Written now because it
    // cannot be retrofitted onto lines logged without it — staleness detection
    // (#34) consumes it later.
    logCandidate(root, candidate, {
      decision,
      similar: similar.map(({ note, hash, score }) => ({ path: note.path, hash, score })),
    });
    return { action: "discard" };
  }
  if (decision.action === "create") {
    return await create(db, root, candidate, namespace, decision.body, ctx);
  }

  const hit = similar.find((s) => s.note.path === decision.target)!; // checked in propose
  const rel = hit.note.path;
  const abs = confinedPath(root, rel);
  // Check-and-write: the file must still be byte-for-byte what the decider saw.
  const raw = await readRaw(root, rel);
  if (raw === null || parseNote(raw, rel).hash !== hit.hash) return null;

  if (decision.action === "update") {
    // Frontmatter is the note's identity — keep it verbatim, bump `updated`
    // and record who wrote it. Both are textual patches: we did not author
    // this note, so it is never re-serialized. Every gate-owned key is set or
    // *unset* to match this call exactly — a leftover `vault_source` beside a
    // new `vault_agent` would assert a pairing that never happened.
    const stamp = provenance(ctx);
    let patched = patchFrontmatter(raw, "updated", now());
    for (const key of PROVENANCE_KEYS) {
      patched = patchFrontmatter(patched, key, stamp[key] ?? null);
    }
    await writeAtomic(abs, replaceBody(patched, decision.body ?? candidate.body));
    return { action: "update", path: rel };
  }

  // supersede: the successor is written first, then the old note is marked and
  // linked forward to it.
  const created = await create(db, root, candidate, namespace, decision.body, ctx);
  return {
    action: "supersede",
    path: created.path,
    ...(await markSuperseded(root, { path: rel, hash: hit.hash }, created.path!)),
  };
}

/**
 * Retire one note in favour of another: mark it `superseded_by` and link it
 * forward. A textual patch, never a re-serialization — we did not author this
 * note — and marked, *not* restamped: the marking is bookkeeping, not
 * authorship, and the agent that wrote the successor is already recorded there.
 *
 * Check-and-write, like every other touch: `hash` is what the caller read the
 * note at, and writing the successor took time. If the note moved in that
 * window it comes back `unmarked` and untouched — the successor stands, and a
 * human's edit does not get overwritten to tidy up our own bookkeeping.
 *
 * Shared with the consolidation pass, which marks N members of a cluster
 * against one merged note (DESIGN.md § Consolidation pass).
 */
export async function markSuperseded(
  root: string,
  note: { path: string; hash: string },
  successor: string,
): Promise<{ superseded: string } | { unmarked: string }> {
  const abs = confinedPath(root, note.path);
  const current = await readRaw(root, note.path);
  if (current === null || parseNote(current, note.path).hash !== note.hash) {
    return { unmarked: note.path };
  }
  const marked = patchFrontmatter(current, "superseded_by", successor);
  // The caller knows the exact path, so it links by path rather than by stem: a
  // namespaced successor cannot go ambiguous later. (A successor written to the
  // vault root has no qualified form, so its link is a bare stem and *can* —
  // DESIGN.md § Data model.)
  const link = `Superseded by [[${linkTarget(successor)}]].\n`;
  await writeAtomic(abs, marked.endsWith("\n") ? `${marked}\n${link}` : `${marked}\n\n${link}`);
  return { superseded: note.path };
}

/**
 * Write through a temp file and rename it over the target. A reader never sees
 * a half-written note, a crash never truncates one, and — since rename replaces
 * a symlink instead of following it — nothing can be swapped in between
 * `confinedPath` and the write. The temp name is not `.md`, so a crashed write
 * leaves nothing the scan would index. Exported for the indexer's collision
 * auto-qualify pass, which rewrites linking notes through this same rail.
 */
export async function writeAtomic(abs: string, text: string): Promise<void> {
  const tmp = `${abs}.tmp-${process.pid}-${Bun.randomUUIDv7().slice(-8)}`;
  await Bun.write(tmp, text);
  renameSync(tmp, abs);
}

/**
 * The frontmatter keys the gate owns. Namespaced (`vault_`) because `agent:`
 * and `source:` are exactly the keys a human's own frontmatter plausibly holds,
 * and the textual patcher replaces *top-level* lines — patching a human's
 * nested `source:` block would orphan its children into a parse error. Listed
 * rather than derived, because `update` has to clear the ones this call does
 * not set, which means knowing them all.
 */
const PROVENANCE_KEYS = ["vault_agent", "vault_source"] as const;

/**
 * The provenance for a note the gate authors or rewrites, or `{}` when the
 * caller gave no context. Both values are single-line, so `update`'s textual
 * patch applies cleanly. `vault_agent` answers "which agent last wrote this
 * note through the gate".
 */
function provenance(ctx?: VaultContext): Record<string, string> {
  if (ctx === undefined) return {};
  return { vault_agent: ctx.agent, ...(ctx.source !== undefined && { vault_source: ctx.source }) };
}

/**
 * Write a note we authored ourselves — the one place `serializeNote` is used.
 * Exported for the consolidation pass, which writes its merged note through
 * this same rail: slug confinement, collision suffixing, provenance stamping.
 * `namespace` is canonical (`normalizePrefix`), and the caller has already
 * decided the write is allowed — `create` does not scope-check.
 */
export async function create(
  db: Database,
  root: string,
  candidate: Candidate,
  namespace: string,
  body?: string,
  ctx?: VaultContext,
): Promise<Applied> {
  const { rel, abs } = freePath(db, root, candidate, namespace);
  const at = now();
  const text = serializeNote({
    frontmatter: {
      title: candidate.title,
      ...(candidate.type !== undefined && { type: candidate.type }),
      created: at,
      updated: at,
      ...provenance(ctx),
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
function freePath(
  db: Database,
  root: string,
  candidate: Candidate,
  namespace: string,
): { rel: string; abs: string } {
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
    // `namespace` is canonical and ends in `/` (or is empty, at the root), and
    // it is the string `propose` write-checked — so the file lands exactly
    // where the scope decision was made, and `rel` is the identity the caller
    // gets back and the wikilink a supersede writes.
    const rel = `${namespace}${slug}.md`;
    const abs = confinedPath(root, rel);
    if (!existsSync(abs) && taken.get(slug) === null) return { rel, abs };
  }
  // Nowhere to put it is still not a reason to drop it on the floor.
  const reason = `write gate: no free filename for ${JSON.stringify(candidate.title)}`;
  // similar is [] here not because nothing was similar but because this line is
  // a placement failure, not a decision — the field stays uniform either way.
  logCandidate(root, candidate, { reason, similar: [] });
  throw new Error(reason);
}

/**
 * The discard log: durable history, so it sits beside the notes rather than in
 * `.vault/`, which is a disposable index directory — a `doctor --rebuild` or a
 * nuke of `.vault/` must not take it. A dot-file, so the scan never indexes it.
 * It carries whole candidate bodies, so a vault kept in git may want it in
 * `.gitignore`.
 */
export const discardLog = (root: string): string => join(root, ".discarded.log");

/**
 * Rotation cap (#33): bounded growth, nothing deleted. At this size the live
 * log moves aside to `.discarded.<n>.log` and a fresh one starts; the read
 * side (`discards.ts`) numbers entries across all of them.
 * ponytail: a constant. An env knob or an option if a vault ever needs one.
 */
export const DISCARD_LOG_CAP = 5 * 1024 * 1024;

/**
 * Append a candidate to the discard log as JSONL, whole: a wrong decider call —
 * or a gate that cannot place the note — costs a line in a log, never the
 * information itself.
 *
 * `O_NOFOLLOW`, because this is the gate's one write to a fixed, user-visible
 * path — every other write lands through a rename, which replaces a symlink
 * rather than following it. A symlink dropped here would redirect whole
 * candidate bodies out of the vault; opening one fails instead (ELOOP).
 *
 * Exported for the discards tests, which are what `cap` is a parameter for —
 * production callers never pass one.
 */
export function logCandidate(
  root: string,
  candidate: Candidate,
  extra: Record<string, unknown>,
  cap = DISCARD_LOG_CAP,
): void {
  const path = discardLog(root);
  // lstat, not stat: a symlink's own (tiny) size never trips a rotation, so a
  // planted link still reaches the O_NOFOLLOW refusal below instead of being
  // quietly renamed into history as if it were the log.
  if ((lstatSync(path, { throwIfNoEntry: false })?.size ?? -1) >= cap) {
    // Rotate to the next free suffix — .1 is the oldest, nothing is deleted or
    // shifted, and old entries stay exactly where a reader last saw them.
    let n = 0;
    for (const f of readdirSync(root)) {
      const m = /^\.discarded\.(\d+)\.log$/.exec(f);
      if (m !== null) n = Math.max(n, Number(m[1]));
    }
    renameSync(path, join(root, `.discarded.${n + 1}.log`));
  }
  // First write to a fresh log (including the one a rotation just started):
  // make sure a committed vault does not carry refused note bodies into git
  // history. Only then — a user who strips the line while the log exists has
  // decided, and drift-repairing their .gitignore is not this function's call.
  if (!existsSync(path)) ensureGitignore(root);
  const { O_WRONLY, O_APPEND, O_CREAT, O_NOFOLLOW } = constants;
  const fd = openSync(path, O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW);
  try {
    writeSync(fd, `${JSON.stringify({ at: now(), candidate, ...extra })}\n`);
  } finally {
    closeSync(fd);
  }
}

/** The one pattern that covers the live log and every rotation of it. */
const IGNORE_LINE = ".discarded.log*";

/**
 * Ensure `<root>/.gitignore` covers the discard log. Append-only and
 * line-checked, so it lands exactly once and never rewrites what a human keeps
 * in that file. The same `O_NOFOLLOW` rail as the log itself: this is the
 * gate's other fixed-path write, and a symlink here would redirect it.
 */
function ensureGitignore(root: string): void {
  const { O_RDONLY, O_WRONLY, O_APPEND, O_CREAT, O_NOFOLLOW } = constants;
  const path = join(root, ".gitignore");
  let current = "";
  let fd = -1;
  try {
    fd = openSync(path, O_RDONLY | O_NOFOLLOW);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }
  if (fd >= 0) {
    try {
      current = readFileSync(fd, "utf8");
    } finally {
      closeSync(fd);
    }
  }
  if (current.split("\n").some((line) => line.trim() === IGNORE_LINE)) return;
  const out = openSync(path, O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW);
  try {
    // A file someone left without a trailing newline must not fuse its last
    // line with ours into a pattern neither of us wrote.
    writeSync(out, `${current === "" || current.endsWith("\n") ? "" : "\n"}${IGNORE_LINE}\n`);
  } finally {
    closeSync(out);
  }
}

const now = (): string => new Date().toISOString();

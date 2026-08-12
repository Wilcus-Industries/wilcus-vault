// A note is one .md file: `---` YAML frontmatter + markdown body. Files are
// truth, so parsing never throws — a note with broken frontmatter still indexes
// (whole file as body) and carries a warning flag for `doctor` to report.
import { basename } from "node:path";

export type Note = {
  /** vault-relative path — the note's identity */
  path: string;
  /** filename stem — what a bare `[[stem]]` link resolves against */
  slug: string;
  title: string;
  type: string | undefined;
  frontmatter: Record<string, unknown>;
  body: string;
  /**
   * Deduped wikilink targets as written, in first-seen order: a bare filename
   * stem (`[[acme]]`) or a vault-relative path without its extension
   * (`[[customers/acme]]`). Resolution is `resolveEdges`' job — a target is
   * matched against the indexed note paths, never joined onto the filesystem.
   */
  links: string[];
  /**
   * sha256 (hex) of `raw` re-encoded as UTF-8 — the decoded string, not the
   * bytes on disk, so an encoding quirk the decoder normalizes away is
   * invisible here. It is a dirtiness check for reindexing, not an identity:
   * identity is `path`.
   */
  hash: string;
  /**
   * Frontmatter the parser could not use as written: unterminated or malformed
   * block, non-mapping value, alias expansion over budget, or a non-string
   * `title`/`type`. The note still indexes; `doctor` reports the flag.
   */
  malformedFrontmatter: boolean;
};

/**
 * The path-qualified wikilink target of a vault-relative path: the path minus
 * `.md`. What the write gate links with and what `doctor` offers as the fix for
 * an ambiguous link, so both say the same thing. A note at the vault root has
 * no qualified form — its path minus `.md` *is* its bare stem.
 */
export const linkTarget = (rel: string): string => rel.slice(0, -".md".length);

/**
 * Can this link target survive a round trip through `[[…]]`? A `[`, `]` or `|`
 * is legal in a filename but destroys or misdirects the link when spliced into
 * one (`[[[archive]/acme]]` parses as no edge at all; `[[my|dir/acme]]` parses
 * as link `my` with an alias), and `linkOf` trims, so leading or trailing
 * whitespace does not survive either (`[[ archive/acme]]` reads back as a
 * different note's path). A rewrite that cannot round-trip is never made.
 */
export const isWritableTarget = (target: string): boolean =>
  !/[\[\]\r\n]/.test(target) && linkOf(target) === target;

// The one wikilink scanner: `parseNote` finds edges with it and `qualifyLinks`
// rewrites with it, so what counts as a link and what gets rewritten cannot
// disagree. Single-line and length-capped: a stray `[[` must not pair with a
// `]]` pages later and invent an edge.
const WIKILINK = /\[\[([^\[\]\n]{1,256})\]\]/g;

/** A match's link target: the text before any `|alias`, trimmed. */
const linkOf = (inner: string): string => inner.split("|")[0]!.trim();

/** Split a leading `---` block off the top of the file. */
function splitFrontmatter(text: string): {
  yaml: string | null;
  body: string;
  unterminated: boolean;
} {
  const none = { yaml: null, body: text };
  if (!text.startsWith("---\n")) return { ...none, unterminated: false };
  const rest = text.slice(4);
  const close = rest.search(/^---[ \t]*$/m);
  // No closing fence (or one with trailing junk): keep the whole file as body,
  // but flag it — a mangled fence would silently drop superseded_by otherwise.
  if (close === -1) return { ...none, unterminated: true };
  return {
    yaml: rest.slice(0, close),
    body: rest.slice(close).replace(/^---[ \t]*\n?/, ""),
    unterminated: false,
  };
}

/**
 * YAML aliases re-expand on every reference, so a 250-byte billion-laughs note
 * parses into megabytes. Count every visit (shared refs accumulate) and bail
 * once the object is too big to belong in an index row.
 */
function withinNodeBudget(value: unknown, budget = 4096): boolean {
  let seen = 0;
  const visit = (v: unknown): boolean => {
    if (++seen > budget) return false;
    if (v && typeof v === "object") {
      for (const child of Object.values(v)) if (!visit(child)) return false;
    }
    return true;
  };
  return visit(value);
}

/**
 * A frontmatter block's YAML as a mapping we can index, or null when it is
 * unusable: invalid YAML, not a mapping, or over the alias budget. Both readers
 * of a block — `parseNote` and the textual patcher below — decide through this
 * one function, so they can never disagree about whether a block is real. (They
 * did once: a fenced-but-invalid block looked patchable while `parseNote`
 * ignored it, so a `superseded_by` written into it was never read.)
 */
function usableFrontmatter(yaml: string): Record<string, unknown> | null {
  if (yaml.trim() === "") return {};
  try {
    const parsed = Bun.YAML.parse(yaml);
    // Must be a mapping, and small enough to store in an index row.
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) && withinNodeBudget(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

export function parseNote(raw: string, relPath: string): Note {
  const hash = new Bun.CryptoHasher("sha256").update(raw).digest("hex");
  const lf = raw.replaceAll("\r\n", "\n");
  const text = lf.charCodeAt(0) === 0xfeff ? lf.slice(1) : lf; // strip UTF-8 BOM
  const slug = basename(relPath, ".md");

  const { yaml, body: split, unterminated } = splitFrontmatter(text);
  let frontmatter: Record<string, unknown> = {};
  let body = text;
  let malformedFrontmatter = unterminated;
  if (yaml !== null) {
    const parsed = usableFrontmatter(yaml);
    if (parsed === null) malformedFrontmatter = true;
    else {
      frontmatter = parsed;
      body = split;
    }
  }

  const fmTitle = frontmatter["title"];
  const fmType = frontmatter["type"];
  // Present but not a string: unusable, and doctor should hear about it.
  if ((fmTitle !== undefined && typeof fmTitle !== "string") ||
      (fmType !== undefined && typeof fmType !== "string")) {
    malformedFrontmatter = true;
  }

  // ponytail: regex scan, no markdown parse — wikilinks and headings inside code
  // fences are counted (documented MVP simplification); parse structure if it
  // bites. `/` is ordinary content here — a path-qualified target travels whole
  // and is resolved as SQL, never as a filesystem path.
  const links = [
    ...new Set(Array.from(body.matchAll(WIKILINK), (m) => linkOf(m[1]!))),
  ].filter((s) => s !== "");

  return {
    path: relPath,
    slug,
    title:
      (typeof fmTitle === "string" && fmTitle.trim() !== "" ? fmTitle : undefined) ??
      body.match(/^#[ \t]+(.+?)[ \t]*$/m)?.[1] ??
      slug,
    type: typeof fmType === "string" ? fmType : undefined,
    frontmatter,
    body,
    links,
    hash,
    malformedFrontmatter,
  };
}

/**
 * Render a note we authored. **Never use this to rewrite a human-authored
 * file:** a YAML round-trip is lossy against hand-written data (comments are
 * dropped, `01234` becomes `1234`, `1.0` becomes `1`). Per DESIGN.md § Write
 * gate, edits to existing notes are textual patches of the frontmatter block.
 */
export function serializeNote(note: Pick<Note, "frontmatter" | "body">): string {
  if (Object.keys(note.frontmatter).length === 0) return note.body;
  const yaml = Bun.YAML.stringify(note.frontmatter, null, 2).replace(/\n*$/, "\n");
  return `---\n${yaml}---\n${note.body}`;
}

/**
 * The frontmatter block as *text*: where its YAML starts and ends inside `raw`,
 * and where the body begins after the closing fence. Offsets are into `raw`, so
 * a leading BOM and CRLF endings are stepped over rather than normalized away.
 * Null when `parseNote` would not use the block — no opening fence, no closing
 * fence, or YAML it cannot read — so text-level edits and the parsed view never
 * disagree about what counts as frontmatter.
 */
function frontmatterBlock(
  raw: string,
): { start: number; end: number; bodyStart: number } | null {
  const bom = raw.charCodeAt(0) === 0xfeff ? 1 : 0;
  const open = /^---\r?\n/.exec(raw.slice(bom));
  if (open === null) return null;
  const start = bom + open[0].length;
  const close = /^---[ \t]*\r?(\n|$)/m.exec(raw.slice(start));
  if (close === null) return null;
  const at = {
    start,
    end: start + close.index,
    bodyStart: start + close.index + close[0].length,
  };
  const yaml = raw.slice(at.start, at.end).replaceAll("\r\n", "\n");
  return usableFrontmatter(yaml) === null ? null : at;
}

/**
 * Set one frontmatter key by editing the text, never by re-serializing: append
 * the key's line, or replace the single line that already defines it, leaving
 * every other byte of the file identical. Per DESIGN.md § Write gate this is
 * how the gate touches notes it did not author — a YAML round-trip drops
 * comments and rewrites `01234` and `1.0`. A file with no usable frontmatter
 * block (absent, or malformed enough that `parseNote` ignores it) gets a fresh
 * block prepended; its content is left untouched below.
 *
 * A `null` value **unsets** the key instead — every line defining it goes. A
 * gate-owned key has to be removable, or it would outlive the fact it records
 * (a stale `vault_source` beside a fresh `vault_agent` asserts a pairing that
 * never happened).
 */
export function patchFrontmatter(raw: string, key: string, value: string | null): string {
  // Keys are ours (`superseded_by`, `updated`); anything else could be YAML or
  // regex syntax rather than a key, and this function does not sanitize.
  if (!/^[A-Za-z0-9_-]+$/.test(key)) throw new Error(`patchFrontmatter: unusable key ${key}`);
  const at = frontmatterBlock(raw);
  if (value === null) {
    // Unsetting is only ever a deletion: a file with no usable block does not
    // gain one to hold the absence of a key, and an absent key is already gone.
    if (at === null) return raw;
    const gone = raw
      .slice(at.start, at.end)
      .replace(new RegExp(`^${key}[ \t]*:[^\r\n]*\r?\n?`, "gm"), "");
    return raw.slice(0, at.start) + gone + raw.slice(at.end);
  }
  // A timestamp is a YAML-native scalar and reads as one; anything else is
  // quoted, so a path or a colon can never become syntax.
  const line = `${key}: ${ISO_TIMESTAMP.test(value) ? value : JSON.stringify(value)}`;
  if (at === null) return `---\n${line}\n---\n${raw}`;

  const eol = raw.slice(0, at.start).endsWith("\r\n") ? "\r\n" : "\n";
  const block = raw.slice(at.start, at.end);
  // Top-level only: an indented `key:` belongs to a nested mapping. `[^\r\n]*`
  // rather than `.*` so a CRLF file keeps its carriage return. Every occurrence
  // is taken: YAML's last-wins would otherwise let a duplicate key downstream
  // resurrect the value we just replaced.
  const existing = new RegExp(`^${key}[ \t]*:[^\r\n]*`, "gm");
  let seen = 0;
  const swapped = block.replace(existing, () => (++seen === 1 ? line : ""));
  const patched =
    seen > 0 ? swapped : block + (block === "" || block.endsWith("\n") ? "" : eol) + line + eol;
  return raw.slice(0, at.start) + patched + raw.slice(at.end);
}

/** `2026-08-01T09:41:00.000Z` — what `new Date().toISOString()` produces. */
const ISO_TIMESTAMP = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/;

/**
 * Rewrite every bare `[[stem]]` / `[[stem|alias]]` link in `raw`'s body to the
 * path-qualified `target`, leaving every other byte — the frontmatter block,
 * CRLF endings, a BOM — untouched: this edits files the vault did not author.
 * The scan is the parser's own link regex over the same body region `parseNote`
 * reads (`frontmatterBlock` decides both), so what this rewrites and what
 * counts as an edge cannot disagree — which also means links inside code fences
 * are rewritten, the parser's documented MVP simplification. Alias text travels
 * verbatim. Used by the indexer when a stem collision is created (DESIGN.md
 * § Data model).
 */
export function qualifyLinks(raw: string, stem: string, target: string): string {
  const start = frontmatterBlock(raw)?.bodyStart ?? 0;
  // The link regex never spans a line, so a CRLF file's `\r`s all sit outside
  // any match and survive; the frontmatter's bytes are not scanned at all.
  const body = raw.slice(start).replace(WIKILINK, (whole, inner: string) => {
    if (linkOf(inner) !== stem) return whole;
    const pipe = inner.indexOf("|");
    return `[[${target}${pipe === -1 ? "" : inner.slice(pipe)}]]`;
  });
  return raw.slice(0, start) + body;
}

/** Replace the body, keeping the frontmatter block byte-identical. */
export function replaceBody(raw: string, body: string): string {
  const at = frontmatterBlock(raw);
  // No usable block ⇒ the whole file is the body, as `parseNote` reads it.
  return at === null ? body : raw.slice(0, at.bodyStart) + body;
}

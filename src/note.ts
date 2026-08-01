// A note is one .md file: `---` YAML frontmatter + markdown body. Files are
// truth, so parsing never throws — a note with broken frontmatter still indexes
// (whole file as body) and carries a warning flag for `doctor` to report.
import { basename } from "node:path";

export type Note = {
  /** vault-relative path — the note's identity */
  path: string;
  /** filename stem — the wikilink target */
  slug: string;
  title: string;
  type: string | undefined;
  frontmatter: Record<string, unknown>;
  body: string;
  /** deduped wikilink slugs, in first-seen order */
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
    if (yaml.trim() === "") {
      body = split;
    } else {
      try {
        const parsed = Bun.YAML.parse(yaml);
        // Must be a mapping, and small enough to store in an index row.
        if (
          parsed &&
          typeof parsed === "object" &&
          !Array.isArray(parsed) &&
          withinNodeBudget(parsed)
        ) {
          frontmatter = parsed as Record<string, unknown>;
          body = split;
        } else {
          malformedFrontmatter = true;
        }
      } catch {
        malformedFrontmatter = true;
      }
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
  // fences are counted (documented MVP simplification); parse structure if it bites.
  // Single-line and length-capped: a stray `[[` must not pair with a `]]` pages
  // later and invent an edge.
  const links = [
    ...new Set(
      Array.from(body.matchAll(/\[\[([^\[\]\n]{1,256})\]\]/g), (m) =>
        m[1]!.split("|")[0]!.trim(),
      ),
    ),
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
 * and where the body begins after the closing fence. Null when there is no
 * usable block — which is exactly when `parseNote` treats the whole file as
 * body, so the two agree on what "has frontmatter" means.
 */
function frontmatterBlock(
  raw: string,
): { start: number; end: number; bodyStart: number } | null {
  const open = /^---\r?\n/.exec(raw);
  if (open === null) return null;
  const start = open[0].length;
  const close = /^---[ \t]*\r?(\n|$)/m.exec(raw.slice(start));
  if (close === null) return null;
  return {
    start,
    end: start + close.index,
    bodyStart: start + close.index + close[0].length,
  };
}

/**
 * Set one frontmatter key by editing the text, never by re-serializing: append
 * the key's line, or replace the single line that already defines it, leaving
 * every other byte of the file identical. Per DESIGN.md § Write gate this is
 * how the gate touches notes it did not author — a YAML round-trip drops
 * comments and rewrites `01234` and `1.0`. A file with no usable frontmatter
 * block (absent, or malformed enough that `parseNote` ignores it) gets a fresh
 * block prepended; its content is left untouched below.
 */
export function patchFrontmatter(raw: string, key: string, value: string): string {
  // Keys are ours (`superseded_by`, `updated`); anything else could be YAML or
  // regex syntax rather than a key, and this function does not sanitize.
  if (!/^[A-Za-z0-9_-]+$/.test(key)) throw new Error(`patchFrontmatter: unusable key ${key}`);
  const line = `${key}: ${JSON.stringify(value)}`; // JSON strings are valid YAML scalars
  const at = frontmatterBlock(raw);
  if (at === null) return `---\n${line}\n---\n${raw}`;

  const eol = raw.slice(0, at.start).endsWith("\r\n") ? "\r\n" : "\n";
  const block = raw.slice(at.start, at.end);
  // Top-level only: an indented `key:` belongs to a nested mapping. `[^\r\n]*`
  // rather than `.*` so a CRLF file keeps its carriage return.
  const existing = new RegExp(`^${key}[ \t]*:[^\r\n]*`, "m");
  const patched = existing.test(block)
    ? block.replace(existing, line)
    : block + (block === "" || block.endsWith("\n") ? "" : eol) + line + eol;
  return raw.slice(0, at.start) + patched + raw.slice(at.end);
}

/** Replace the body, keeping the frontmatter block byte-identical. */
export function replaceBody(raw: string, body: string): string {
  const at = frontmatterBlock(raw);
  // No usable block ⇒ the whole file is the body, as `parseNote` reads it.
  return at === null ? body : raw.slice(0, at.bodyStart) + body;
}

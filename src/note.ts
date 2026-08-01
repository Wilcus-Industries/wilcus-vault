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

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
  /** sha256 of the raw file, hex */
  hash: string;
  malformedFrontmatter: boolean;
};

/** Split a leading `---` block off the top of the file. */
function splitFrontmatter(text: string): { yaml: string | null; body: string } {
  if (!text.startsWith("---\n")) return { yaml: null, body: text };
  const rest = text.slice(4);
  const close = rest.search(/^---[ \t]*$/m);
  if (close === -1) return { yaml: null, body: text }; // unterminated: it's just prose
  return {
    yaml: rest.slice(0, close),
    body: rest.slice(close).replace(/^---[ \t]*\n?/, ""),
  };
}

export function parseNote(raw: string, relPath: string): Note {
  const hash = new Bun.CryptoHasher("sha256").update(raw).digest("hex");
  const text = raw.replaceAll("\r\n", "\n");
  const slug = basename(relPath, ".md");

  const { yaml, body: split } = splitFrontmatter(text);
  let frontmatter: Record<string, unknown> = {};
  let body = text;
  let malformedFrontmatter = false;
  if (yaml !== null) {
    if (yaml.trim() === "") {
      body = split;
    } else {
      try {
        const parsed = Bun.YAML.parse(yaml);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          frontmatter = parsed as Record<string, unknown>;
          body = split;
        } else {
          malformedFrontmatter = true; // frontmatter must be a mapping
        }
      } catch {
        malformedFrontmatter = true;
      }
    }
  }

  const fmTitle = frontmatter["title"];
  const fmType = frontmatter["type"];

  // ponytail: regex scan, no markdown parse — wikilinks and headings inside code
  // fences are counted (documented MVP simplification); parse structure if it bites.
  const links = [
    ...new Set(
      Array.from(body.matchAll(/\[\[([^\[\]]+)\]\]/g), (m) => m[1]!.split("|")[0]!.trim()),
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

export function serializeNote(note: Pick<Note, "frontmatter" | "body">): string {
  if (Object.keys(note.frontmatter).length === 0) return note.body;
  const yaml = Bun.YAML.stringify(note.frontmatter, null, 2).replace(/\n*$/, "\n");
  return `---\n${yaml}---\n${note.body}`;
}

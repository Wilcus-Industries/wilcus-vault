import { test, expect } from "bun:test";
import { parseNote, serializeNote } from "../src/note";

const sha256 = (s: string) => new Bun.CryptoHasher("sha256").update(s).digest("hex");

test("parses frontmatter, body, title, type, slug, hash", () => {
  const raw = `---
type: customer
created: 2026-07-31
title: Acme Corp
---
# Ignored Heading

Body text.
`;
  const n = parseNote(raw, "customers/acme.md");
  expect(n.path).toBe("customers/acme.md");
  expect(n.slug).toBe("acme");
  expect(n.title).toBe("Acme Corp"); // frontmatter wins over heading
  expect(n.type).toBe("customer");
  expect(n.frontmatter).toEqual({
    type: "customer",
    created: "2026-07-31",
    title: "Acme Corp",
  });
  expect(n.body).toBe("# Ignored Heading\n\nBody text.\n");
  expect(n.hash).toBe(sha256(raw));
  expect(n.malformedFrontmatter).toBe(false);
});

test("no frontmatter: whole file is body, title from first # heading", () => {
  const raw = "# Globex Renewal\n\nsome prose\n";
  const n = parseNote(raw, "notes/globex-renewal.md");
  expect(n.frontmatter).toEqual({});
  expect(n.body).toBe(raw);
  expect(n.title).toBe("Globex Renewal");
  expect(n.type).toBeUndefined();
  expect(n.malformedFrontmatter).toBe(false);
});

test("title falls back to filename stem", () => {
  expect(parseNote("just prose, no heading\n", "a/b/my-note.md").title).toBe("my-note");
  // ## is not an h1; still falls through to the stem
  expect(parseNote("## sub\n", "deep/sub-note.md").title).toBe("sub-note");
});

test("malformed YAML still indexes: whole file as body, warning flag, no throw", () => {
  const raw = `---
type: [unclosed
title: nope
---
# Real Heading

body
`;
  const n = parseNote(raw, "x/broken.md");
  expect(n.malformedFrontmatter).toBe(true);
  expect(n.frontmatter).toEqual({});
  expect(n.body).toBe(raw); // whole file, delimiters included
  expect(n.title).toBe("Real Heading");
  expect(n.hash).toBe(sha256(raw));
});

test("non-mapping frontmatter is malformed", () => {
  const n = parseNote("---\njust a scalar\n---\nbody\n", "x/scalar.md");
  expect(n.malformedFrontmatter).toBe(true);
  expect(n.frontmatter).toEqual({});
});

test("empty frontmatter block is not malformed", () => {
  const n = parseNote("---\n---\nbody\n", "x/empty.md");
  expect(n.malformedFrontmatter).toBe(false);
  expect(n.frontmatter).toEqual({});
  expect(n.body).toBe("body\n");
});

test("CRLF input parses and normalizes to LF", () => {
  const raw = "---\r\ntype: customer\r\n---\r\n# Title\r\n\r\nsee [[acme]]\r\n";
  const n = parseNote(raw, "x/crlf.md");
  expect(n.frontmatter).toEqual({ type: "customer" });
  expect(n.body).toBe("# Title\n\nsee [[acme]]\n");
  expect(n.title).toBe("Title");
  expect(n.links).toEqual(["acme"]);
  expect(n.hash).toBe(sha256(raw)); // hash is of the raw bytes, pre-normalization
});

test("wikilinks: aliases stripped, deduped, code fences counted", () => {
  const raw = `# Links

See [[acme]] and [[globex|Globex Inc]] and [[acme]] again.

\`\`\`md
[[in-a-fence]]
\`\`\`

[[ spaced ]] and [[]] and [[a|b|c]]
`;
  const n = parseNote(raw, "x/links.md");
  // MVP does not parse markdown structure: fenced wikilinks count (documented)
  expect(n.links).toEqual(["acme", "globex", "in-a-fence", "spaced", "a"]);
});

test("links come from the body only", () => {
  const n = parseNote("---\nsuperseded_by: old/[[trap]]\n---\n[[real]]\n", "x/fm.md");
  expect(n.links).toEqual(["real"]);
});

test("parse -> serialize -> parse round-trips", () => {
  const raw = `---
type: customer
tags:
  - a
  - b
superseded_by: null
---
# Acme

body with [[globex]]
`;
  const a = parseNote(raw, "customers/acme.md");
  const out = serializeNote(a);
  const b = parseNote(out, "customers/acme.md");
  expect(b.frontmatter).toEqual(a.frontmatter);
  expect(b.body).toBe(a.body);
  expect(b.title).toBe(a.title);
  expect(b.type).toBe(a.type);
  expect(b.links).toEqual(a.links);
  expect(b.malformedFrontmatter).toBe(false);
  expect(serializeNote(b)).toBe(out); // serialization is stable
});

test("serialize without frontmatter emits body only", () => {
  expect(serializeNote({ frontmatter: {}, body: "# Bare\n" })).toBe("# Bare\n");
});

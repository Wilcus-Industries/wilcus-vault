import { test, expect } from "bun:test";
import { parseNote, patchFrontmatter, qualifyLinks, replaceBody, serializeNote } from "../src/note";

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

test("wikilinks keep a path-qualified target whole", () => {
  const n = parseNote("[[customers/acme]] [[vendors/acme|the vendor]] [[acme]]\n", "x/q.md");
  expect(n.links).toEqual(["customers/acme", "vendors/acme", "acme"]);
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

test("alias bomb: expanded frontmatter over the node budget is malformed", () => {
  // Billion-laughs: aliases re-expand on every reference, so a tiny file blows
  // up into a multi-megabyte object that would detonate on JSON.stringify.
  const raw = `---
a: &a ["x","x","x","x","x","x","x","x","x"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: [*d,*d,*d,*d,*d,*d,*d,*d,*d]
---
body
`;
  const n = parseNote(raw, "x/bomb.md");
  expect(n.malformedFrontmatter).toBe(true);
  expect(n.frontmatter).toEqual({});
  expect(n.body).toBe(raw);
  expect(JSON.stringify(n).length).toBeLessThan(raw.length * 20); // stays small
});

test("ordinary nested frontmatter stays under the node budget", () => {
  const n = parseNote(
    "---\nmeta:\n  tags: [a, b, c]\n  nested:\n    k: v\n---\nbody\n",
    "x/nested.md",
  );
  expect(n.malformedFrontmatter).toBe(false);
  expect(n.frontmatter).toEqual({ meta: { tags: ["a", "b", "c"], nested: { k: "v" } } });
});

test("unterminated frontmatter is flagged", () => {
  // A mangled closing fence must not silently drop superseded_by from the index.
  const raw = "---\ntype: customer\nsuperseded_by: new/acme.md\n\nbody\n";
  const n = parseNote(raw, "x/unterminated.md");
  expect(n.malformedFrontmatter).toBe(true);
  expect(n.frontmatter).toEqual({});
  expect(n.body).toBe(raw);
});

test("closing fence with trailing junk does not close the block", () => {
  const raw = "---\ntype: customer\n--- oops\nbody\n";
  const n = parseNote(raw, "x/junk-fence.md");
  expect(n.malformedFrontmatter).toBe(true);
  expect(n.body).toBe(raw);
});

test("wikilinks never span newlines and are length-capped", () => {
  const n = parseNote("[[foo\nbar]]\n\n[[ok]]\n", "x/nl.md");
  expect(n.links).toEqual(["ok"]);

  const distant = parseNote("[[\n\nlots of prose\n\n]]\n", "x/distant.md");
  expect(distant.links).toEqual([]);

  const long = parseNote(`[[${"a".repeat(300)}]] and [[short]]\n`, "x/long.md");
  expect(long.links).toEqual(["short"]);
});

test("strips a leading UTF-8 BOM", () => {
  const n = parseNote("﻿---\ntype: customer\n---\n# Title\n", "x/bom.md");
  expect(n.frontmatter).toEqual({ type: "customer" });
  expect(n.malformedFrontmatter).toBe(false);
  expect(n.body).toBe("# Title\n");

  const noFm = parseNote("﻿# Heading\n", "x/bom2.md");
  expect(noFm.title).toBe("Heading");
  expect(noFm.body).toBe("# Heading\n");
});

test("non-string title/type are ignored and flagged for doctor", () => {
  const n = parseNote("---\ntitle: 42\ntype: [a, b]\n---\n# Heading\n", "x/badtypes.md");
  expect(n.title).toBe("Heading"); // falls through to the heading
  expect(n.type).toBeUndefined();
  expect(n.malformedFrontmatter).toBe(true);
  expect(n.frontmatter).toEqual({ title: 42, type: ["a", "b"] }); // kept as written
});

// The write gate patches frontmatter textually because a YAML round-trip is
// lossy against hand-written data (DESIGN.md § Write gate).
test("patchFrontmatter appends a key, leaving every other byte alone", () => {
  const raw = "---\ntype: customer\nid: 01234 # legacy\nrate: 1.0\n---\n# Acme\n\nbody\n";
  const out = patchFrontmatter(raw, "superseded_by", "notes/new.md");
  expect(out).toBe(
    `---\ntype: customer\nid: 01234 # legacy\nrate: 1.0\nsuperseded_by: "notes/new.md"\n---\n# Acme\n\nbody\n`,
  );
  expect(parseNote(out, "x/a.md").frontmatter["superseded_by"]).toBe("notes/new.md");
});

test("patchFrontmatter replaces exactly one existing top-level key line", () => {
  const raw = "---\nupdated: 2020-01-01\nmeta:\n  updated: nested\n---\nbody\n";
  const out = patchFrontmatter(raw, "updated", "2026-08-01");
  expect(out).toBe(`---\nupdated: "2026-08-01"\nmeta:\n  updated: nested\n---\nbody\n`);
});

test("patchFrontmatter prepends a block when there is no usable one", () => {
  for (const raw of [
    "# Acme\n\nbody\n", // no frontmatter at all
    "---\ntype: customer\n\nunterminated\n", // no closing fence
    // Fenced but not parseable: parseNote ignores the block and takes the whole
    // file as body, so patching *inside* it would write a key nothing reads.
    "---\ntype: [unclosed\n---\n# Acme\n\nbody\n",
    "---\n- a\n- b\n---\nbody\n", // a sequence, not a mapping
    "", // empty file
  ]) {
    const out = patchFrontmatter(raw, "superseded_by", "notes/new.md");
    expect(out).toBe(`---\nsuperseded_by: "notes/new.md"\n---\n${raw}`);
    expect(out.endsWith(raw)).toBe(true); // the body is never touched
    expect(parseNote(out, "x/a.md").frontmatter["superseded_by"]).toBe("notes/new.md");
  }
});

test("a duplicate key cannot survive the patch and win by YAML last-wins", () => {
  const out = patchFrontmatter(
    "---\nupdated: old\ntype: customer\nupdated: older\n---\nbody\n",
    "updated",
    "2026-08-01",
  );
  expect(parseNote(out, "x/a.md").frontmatter["updated"]).toBe("2026-08-01");
  expect(out).not.toContain("older");
  expect(out).toContain("type: customer");
});

test("patchFrontmatter unsets a key, and unsetting one that is absent is a no-op", () => {
  // A gate-owned key has to be removable, or it outlives the fact it records.
  const raw = `---\ntype: customer\nvault_agent: "a"\nid: 01234 # legacy\nvault_agent: "dupe"\n---\nbody\n`;
  expect(patchFrontmatter(raw, "vault_agent", null)).toBe(
    "---\ntype: customer\nid: 01234 # legacy\n---\nbody\n",
  );
  const clean = "---\ntype: customer\n---\nbody\n";
  expect(patchFrontmatter(clean, "vault_agent", null)).toBe(clean);
  // no usable block to remove a key from: prepending one would be absurd
  expect(patchFrontmatter("# Acme\n\nbody\n", "vault_agent", null)).toBe("# Acme\n\nbody\n");
  expect(patchFrontmatter("---\r\nvault_agent: \"a\"\r\ntype: c\r\n---\r\nb\r\n", "vault_agent", null)).toBe(
    "---\r\ntype: c\r\n---\r\nb\r\n",
  );
});

test("patchFrontmatter sees through a BOM instead of prepending past it", () => {
  const out = patchFrontmatter("﻿---\ntype: customer\n---\nbody\n", "updated", "2026-08-01");
  expect(out).toBe(`﻿---\ntype: customer\nupdated: "2026-08-01"\n---\nbody\n`);
  expect(replaceBody("﻿---\ntype: customer\n---\nold\n", "new\n")).toBe(
    "﻿---\ntype: customer\n---\nnew\n",
  );
});

test("an ISO timestamp is written as a YAML-native scalar, other values are quoted", () => {
  expect(patchFrontmatter("---\na: b\n---\nx", "updated", "2026-08-01T09:41:00.000Z")).toContain(
    "updated: 2026-08-01T09:41:00.000Z\n",
  );
  expect(patchFrontmatter("---\na: b\n---\nx", "superseded_by", "notes/new.md")).toContain(
    `superseded_by: "notes/new.md"\n`,
  );
});

test("patchFrontmatter keeps CRLF line endings and rejects an unusable key", () => {
  const out = patchFrontmatter("---\r\ntype: customer\r\n---\r\nbody\r\n", "updated", "2026-08-01");
  expect(out).toBe(`---\r\ntype: customer\r\nupdated: "2026-08-01"\r\n---\r\nbody\r\n`);
  expect(patchFrontmatter("---\r\nupdated: old\r\n---\r\nb\r\n", "updated", "x")).toBe(
    `---\r\nupdated: "x"\r\n---\r\nb\r\n`,
  );
  expect(() => patchFrontmatter("body", "a: b\nevil", "x")).toThrow(/key/);
});

test("replaceBody swaps the body and keeps the frontmatter block verbatim", () => {
  const raw = "---\ntype: customer\nid: 01234 # legacy\n---\nold body\n";
  expect(replaceBody(raw, "new body\n")).toBe(
    "---\ntype: customer\nid: 01234 # legacy\n---\nnew body\n",
  );
  // no usable block: the whole file is the body, exactly as parseNote sees it
  expect(replaceBody("# Acme\n\nold\n", "new\n")).toBe("new\n");
  expect(replaceBody("---\ntype: [unclosed\n---\nold\n", "new\n")).toBe("new\n");
});

test("qualifyLinks rewrites bare links to one stem, aliases kept, everything else verbatim", () => {
  const body =
    "See [[acme]] and [[acme|Acme Corp]] and [[ acme ]].\n" +
    "Not these: [[acmena]], [[customers/acme]], [[other]], plain acme.\n" +
    "```\ncode fence: [[acme]]\n```\n";
  expect(qualifyLinks(body, "acme", "customers/acme")).toBe(
    "See [[customers/acme]] and [[customers/acme|Acme Corp]] and [[customers/acme]].\n" +
      "Not these: [[acmena]], [[customers/acme]], [[other]], plain acme.\n" +
      // the parser counts fenced links as edges, so the rewrite matches it —
      // the documented MVP simplification
      "```\ncode fence: [[customers/acme]]\n```\n",
  );
  // an alias holding a pipe travels whole
  expect(qualifyLinks("[[acme|a|b]]", "acme", "customers/acme")).toBe("[[customers/acme|a|b]]");
  // nothing to do leaves the body byte-identical
  const untouched = "no links here, [[other]] only\n";
  expect(qualifyLinks(untouched, "acme", "customers/acme")).toBe(untouched);
});

test("qualifyLinks edits raw text in place: frontmatter, CRLF and BOM survive untouched", () => {
  // only the parser's body region is scanned — a decoy in frontmatter is YAML,
  // not a link, and `parseNote` would never edge it either
  const raw = '---\nnote: "[[acme]] is not a link here"\n---\nbody [[acme]]\n';
  expect(qualifyLinks(raw, "acme", "customers/acme")).toBe(
    '---\nnote: "[[acme]] is not a link here"\n---\nbody [[customers/acme]]\n',
  );
  // a CRLF file keeps every carriage return; a BOM stays where it was
  expect(qualifyLinks("﻿---\r\ntype: x\r\n---\r\nsee [[acme]]\r\nplain\r\n", "acme", "customers/acme")).toBe(
    "﻿---\r\ntype: x\r\n---\r\nsee [[customers/acme]]\r\nplain\r\n",
  );
  // malformed frontmatter is body, exactly as parseNote reads it
  expect(qualifyLinks("---\ntype: [unclosed\n---\n[[acme]]\n", "acme", "customers/acme")).toBe(
    "---\ntype: [unclosed\n---\n[[customers/acme]]\n",
  );
});

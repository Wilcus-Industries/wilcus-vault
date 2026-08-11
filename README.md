# @wilcus/vault

Files-are-truth markdown memory vault for AI agents: atomic notes, wikilink graph,
hybrid semantic + keyword search (sqlite-vec + FTS5 + reciprocal rank fusion), and a
write gate so nothing is silently overwritten. Open the vault in Obsidian or any
editor — the SQLite index is derived and disposable (`vault doctor --rebuild`).

Status: MVP. Standalone (zero wilcus dependencies), MIT. See DESIGN.md for the
architecture contract.

## Files are truth

The `.md` files on disk are the only source of truth. Everything in
`<vault>/.vault/index.db` — note rows, wikilink edges, FTS5 text, embeddings — is
derived and rebuildable, and no code path may treat it as authoritative. So:

- edit, move, rename or delete notes with any editor, or `git checkout` the whole
  vault, and nothing is lost — `vault doctor` reconciles the index to the files;
- delete `index.db` whenever you like; `vault doctor --rebuild` recreates it;
- the vault is a plain directory of markdown. Nothing here owns it.

## Install

Bun 1.3+ (this uses `bun:sqlite`, `Bun.YAML` and `Bun.CryptoHasher` — there is no
Node build). Not published to npm yet:

```
git clone https://github.com/CrazyWillBear/wilcus-vault && cd wilcus-vault
bun install
bun run src/cli.ts --help
```

For a `vault` on your PATH, `bun link` in the clone. As a library, depend on the
directory (`bun add file:../wilcus-vault`) and import from `@wilcus/vault`.

## CLI

```
vault reindex [--vault <dir>]            # index new and changed notes
vault doctor [--rebuild] [--vault <dir>] # check and repair the index
vault search <query> [--vault <dir>]     # hybrid search, best first
vault watch [--vault <dir>]              # index changes as they are saved
vault consolidate --ceiling <d>          # report near-duplicate clusters
vault --help                             # every command and flag
```

`--vault` defaults to the current directory; `--` ends flag parsing, so a query
may start with a dash. Exit code 0 on success, 1 on error — and 1 from `doctor`
when it found links only a human can fix: broken (nothing to point at) or
ambiguous (a bare `[[stem]]` several notes answer to — `doctor` prints the
candidate paths, and qualifying the link with one of them is the fix).

Every command embeds through the zero-config default described under
[Embedders](#embedders): a local Ollama (`all-minilm`, 384 dims), or whatever
`VAULT_EMBED_*` points at. Run `ollama pull all-minilm` once; with nothing
listening the command says so in one line — `no embedder configured: start
Ollama ...` — and exits 1, rather than quietly falling back to a cloud API.

`--lexical` swaps in the deterministic `TokenOverlapEmbedder` instead: no
daemon, no network, and no semantics either — for an offline machine, and what
the test suite runs on. The two are different vector spaces, so switching costs
a full re-embed: `reindex`, `doctor` and `watch` do it on their next pass, and
`search` refuses in the meantime (`... — run vault doctor to re-embed`) rather
than compare vectors that are not comparable.

```
$ vault search renewal terms
0.0328  customers/acme.md — Acme Corp
0.0161  customers/globex.md — Globex
```

One line per hit: fused RRF score, vault-relative path, title. The CLI sets no
relevance cutoffs — the ceiling that means "irrelevant" is a property of the
embedder, and it cannot know yours (under `--lexical` no fixed one is meaningful
at all) — so it shows the ranking and lets you judge it. Library callers pass
their own (`cutoffs`), and the write gate must.

`vault watch` reindexes once, then follows `fs.watch` (recursive) with a ~250ms
per-path debounce, re-embedding only notes whose content hash actually changed.
It logs the passes that changed something and stops on ctrl-c (finishing the
pass in flight first). It is a convenience, never a source of
truth: anything it misses — a directory rename, a pass that failed, a crash —
`vault doctor` finds and fixes.

```
$ vault consolidate --ceiling 0.15
indexed 0 new, 0 changed, 0 removed, 214 unchanged
0.0412  customers/acme.md customers/acme-corp.md
0.1180  cross-namespace  notes/pager.md support/rota.md
```

`vault consolidate` is report-only: one line per near-duplicate cluster —
widest distance inside it, then the note paths, with the clusters that span
namespaces flagged, because those are never merged. It reindexes first, so the
report describes the files rather than a stale index. Merging is a library
call — it needs a merger you inject, the same reason there is no
`vault propose` — so this is the pass you run to find the ceiling your embedder
calls a duplicate. See [Consolidation](#consolidation).

## Obsidian

Point Obsidian (or any editor) at the vault directory and work normally. Notes
are ordinary markdown with YAML frontmatter and `[[wikilink]]`s, one note per
file, subdirectories as namespaces.

Links resolve the way Obsidian resolves them, namespace-aware:

- `[[customers/acme]]` — a path-qualified link: the vault-relative path without
  `.md`. Always unambiguous, whatever else the vault holds. Prefer it; it is what
  the write gate writes for a note in a namespace (a note at the vault root has
  no qualified form — its path without `.md` *is* its stem — so keep notes you
  link to in namespaces);
- `[[acme]]` — a bare stem: resolves only while exactly one note in the vault is
  named `acme.md`. `customers/acme.md` and `vendors/acme.md` are two perfectly
  good notes, but a bare `[[acme]]` between them means nothing, so it stays
  unresolved rather than picking one. `vault doctor` lists it as ambiguous with
  both candidates, written as links, so the fix is a copy-paste:

```
$ vault doctor
ambiguous link: notes/deal.md -> [[acme]] (customers/acme, vendors/acme)
broken link:    notes/deal.md -> [[ghots]]
```

Obsidian hides dot-directories, so `.vault/` stays out of the way; the scan skips
it (and `.git/`, `.obsidian/`, …) for the same reason. If the vault is a git repo,
add `.vault/` to its `.gitignore` — the index is a build artifact, not content.
Run `vault watch` alongside your editing session to keep search current.

## Library

```ts
import { open, gatePrompt, parseDecision, TokenOverlapEmbedder } from "@wilcus/vault";

const vault = open("/path/to/vault", {
  embedder: new TokenOverlapEmbedder(),
  gate: {
    // your LLM call; `gatePrompt` and `parseDecision` are the wiring, not the model
    decider: async (input) => parseDecision(await askYourModel(gatePrompt(input))),
    // mandatory: without cutoffs "most similar" degrades into "least unrelated"
    cutoffs: { distanceCeiling: 0.35, bm25Ceiling: -1 },
  },
  // optional: per-agent namespace rules. Omitted, every caller may do anything.
  scopes: { "core/scheduler": [{ prefix: "", read: true, write: true }] },
});

const ctx = { agent: "core/scheduler", source: "task-42" }; // who is calling, per call

await vault.reindex();
await vault.search("acme renewal", { n: 5, cutoffs: { distanceCeiling: 0.35 }, ctx });
await vault.get("customers/acme.md", ctx);  // one note, parsed — read from the file
vault.list("customers", ctx);               // note paths under a namespace, sorted
await vault.propose(
  { title: "Acme renewal 2026", type: "customer", namespace: "customers", body },
  ctx, // optional — until a scope policy is in force, which needs it to decide
);
await vault.doctor();

const watcher = vault.watch();     // keep the index warm while a human edits
await watcher.close();             // resolves when the pass in flight is done
vault.close();                     // ...so this cannot close the DB under a write
```

### Reading notes

`get` is a note's identity — its vault-relative path, `.md` and all — turned
into the parsed note: frontmatter, body, title, wikilinks, hash. It reads the
**file**, so it is never stale, whatever the index thinks; it returns `null`
when nothing is there, and a directory or a symlink at the path counts as
nothing. A path that leaves the vault, or runs through `.vault/` or a symlinked
directory, throws — that is a caller bug, not a missing note.

Spell the path however your code built it — `./customers/acme.md`, a doubled
slash, an absolute path inside the vault — and `note.path` still comes back as
the one canonical identity, which is what makes it safe to store.

`list` answers the cheap question from the index: which notes exist. Paths only,
sorted, optionally under one namespace — and a namespace means whole segments,
so `list("ledger")` never sweeps in `ledger-archive/`. Superseded notes are
listed; `list` is the note set, not the search set.

```ts
const note = await vault.get("customers/acme.md"); // Note | null
note?.links;                                       // ["support-rota", ...]
vault.list();                                      // every note path, sorted
vault.list("customers/");                          // "customers" works too
```

Both take an optional trailing `VaultContext`, which decides what they answer
once the vault has a scope policy — see [Scopes](#scopes).

### The write gate

`propose` is the only way a program writes to the vault:

1. hybrid-search the candidate against the vault, with mandatory relevance
   cutoffs, and re-read each hit from disk;
2. ask your `decider` for one action — `update`, `supersede`, `create` or
   `discard` — over those notes;
3. apply it behind two rails. **Check-and-write:** a target is re-hashed
   immediately before it is touched, so a human edit mid-flight aborts the apply,
   re-runs the gate once, then falls back to `create`. **Path confinement:** every
   written path is `<namespace>/<slug>.md` with a single slugified segment,
   resolved under the vault root, never through a symlink or dot-directory.

Writes land through a temp file renamed into place, so a reader never sees half a
note. A discarded candidate — or one the gate cannot place at all — is appended
whole to `<root>/.discarded.log`; losing the note is never an outcome. That log is
history, not index, so it sits beside the notes rather than in the disposable
`.vault/` — a `doctor --rebuild` or an `rm -rf .vault` leaves it alone (a log left
in the old place is moved out by the next repairing `vault doctor`). It holds whole
candidate bodies, so a vault kept in git may want it in `.gitignore` too. Notes the
gate did not author are patched textually, never re-serialized, so comments,
`01234` and `1.0` survive. Human edits bypass the gate by definition:
`vault watch` and `vault doctor` pick them up.

```ts
const result = await vault.propose(candidate, { agent: "core/scheduler" });
// { action: "supersede", path: "customers/acme-renewal-2026.md",
//   superseded: "customers/acme.md", fellBack: false }
```

The second argument is a `VaultContext` — `{ agent, source? }`, the caller's
identity for that one call. Given, the gate stamps `vault_agent` (and
`vault_source`) into the frontmatter of every note it *authors*: a `create`, and
a `supersede`'s successor, get them serialized in; an `update` gets them patched
in beside its `updated` bump. Marking the superseded note is bookkeeping rather
than authorship, so its own provenance is left alone. Omit the context and
nothing is stamped — on an `update` that also means the previous call's keys are
*removed*, so `vault_agent` never names an agent that did not write the note it
sits on. Omitting it stops being an option once the vault has a scope policy,
which has nothing to check the write against without it.

### Consolidation

A vault accretes near-duplicates: the gate only sees the top-k similar notes at
write time, and humans add notes behind its back. `consolidate` is the
deliberate, occasional merge pass — manually triggered, never a daemon.

```ts
const vault = open("/path/to/vault", {
  embedder,
  consolidate: {
    // your LLM call again; `mergePrompt` and `parseMerged` are the wiring
    merger: async (input) => parseMerged(await askYourModel(mergePrompt(input))),
  },
});

// dry run: what a merge pass *would* do, and it is the default
const report = await vault.consolidate({ ceiling: 0.15 });
report.merges[0];       // { cluster: { members, namespace, distance }, candidate }
report.crossNamespace;  // clusters spanning namespaces — reported, never merged
report.remaining;       // clusters the cap did not reach

await vault.consolidate({ ceiling: 0.15, cap: 3, write: true, ctx });
// merges[0] → { ..., path: "customers/acme.md", superseded: [...], unmarked: [] }
```

- **The ceiling is mandatory**, like the gate's cutoffs and for the same
  reason: there is no universal number for "duplicate", it is a property of
  your embedder. Find yours with `vault consolidate --ceiling <d>` before you
  let anything write.
- A cluster admits a note only if it is under the ceiling from **every** note
  already in it (complete linkage). Single linkage chains A~B~C and merges A
  with a C it does not resemble. The scan is all-pairs over the vectors table,
  O(n²) and accepted — notes embed whole, so n is the note count.
- Clusters that **span namespaces** are reported and never merged: namespaces
  are boundaries, and collapsing one is a human call. Superseded notes never
  cluster, and neither do notes the embedder had no tokens for (a CJK or
  emoji-only note has no vector row — it stays findable through FTS).
- **Dry-run is the default**; `write: true` is what makes a run act. A wrong
  ceiling found in a report costs nothing. A dry run still reindexes first (so
  discovery sees the files, not a stale index) and still calls your merger once
  per cluster it would merge, up to the cap.
- A write goes through the gate's own rails: the merged note is created with
  slug confinement, collision suffixing and `ctx`'s provenance, then **each**
  member is marked `superseded_by` it — a textual patch, check-and-write per
  member, and a member a human edited mid-flight comes back in `unmarked`
  rather than being clobbered. Nothing is ever deleted: the originals stay on
  disk, marked, out of search.
- The **cap** (default 5) counts clusters merged; the rest come back in
  `remaining`. A pass that wants to rewrite half the vault is evidence the
  ceiling is wrong, and the cap turns that into a short report instead of a
  long mess.

Consolidation is an operator operation like `doctor` — unscoped, whole-vault —
so its `ctx` is provenance for the notes it writes, not a permission check.

### Scopes

Several agents usually share one vault. `scopes` says who may touch what —
namespace prefixes, per agent, for `read` and `write` independently:

```ts
const vault = open("/path/to/vault", {
  embedder,
  gate,
  scopes: {
    "core/scheduler": [{ prefix: "", read: true, write: true },
                       { prefix: "ledger/", write: false }], // reads it, cannot rewrite it
    "core/support":   [{ prefix: "support/", read: true, write: true },
                       { prefix: "customers/", read: true }], // read-only next door
  },
});

await vault.search("acme renewal", { ctx: { agent: "core/support" } });
await vault.get("ledger/q3.md", { agent: "core/support" }); // null: not readable
vault.list("customers", { agent: "core/support" });         // only what it may read
await vault.propose(candidate, { agent: "core/support" });
```

- **No `scopes` means allow-all**, so a single-agent caller changes nothing.
  With one, the vault fails closed: every call needs a `VaultContext`, an agent
  the policy does not name is refused with a throw rather than an empty result,
  and `{}` denies everyone.
- A prefix matches **whole segments** — `ledger/` never matches
  `ledger-archive/`; `""` is the root rule. For each of `read` and `write`
  separately, the longest matching prefix that *specifies* it wins; a rule that
  leaves one out defers to the next-shorter match, and nothing specifying it
  means denied. `open()` refuses a policy that answers one question twice, one
  with a subtree writable but not readable (an agent that cannot see its own
  notes re-creates them on every propose), a rule that is not
  `{prefix, read?, write?}` with booleans — a JSON `read: "false"` is truthy,
  and would grant where it meant to deny — and a prefix that is not a canonical
  path (`./ledger`, `ledger//sub`), which would match nothing and deny nothing.
- `search` filters unreadable notes out of the over-fetched set before capping,
  so you get *up to* N readable hits (and `expandLinks` neighbours are filtered
  too). `get` returns null for an unreadable note, exactly like an absent one.
  `propose` checks the candidate's namespace before your decider runs, shows the
  decider only notes the agent may read, marks the ones it may not write
  read-only in the prompt, and falls back to `create` if a decision targets one
  anyway — the candidate always lands somewhere.
- `doctor`, `reindex`, `watch` and `close` are operator operations on the whole
  vault and take no context.

Scopes are **advisory containment at the library API, not security**: any
process with filesystem access can read or edit the notes directly. That is the
files-are-truth contract, not a hole in it — the boundary that matters for
hostile code is the OS, not this policy object.

### Embedders

An `Embedder` is `{ model, dims, embed(texts) }` and is always injected — the
vault never hardcodes a provider. Two ship:

- `FetchEmbedder` — any OpenAI-compatible `POST /v1/embeddings`. Unconfigured it
  is a local Ollama (`all-minilm`, 384 dims, no API key), so the zero-config
  default keeps every note on your machine. What the CLI uses.
- `TokenOverlapEmbedder` — deterministic bag-of-tokens, no network. What the test
  suite and `vault --lexical` use: it exercises the plumbing, not semantics.

```ts
import { open, FetchEmbedder } from "@wilcus/vault";

// zero config: http://localhost:11434/v1/embeddings, all-minilm, 384 dims.
// Run `ollama pull all-minilm` first; if nothing is listening the embed fails
// with "no embedder configured: start Ollama ... or configure a remote
// provider" — it never quietly falls back to a cloud API.
const vault = open(root, { embedder: new FetchEmbedder() });
await vault.doctor(); // first run with a new model: re-embeds everything
```

A `VAULT_EMBED_API_KEY` sitting in the environment is *not* sent to that default
endpoint — it belongs to whichever remote provider you configured it for, and
"whatever is listening on :11434" does not get to collect it. Configure an
endpoint, or pass `apiKey` (a local gateway may want one), and it travels.

A remote provider is supported, but only as an explicit choice — whole note
bodies leave the machine on every embed. Each option falls back to its env var
(`VAULT_EMBED_ENDPOINT`, `VAULT_EMBED_MODEL`, `VAULT_EMBED_DIMS`,
`VAULT_EMBED_API_KEY`); a non-localhost endpoint requires the key, and it is
never persisted, logged, or echoed back in a provider's error message. A remote
endpoint must also name its `model` and `dims` — the defaults describe the local
model, not yours, and a wrong one would be filed as if it were right.

```ts
const vault = open(root, {
  embedder: new FetchEmbedder({
    endpoint: "https://api.openai.com/v1/embeddings",
    model: "text-embedding-3-small",
    dims: 1536, // apiKey from VAULT_EMBED_API_KEY
  }),
});
```

`dims` must match what the model returns — it is part of the vec0 table's schema.
Changing either the model or the dims invalidates every stored vector, so the next
`doctor` (or `reindex`) drops the vector table and re-embeds every note; search
refuses to mix vector spaces until it has. Notes are embedded whole; there is no
chunking.

## Development

```
bun run check     # bun test && tsc --noEmit — green before any PR
```

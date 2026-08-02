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
vault --help                             # every command and flag
```

`--vault` defaults to the current directory; `--` ends flag parsing, so a query
may start with a dash. Exit code 0 on success, 1 on error — and 1 from `doctor`
when it found links only a human can fix: broken (nothing to point at) or
ambiguous (a bare `[[stem]]` several notes answer to — `doctor` prints the
candidate paths, and qualifying the link with one of them is the fix).

```
$ vault search renewal terms
0.0328  customers/acme.md — Acme Corp
0.0161  customers/globex.md — Globex
```

One line per hit: fused RRF score, vault-relative path, title. The CLI sets no
relevance cutoffs — no fixed cosine ceiling is meaningful for the bag-of-tokens
embedder — so it shows the ranking and lets you judge it. Library callers with a
real embedder pass their own (`cutoffs`), and the write gate must.

`vault watch` reindexes once, then follows `fs.watch` (recursive) with a ~250ms
per-path debounce, re-embedding only notes whose content hash actually changed.
It logs the passes that changed something and stops on ctrl-c (finishing the
pass in flight first). It is a convenience, never a source of
truth: anything it misses — a directory rename, a pass that failed, a crash —
`vault doctor` finds and fixes.

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
});

await vault.reindex();
await vault.search("acme renewal", { n: 5, cutoffs: { distanceCeiling: 0.35 } });
await vault.get("customers/acme.md");   // one note, parsed — read from the file
vault.list("customers");                // every note path under a namespace, sorted
await vault.propose(
  { title: "Acme renewal 2026", type: "customer", namespace: "customers", body },
  { agent: "core/scheduler", source: "task-42" }, // optional: who is writing, per call
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
sits on.

### Embedders

An `Embedder` is `{ model, dims, embed(texts) }` and is always injected — the
vault never hardcodes a provider. Two ship:

- `TokenOverlapEmbedder` — deterministic bag-of-tokens, no network. What the CLI
  and the test suite use: it exercises the plumbing, not semantics.
- `FetchEmbedder` — any OpenAI-compatible `POST /v1/embeddings`. Unconfigured it
  is a local Ollama (`all-minilm`, 384 dims, no API key), so the zero-config
  default keeps every note on your machine.

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

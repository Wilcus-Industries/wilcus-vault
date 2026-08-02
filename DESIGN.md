# @wilcus/vault — design

Files-are-truth memory vault per the wilcus-agents SPEC (§ @wilcus/vault). Markdown
notes on disk are the source of truth; SQLite holds only derived, disposable index
data. Standalone: zero wilcus deps, MIT.

Verified in bootstrap research (test/hybrid-smoke.test.ts): sqlite-vec
0.1.7-alpha.2 (pinned exact — pre-1.0 alpha native extension; re-evaluate the pin
when 0.2/1.0 lands or if a KNN correctness bug appears) loads under `bun:sqlite`
on Linux via `sqliteVec.load(db)`; FTS5 and `FULL OUTER JOIN` are available in
Bun's bundled SQLite, so hybrid RRF fusion is one SQL statement (the smoke test
exercises rows unique to each side of the join); `Bun.YAML` parses frontmatter —
no YAML dependency.

## Layout

```
src/
  note.ts    # parse/serialize a note: frontmatter (Bun.YAML), wikilinks, sha256 hash,
             # textual frontmatter patch for files we did not author
  db.ts      # open DB (WAL, busy_timeout), load sqlite-vec, schema/migrations,
             # vectorsStale (read-only) / resetVectors (destructive)
  term.ts    # scrub control characters out of anything echoed to a terminal
  embed.ts   # Embedder interface + deterministic test embedder + fetch-based API embedder
  indexer.ts # scan vault dir, hash-diff, upsert notes/fts/vectors, rewrite edges
  doctor.ts  # drift report + repair + --rebuild into a temp DB, renamed into place
  search.ts  # hybrid: vec KNN + FTS5 BM25 → pre-fusion cutoffs → RRF ordering
  scope.ts   # ScopePolicy: validate/normalize at open(), one prefix resolver for
             # every enforcement point (and its SQL twin, for the search filters)
  gate.ts    # write gate: top-k similar → decider → update|supersede|create|discard
  watch.ts   # fs.watch + debounce + hash dirty-check → reindex changed files
  vault.ts   # Vault facade (public API), incl. the direct reads: get(path), list(prefix?)
  cli.ts     # vault doctor|reindex|search|watch — FetchEmbedder by default,
             # --lexical for the offline (TokenOverlap) one
```

`indexer.ts` has one write path, `indexPaths(paths)`: hash-diff those paths,
write what changed, purge the rows whose file is gone. `reindex` passes every
path the files *or* the index know about (so a row with no file is a deletion);
the watcher passes the handful that just changed. A watched vault and a rebuilt
one cannot drift apart, because only one function ever writes an index row.

What is at a path is decided by an `lstat`, before it is read (`noteEntry`, one
function so the indexer and `get` cannot drift): **only a regular file is a
note**, and a path that is gone, is a directory, or has become a symlink counts
as a deletion. The scan applies the same rule (it skips both), so
a path that survives only the read would be a row the scan never lists again —
`doctor` would report it missing forever while a repair happily re-read it,
indexing a symlink's target from *outside* the vault. A directory would be worse
than wrong: `EISDIR` out of `reindex` and `doctor` alike, with no way left to
repair the vault.

## Data model

- A note = one `.md` file under the vault root (subdirs = namespaces). The scan
  skips dot-directories (`.vault/`, `.git/`, `.obsidian/`, …) and does not follow
  symlinks.
- Identity = vault-relative path. **Link resolution is namespace-aware**
  (Obsidian-compatible), because `customers/acme.md` and `vendors/acme.md` are
  two legitimate notes, not a collision:
  - a **path-qualified** link, `[[customers/acme]]` — any target containing `/`
    — matches the note whose vault-relative path minus `.md` is exactly that.
    One note or none; never ambiguous. This is how an agent should link;
  - a **bare stem**, `[[acme]]`, resolves only when exactly one note in the
    vault carries that filename stem. Two or more candidates leave `to_id`
    null — the link is *ambiguous*, never resolved by first match, shortest
    path, or any other tiebreak that could silently mean the wrong note.
    `doctor` names the candidates so a human or agent can qualify it.

  A target is matched against indexed note paths in SQL and is **never joined
  onto the filesystem**, so `[[../../etc/passwd]]` is not a traversal — it is a
  string that matches no row. One consequence, recorded rather than discovered:
  a note at the vault root has no qualified form (its path minus `.md` *is* its
  stem), so two notes named `acme.md` at the root and in `customers/` make
  `[[acme]]` permanently ambiguous — move the root one into a namespace.
  A rename/move is a delete + create (identity is the path); `doctor` reports
  the resulting broken edges.
- Frontmatter: `type`, `created`, `updated`, optional `superseded_by`
  (**vault-relative path** of the superseding note), plus free keys. Written by
  us, editable by humans. `parseNote` never throws: a file whose frontmatter is
  unterminated, non-mapping, invalid YAML, or over the alias-expansion node
  budget (YAML aliases re-expand — a 250-byte billion-laughs note would
  otherwise become a megabytes-wide index row) still indexes with the whole file
  as its body and a `malformedFrontmatter` flag for `doctor` to report; so does
  a non-string `title`/`type`, whose value is ignored. Title = frontmatter
  `title` ?? first `# ` heading ?? filename stem.
- Wikilinks (`[[target]]`, `[[target|alias]]` — target only, deduped) and the fallback
  heading are found by regex over the body, not a markdown parse: links and
  headings inside code fences count. Deliberate MVP simplification — a spurious
  edge is visible in `doctor`, and no note is ever lost to a parse failure.
- DB at `<vault>/.vault/index.db`, opened in WAL mode with `busy_timeout=5000`
  (watcher, CLI, and library callers share it). Never committed to the vault's
  own git. Tables:
  - `notes(id, path unique, slug, title, type, hash, frontmatter, superseded_by,
    mtime, malformed)` — `slug` (the filename stem) and `malformed` are
    denormalized off the parsed note so link resolution and `doctor`'s report
    are plain SQL; both are derived, like every other column here.
  - `edges(from_id, to_slug, to_id nullable, unique(from_id, to_slug))` —
    `to_slug` is the link *as written*: a bare stem or a path (the column keeps
    its name so an index written by an older version still opens). Reindexing a
    note deletes its edges by `from_id` and reinserts. `to_id is null` ⇒ broken
    or ambiguous link; backlinks/orphans are trivial SQL. Resolution is one
    `update` over every edge — cheap, and correct when adding or removing a note
    flips links in notes that did not themselves change.
  - `vectors` vec0 virtual table (`note_id`, `emb float[dims] distance_metric=cosine`)
    + `vector_meta(note_id, model, dims)`. A change of model **or dims** ⇒ doctor
    drops and recreates the vec0 table (dims live in its DDL) and re-embeds all
    notes — deterministic, cheap at this scale.
  - `notes_fts` FTS5 (`title`, `body`).

## Retrieval

Embed query → vec0 KNN with over-fetch (`k = 3×N`, so post-filters can't starve
the result set) and FTS5 BM25 `LIMIT 3×N`. **Relevance cutoffs apply per signal,
before fusion** — cosine-distance ceiling on the KNN side, BM25 ceiling on the
FTS side — because RRF scores are ordinal (top hit always scores 1/61 no matter
how bad it is); a threshold on the fused score cannot filter irrelevance. RRF
(`score = Σ 1/(60+rank)`, FULL OUTER JOIN) then only orders the survivors; cap at
N. Superseded notes are filtered from the over-fetched set before ranking. When
nothing survives the cutoffs, the result is empty — callers (the write gate
especially) must treat that as "no similar notes exist", not an error.
One-hop wikilink expansion is an opt-in second pass, never an LLM graph walk:
neighbours (either direction) of the survivors are appended below every direct
hit, capped at N of their own, so `expandLinks` returns at most 2N.

The user's query never reaches FTS5 as syntax: each whitespace-separated run
becomes one quoted phrase (embedded quotes doubled), so `NEAR(`, `OR`, `*` and
`^` are matched as words, and the keyword side is capped at the first 32
distinct terms — a whole note body is a legitimate query (the write gate passes
one) but not a legitimate 400-term MATCH. If FTS5 rejects a query anyway, that
signal drops out and the search continues on vectors alone. The cutoffs are the
caller's policy — there is no default, because the ceiling that means
"irrelevant" is a property of the embedder, so the `vault search`
CLI sets none: it cannot know the ceiling for whichever provider is configured,
and under `--lexical` (bag-of-tokens) there is no meaningful fixed one to know.
Both are upper bounds on a
lower-is-better quantity: cosine distance, and FTS5's negative `rank`. They live
in one `cutoffs` option so a caller has to decide about them rather than inherit
silence.

Search is not the only read. Two direct paths sit beside it on the facade, for a
caller that already knows which note it wants (wilcus-core#42: one note parser
in the ecosystem, this one):

- **`get(path)`** takes a note's identity — the vault-relative path *including*
  `.md`; `ledger/q3` names nothing — and returns the parsed note or null. It
  reads the **file**, never the index row, so a stale, missing or half-written
  row cannot change the answer; that is what makes it safe for another package
  to delete its own parser. The argument is canonicalized first, into the same
  form the scan stores (`relative` + forward slashes): `./ledger/q3.md`,
  `ledger//q3.md`, a Windows-joined `ledger\q3.md` and an absolute path inside
  the vault are all the one note, and the `path` handed back is the identity a
  caller may store — it cannot vary with how the caller spelled it. Only a regular file is a note, so a directory or a
  symlink at the path is null exactly like an absent one. The path's *parent* is
  put through the same `confinedPath` rail the write gate uses, so an escape, a
  dot-directory or a symlinked directory on the way down **throws** — a caller
  that built such a path has a bug, and silence would hide it. The leaf needs no
  confinement of its own: a symlink there is already null, and a path cannot
  escape the root through its last segment alone.
- **`list(prefix?)`** returns vault-relative paths from the index rows, sorted.
  Derived data is legitimate here — paths are precisely what the scan rebuilds,
  and a caller that wants content calls `get`. An optional `prefix` names a
  namespace and matches on **segment boundaries**, the same rule as § Scopes and
  context: `ledger` and `ledger/` both match `ledger/q3.md`, neither matches
  `ledger-archive/q3.md`. It is the note set, not the search set — superseded
  notes are listed and no cutoff applies.

Both take a `VaultContext` as an optional trailing parameter, and under a
`ScopePolicy` it decides what they answer: `get` returns null for a note the
agent may not read (exactly like an absent one) and `list` is filtered to the
readable set. With no policy in force the parameter is inert — there is nothing
to enforce — but it is refused rather than ignored when one is (§ Scopes and
context).

## Embedding

`Embedder = { model: string; dims: number; embed(texts: string[]): Promise<Float32Array[]> }`
— injected. Vectors are L2-normalized on insert and on query (so cosine distance
is well-defined regardless of provider). Ships: `TokenOverlapEmbedder`
(deterministic bag-of-tokens, for tests/evals — exercises the plumbing, not
semantics) and `FetchEmbedder` (OpenAI-compatible `/v1/embeddings`; endpoint,
model, dims and key from the constructor falling back to `VAULT_EMBED_*` env
vars, never persisted to DB/frontmatter and never echoed in error messages —
a quoted provider error body has the key redacted out).

Unconfigured, `FetchEmbedder` is **local**: `http://localhost:11434/v1/embeddings`
with `all-minilm` at 384 dims — Ollama's OpenAI-compatible route, so the default
costs no dependency and no note ever leaves the machine. A cloud provider is
supported but never inherited: whole note bodies leave the machine on every
embed, so that is a choice a caller makes explicitly — and one it spells out,
since a **remote endpoint must name its `model` and `dims`** (option or env).
The defaults describe the local model; inheriting them would post note bodies
under a model name the provider never heard of and file the answer as if it
were that vector space.

Three rules follow from "the default endpoint is nobody's choice, it is just
whatever holds `:11434`":

- **Keys are not adopted by it.** A `VAULT_EMBED_API_KEY` in the environment was
  put there for someone's remote provider; the defaulted endpoint never sends
  it, so a local process cannot harvest a cloud key. Configure an endpoint (or
  pass `apiKey` — a local gateway may want one) and the key travels.
- **No key is required to reach localhost**; any other endpoint refuses to
  construct without one. A configured endpoint is validated as an http(s) URL
  with a host, and the error names the setting that holds the bad value.
- **"Start Ollama" is only said about that endpoint.** When nothing answers
  there the first request fails with "no embedder configured: start Ollama
  (`ollama pull all-minilm`) or configure a remote provider", the original
  failure attached as its `cause` — one attempt, no retry, and no fallback to a
  remote provider, which would ship note bodies off the machine to fix a daemon
  that is merely not running. An endpoint the caller chose (a vLLM on `:8000`)
  surfaces its own error instead, and a timeout means something *is* listening
  and is reported as itself.

The CLI is a caller like any other: every command builds that same defaulted
`FetchEmbedder`, and `--lexical` substitutes `TokenOverlapEmbedder` for a machine
with no daemon (and for the suite, so CI needs no Ollama). Either way a bad
configuration or an unreachable endpoint reaches the user as the one sentence it
was written as, and exit 1 — never a stack, and never raw: an error now quotes a
provider's response body, so it goes through `term.ts` like every other
untrusted string the vault prints (`safe`/`printable` — control characters
become `?`, so nothing can redraw the terminal's last line).

**A model swap does not destroy anything until its replacements exist.**
Staleness is *detected* read-only (`vectorsStale`) before `embed` is called, and
the drop-and-recreate (`resetVectors`) runs inside the write transaction that
files the new vectors. Embedding is a network call that fails for ordinary
reasons — the daemon is not running, `--lexical` and the default were swapped —
and the old order left the vault with an empty `vectors` table, an empty
`vector_meta` and nothing recording that a re-embed was owed: `search` would
then quietly answer on FTS alone, at exit 0. Now a failed swap rolls back whole,
and `search` keeps refusing stale vectors until a pass has actually replaced
them. (`doctor --rebuild` was always safe — it builds a temp DB and renames it
into place.)

Requests are batched by text count *and* by characters, since a
whole-note payload is what actually blows a provider's per-request limit. Notes
embedded whole — no chunking.

A note whose text yields no tokens the embedder recognizes (CJK, emoji or
punctuation only) embeds to all zeros. A zero vector has no direction, so
cosine distance against it is NaN and would poison KNN: the indexer writes no
`vectors` row for it — the note stays findable through FTS — while its
`vector_meta` row still records the attempt, so it is not mistaken for a
half-indexed note and re-embedded on every pass.

## Write gate

Every programmatic write goes through `vault.propose(candidate, ctx?)`, where
`ctx` is the caller's per-call identity (§ Scopes and context) — given, the gate
stamps its provenance onto every note it authors; absent, nothing is stamped
(and under a `ScopePolicy` the call is refused, since there is then nothing to
check the write against). Two refusals a doomed write earns **before** the
decider runs, so no model call is spent on it: a namespace that fails path
confinement, and one this agent may not write:

1. hybrid-search top-k similar notes, each one re-read from disk so the hash
   captured is the *file's*, not the index's — the index is derived data. The
   gate **must** pass `cutoffs`: without them the search always returns
   *something*, and "most similar note" becomes "least unrelated note" — the
   gate would update or supersede a stranger instead of creating a new note.
   `{}` is refused at runtime, not just discouraged — at least one ceiling must
   be set, or the mandate is only a type. Note bodies reaching the prompt are
   data, not instruction: any line that could pass for one of the prompt's
   delimiters is indented so a note cannot close its own fence;
2. `decider({candidate, similar})` → `{action: update|supersede|create|discard, target?}`
   — decider is an injected async fn (the caller wires an LLM; tests use fakes).
   A prompt template + strict response parser ship here;
3. apply, with two safety rails:
   - **Check-and-write:** before touching a target file, re-hash it; if it
     changed since step 1 (human edit mid-flight), abort the apply and re-run
     the gate once against fresh state; on a second mismatch, fall back to
     `create`. Nothing is ever clobbered silently. Applies to `update` rewrites
     and to `supersede`'s frontmatter edit of the old note — which is checked
     again immediately before that patch, since writing the successor widens
     the window: if the old note moved in it, the successor stands and the
     result reports it `unmarked` rather than overwriting a human's edit for
     bookkeeping. Every write goes to a temp file renamed over the target, so a
     reader never sees a half-written note and nothing can be swapped in
     underneath the path check.
   - **Path confinement:** created paths are `<namespace>/<slug>.md` where slug
     is a single slugified segment (`[a-z0-9-]+`, no dots, no separators) derived
     from the candidate title; the joined path is resolved and asserted to be
     under the vault root; symlinked targets are rejected. LLM-derived strings
     never name a raw filesystem path.
   - **No re-serialization of notes we didn't author:** a YAML round-trip is
     lossy against hand-written data (comments dropped, `01234` → `1234`,
     `1.0` → `1`). Frontmatter edits to existing notes (e.g. `supersede`'s
     `superseded_by`) are **textual patches of the frontmatter block** — append
     a line, or replace every occurrence of a key's line, so YAML last-wins
     cannot resurrect the old value — leaving every other line byte-identical.
     `serializeNote` is only for notes the gate authors from scratch. "Has a
     usable block" is one predicate shared with `parseNote`: a fenced block
     whose YAML the parser cannot read is *not* a block, and gets a fresh one
     prepended, or a `superseded_by` patched into it would be a line nothing
     ever reads.
   - `create` writes a new file; `update` rewrites the target body, bumps
     `updated` and sets *or clears* the provenance keys to match the call
     (textual patches, per the rule above — the patcher takes a `null` value as
     an unset for exactly this); `supersede`
     writes the new note, adds `superseded_by` (vault-relative path) to the old
     note's frontmatter plus a **path-qualified** forward wikilink
     (`[[customers/acme-2026]]`) — the gate knows the exact path, so a
     namespaced successor's link cannot go ambiguous behind a note that shares
     the stem later (a successor written to the vault root has no qualified
     form, so its link is a bare stem and still can);
     — and marking the old note does **not** restamp its provenance, since
     marking is bookkeeping, not authorship;
     `discard` appends the candidate
     as a JSONL line to `<root>/.discarded.log` so a wrong LLM call never silently
     loses information. That log is durable history, so it lives beside the notes
     and not in the disposable `.vault/` index directory — a `--rebuild` or an
     `rm -rf .vault` must not take it with them. It is a dot-file, so the scan
     never indexes it, and `doctor` moves a log left in the old location once.

Two consequences of the rails, recorded so they are not mistaken for slips. A
traversing *title* is slugified rather than refused (`../../evil` is the note
`evil`) — a title legitimately contains `/` and `.`, and the slug is one
`[a-z0-9-]+` segment by construction; a traversing, hidden or symlinked
**namespace**, and a decider `target` that was not one of the notes the search
returned, are refused outright. And a `create` whose slug is already taken — by
a file, or by another note's stem — suffixes (`acme-2`) instead of overwriting:
a shared title is not permission to lose someone else's note. Stems need not be
unique any more, but the gate keeps *its* notes' stems unique anyway, because
adding a second `acme.md` is exactly what turns a human's existing `[[acme]]`
ambiguous. A title that slugifies to nothing (CJK, Cyrillic, emoji)
is named `note-<8 hex of the candidate's hash>`; a candidate the gate cannot
place at all is appended to `<root>/.discarded.log` before it throws. Losing the
note is never one of the outcomes.

Human edits bypass the gate by definition (files are truth); the watcher +
`doctor` pick them up.

## Scopes and context

Multiple agents share one vault; the vault needs to know *who* is calling and
*what they may touch*. Two pieces, deliberately separate: **identity travels
per call, policy is fixed at `open()`** — one process holds one vault handle on
behalf of many agents, so baking the agent in at open time freezes exactly the
values that vary per call (wilcus-core#43 learned this the hard way).

**`VaultContext`** — per-call identity:

```ts
type VaultContext = { agent: string; source?: string };
```

`agent` names the caller (`core/scheduler`); `source` optionally records what
prompted the call — a conversation id, a task id, freeform. It is the second
parameter of `propose(candidate, ctx)`, and the read paths carry it as
`SearchOptions.ctx` and an optional trailing parameter on `get`/`list`. It is
optional exactly as long as no `ScopePolicy` is in force: with one, every call
that omits it is refused, because a policy keyed on an agent has nothing to
decide without one.
The gate stamps provenance into the frontmatter of every note it writes:
**`vault_agent`** and **`vault_source`** — namespaced, because `agent:` and
`source:` are exactly the keys a human's own frontmatter plausibly holds, and
the textual patcher replaces top-level lines (patching a human's nested
`source:` block would orphan its children into a parse error). Both are
single-line values, so `update`'s textual patch applies cleanly; `create` and
`supersede` serialize them fresh on the note they author. `vault_agent`
answers "which agent last wrote this note through the gate" — so on `update`
these keys are set to *exactly* this call's context: a key the call does not
supply is **removed**, not left standing. A stale `vault_source` beside a fresh
`vault_agent` would assert a pairing that never happened, and a leftover
`vault_agent` under a context-free write would name an agent that did not make
it. An `agent` that is empty or whitespace is refused outright. Marking the
*old* note `superseded_by` does not restamp its provenance — the marking is
bookkeeping, not authorship, and the superseding agent is already on the
successor. Provenance lives in the file, like every other truth here.

**`ScopePolicy`** — optional in `VaultOptions` (`scopes?`); **absent means
allow-all**, so existing single-agent callers change nothing. Present, it is
an allowlist, and it fails closed: an agent with no entry is refused with a
throw, not a silently empty result — silence is how an orchestrator typo
makes an agent re-create the memory it thinks it lost — and a call without a
`VaultContext` throws for the same reason. An empty policy `{}` therefore
denies everyone: `{}` and `undefined` sit on opposite sides of the
fail-open/fail-closed line, deliberately.

```ts
type ScopeRule = { prefix: string; read?: boolean; write?: boolean };
type ScopePolicy = Record<string, ScopeRule[]>; // agent name → rules
```

A `prefix` names a namespace subtree and matches on **segment boundaries
only**: every non-empty prefix is normalized to a trailing `/` at `open()`,
and `ledger/` matches `ledger/q3.md` but not `ledger-archive/q3.md` — raw
`startsWith` would grant across sibling namespaces, which is precisely what
an allowlist exists to stop. `""` is the root rule: it matches every note,
and a root-level note (no `/` in its path) matches only it. Resolution is
**per permission, longest prefix wins**: for each of `read` and `write`
independently, the longest matching prefix whose rule *specifies* that
permission decides; a rule that leaves one unspecified defers to the
next-shorter match; nothing specifies ⇒ denied. So
`[{prefix: "", read: true, write: true}, {prefix: "ledger/", write: false}]`
reads everywhere and writes everywhere except `ledger/`, which stays
readable. Two rules with the same normalized prefix specifying the same
permission contradict each other, and `open()` refuses the policy rather
than pick a winner; it likewise refuses a subtree writable but not readable —
a write-blind agent never sees its own notes as `similar`, so every propose
lands as `create`: a duplicate factory, not a scope.

`open()` also refuses what a policy's *type* cannot: it is operator
configuration, so it arrives from a file, an orchestrator, another process's
JSON, and a `read: "false"` there is **truthy** — a rule meant as a denial
would grant. Every rule is checked to be `{prefix: string, read?: boolean,
write?: boolean}`, and a prefix that is not the canonical form of a path
(`./ledger`, `ledger//sub`, `ledger/../x`) is refused rather than normalized:
stored paths are canonical, so such a prefix matches nothing, and a deny rule
that matches nothing is a deny that never fires.

Resolution itself lives in `scope.ts` — validated and normalized once at
`open()`, then one `may(permission, path)` every enforcement point calls, plus
the same rules compiled to a SQL `case` for the two filters that have to run
inside the query. Two spellings of one rule, held to the same answers by the
suite; the alternative was the prefix rule reimplemented in `vault.ts`,
`search.ts` and `gate.ts`.

Enforcement points, all inside the library so no caller re-implements them:

- `search` — the scope filter runs over the **over-fetched** set (alongside
  the supersede filter, before RRF caps at N), so a scoped agent gets **up
  to** N readable hits. Up to: the over-fetch is a fixed 3×N, so an agent
  scoped to a thin slice of the vault can exhaust it and see fewer — accepted,
  and the over-fetch factor is where the fix goes if it bites. The one-hop
  `expandLinks` pass is its own enforcement point: neighbour rows pass the
  same read filter before they are appended, or a scoped agent would read
  forbidden titles one wikilink away;
- `get` / `list` — the read check. An unreadable `get` returns null, exactly
  like an absent note: a scope is not an existence oracle. (`create`'s slug
  collision suffixing can still betray that *something* holds a stem —
  accepted: it leaks a stem's existence, never content.)
- `propose` — the write check, twice. The candidate's target namespace is
  checked *before* the decider runs (fail fast, no model spend on a doomed
  write) — and it is checked in the **canonical** form the file will actually
  be written at, resolved through the confinement rail once and used from
  there on. `notes/../ledger` is inside the vault and starts with `notes/`:
  checking the caller's spelling while writing the resolved one is a scope
  bypass, not a cosmetic difference. Only notes the agent may read feed the
  decider as `similar`: an agent must not have another agent's note bodies
  quoted back to it by the prompt. The SQL filter decides that, and the gate
  re-checks each hit in JS before reading its body off disk — that is where
  note bodies leave the vault for a prompt, so it does not rest on one
  filter. A note readable but not writable is marked read-only in that
  prompt — a hint to the model, never the enforcement, since a title is
  unfenced text and a decider can target a marked note anyway; the rail is
  the write check on the decision's `target`, and a decision that targets one
  **falls back to `create`**, like a target that failed check-and-write
  twice: the candidate always lands somewhere, losing it is never an outcome.

Maintenance is unscoped: `doctor`, `reindex`, `watch` and `close` are
operator operations on the whole vault and take no context — a scoped agent
is not the one running repairs.

Stated plainly: **scopes are advisory containment at the library API, not
security.** Any process with filesystem access can read or edit the files
directly; that is the files-are-truth contract, not a hole in it. The boundary
that matters for hostile code is the OS, not this policy object.

Prefix matching is **byte-exact**, and deliberately: on Linux `Secret/` and
`secret/` are two different namespaces holding two different notes, and
case-folding the comparison would deny an agent a namespace it was granted.
The consequence, recorded rather than discovered: on a case-insensitive
filesystem (macOS, Windows) a path spelled `Secret/plans.md` reaches the same
file a `secret/` rule denies, so the rule does not cover it. One more reason
the sentence above is the operative one — containment, not security.

## Consolidation pass (spec — no implementation yet)

Vaults accrete near-duplicates: the gate only sees top-k similar at write
time, and humans add notes behind its back. Consolidation is the deliberate,
occasional merge pass — **manually triggered** (`consolidate()` /
`vault consolidate`), never a daemon; a background process that rewrites
notes is exactly the surprise the write gate exists to prevent. It is an
operator operation like `doctor`: unscoped, and the provenance it stamps is
the `VaultContext` the operator hands it.

- **Discovery** is embedding distance: pairs of live (non-superseded) notes
  under a caller-set cosine-distance ceiling. The ceiling is mandatory, like
  the gate's cutoffs and for the same reason — there is no universal number
  for "duplicate", it is a property of the embedder. A cluster admits a note
  only if *every* pair inside it is under the ceiling (complete linkage) —
  single-linkage chains A~B~C into merging A with a C it does not resemble.
  All-pairs over the vectors table is O(n²) and accepted: notes embed whole,
  so n is the note count, thousands at most. Clusters that span namespaces
  are reported but never merged — namespaces are boundaries, and collapsing
  across one is a human call.
- **Merging reuses the gate's rails, not its decider.** The caller injects a
  merger (an LLM, like the decider) that turns a cluster — bodies re-read
  from disk, like everything the gate shows a model — into one merged
  candidate. The pass writes that candidate through the gate's create path
  (slug confinement, collision suffixing) into the cluster's namespace, then
  marks *each* member `superseded_by` the new note with the gate's existing
  supersede-marking rail — textual patch, check-and-write per member,
  `unmarked` reported for any member a human edited mid-flight. One cluster,
  one new note, N marked members; no decider run, because the merger already
  decided, and `propose`'s own search could neither see the cluster nor
  supersede more than one note. Nothing is ever deleted: originals stay on
  disk, marked, out of search.
- **Dry-run is the default.** A run reports clusters and each would-be merge;
  writing takes an explicit flag. A wrong ceiling discovered in a report
  costs nothing; discovered in the files, it costs an afternoon.
- **Per-run action cap**, counted in clusters merged (default single digits):
  on hitting it the run stops and reports the remainder. A pass that wants to
  rewrite half the vault is evidence the ceiling is wrong, and the cap turns
  that evidence into a short report instead of a long mess.

## Doctor / watcher

`vault doctor` — report + repair: rebuild stale index rows (hash mismatch), remove
rows for deleted files, drop-and-re-embed on embedding model/dims change, list
broken links, ambiguous links, orphans, and malformed frontmatter. Broken and
ambiguous are different problems and are reported apart: **broken** is 0
candidates (a typo, or a note that is gone), **ambiguous** is 2+ and carries
`candidates: string[]` — link *targets*, not filenames (`customers/acme`, no
`.md`), so the report is the fix and not merely the complaint: paste one into
the note as `[[customers/acme]]`. (A candidate at the vault root has no
qualified form and reads as the ambiguous stem itself; that note has to move
into a namespace.) A duplicate filename stem is
*not* itself reported: two namespaces holding an `acme.md` is the point of
namespaces, and only a bare link to them is a problem.
`--rebuild` reindexes from scratch into a temp DB file, then atomically renames it
over `index.db` (safe against a concurrently running watcher). Doctor also carries
the one migration the vault has: a discard log still sitting in `.vault/` is
appended to `<root>/.discarded.log` and removed, once, before anything else
touches `.vault/`, and the report says so. It runs only on a **repairing** run
(the default, or `--rebuild`); `repair: false` is a report, and a report does
not move files.

`vault watch` — `fs.watch` (recursive) on the root, acting only on `.md` paths
outside dot-directories, so the index's own writes under `.vault/` cannot feed
the watcher its own tail. Debounce is **per path** (~250ms): an editor writing
one file continuously delays that file, never every other change queued behind
it; paths that come due together are indexed in one pass, and a pass in flight
makes later ones queue rather than run concurrently against the same database.
Each pass is `indexPaths`, so the hash check — not the event — decides whether a
note is reindexed and re-embedded, and a delete is just a path whose file is
gone. The CLI reindexes once before watching, so edits made while nothing was
watching are not missed.

Three failure modes, all of which land in the same place. A transient error (a
provider blip, a locked database, an `fs.watch` error event) is logged and the
watcher keeps going — it never throws at the caller mid-run, and never takes the
process down, including when the caller's own `onError` throws. A directory
rename reports only the directory, so notes moved inside it are missed. A
`close()` drops paths still inside their debounce window and anything queued
behind the pass in flight. None of these is data loss: the files are the truth,
and `doctor` rebuilds every row from them.

`close()` returns the pass in flight, because that pass still holds the database
handle: `await watcher.close()` before closing the database is what keeps a
shutdown from racing a write. Queued paths are dropped rather than drained —
nobody is waiting for them, and doctor knows where they are.

## Testing / evals

`bun test` runs everything; done-check: `bun run check` (= `bun test && tsc
--noEmit`). No CI is configured — the check is enforced by the agent workflow
(every PR runs it before merge). Deterministic-first evals (spec §8): retrieval
(exact identifier hits via FTS, overlap paraphrase via vector path, fusion beats
either alone on a seeded vault), write-gate behaviors per action incl. the
mid-flight-edit abort, doctor idempotence (delete DB → rebuild → same results),
watcher re-embed on edit, and a model swap end to end (reopen with another
embedder → search refuses stale vectors → doctor re-embeds → search works).
LLM-judge evals only where a rubric is unavoidable — none needed for MVP.

Real filesystem events are timed by the OS, so the watcher is tested at two
levels: its event core driven directly (`watcher.touch(path)` is the same entry
point `fs.watch` calls, and `idle()` resolves when the queue has drained), plus
one end-to-end pass that edits, creates and deletes real files under a real
`fs.watch` and waits for the index rows to catch up.

## Build slices (issues)

1. Note model: parse/serialize frontmatter + wikilinks + hash.
2. DB + indexer + doctor core: schema, scan/upsert, edges, broken/orphan queries.
3. Hybrid search: embedder interface, vec+FTS+RRF, pre-fusion cutoffs, supersede
   filtering, over-fetch.
4. Write gate: decider contract, apply paths incl. check-and-write + path
   confinement, supersede chain, discard log.
5. Watcher, embedding-model/dims-swap re-embed, CLI polish, README.

Post-MVP (#18): 6. this design — scopes + consolidation spec (#19);
7. provenance + `propose(candidate, ctx)` + discard log to
`<root>/.discarded.log` (#20) — the log is durable history and must survive a
`.vault/` nuke or `--rebuild`; it carries full candidate bodies, so an
operator who commits their vault may want it in `.gitignore`; 8. `get`/`list`
(#21); 9. CLI real embedder (#22); 10. scope enforcement (#23).

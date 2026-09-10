# wilcus-vault — design

Files-are-truth memory vault per the wilcus-agents SPEC (§ @wilcus/vault). Markdown
notes on disk are the source of truth; SQLite holds only derived, disposable index
data. Standalone: zero wilcus deps, MIT.

Verified in bootstrap research (`tests/test_smoke.py`): sqlite-vec
0.1.7a2 (pinned exact — pre-1.0 alpha native extension; re-evaluate the pin
when 0.2/1.0 lands or if a KNN correctness bug appears) loads under the stdlib
`sqlite3` via `enable_load_extension`; FTS5 and `FULL OUTER JOIN` need the
system SQLite ≥ 3.39, and with them hybrid RRF fusion is one SQL statement (the
smoke test exercises rows unique to each side of the join). Frontmatter is
parsed by PyYAML under a SafeLoader that leaves dates and timestamps as strings;
hashes are `hashlib` sha256; provider HTTP is `urllib` on a worker thread; the
watcher rides on `watchfiles`.

Python 3.12+, stdlib first. Runtime dependencies are exactly three, each with
its reason recorded here, and nothing joins them without one:

- `sqlite-vec` — the KNN extension; the stdlib has no vector index.
- `PyYAML` — frontmatter; the stdlib has no YAML parser.
- `watchfiles` — recursive filesystem watching with a debounce; `asyncio` has
  none, and polling the vault is not it.

## Layout

```
src/wilcus_vault/
  __init__.py       # the public surface; everything a caller imports lives here
  vault.py          # Vault facade (public API), incl. the direct reads: get(path), list(prefix?)
  note.py           # parse/serialize a note: frontmatter (PyYAML, dates as strings),
                    # wikilinks, sha256 hash; parsing never raises
  frontmatter.py    # textual frontmatter patch / body replace / link qualify for
                    # files we did not author
  db.py             # open DB (WAL, busy_timeout), load sqlite-vec, schema,
                    # vectors_stale (read-only) / reset_vectors (destructive)
  term.py           # scrub control characters out of anything echoed to a terminal
  paths.py          # confined_path, slugify, write_atomic: every filename rail
  embed.py          # Embedder protocol + the deterministic TokenOverlapEmbedder
  fetch_embedder.py # FetchEmbedder over an OpenAI-compatible /v1/embeddings
  http.py           # the endpoint/key rules, redaction and POST shared by both providers
  indexer.py        # scan vault dir, hash-diff, one write path (index_paths)
  index_rows.py     # every index row a note owns: write, purge, resolve edges
  qualify.py        # auto-qualify bare wikilinks when a pass creates a stem collision
  doctor.py         # drift report + repair + --rebuild in place, in the live index
  search.py         # hybrid: query vector + FTS terms → pre-fusion cutoffs → RRF ordering
  search_sql.py     # the fused-ranking and link-expansion statements, SearchHit, Cutoffs
  scope.py          # ScopePolicy: validate/normalize at open(), one prefix resolver for
                    # every enforcement point (and its SQL twin, for the search filters)
  decision.py       # the decider contract: Candidate, Decision, gate_prompt, parse_decision
  decide.py         # fetch_decider: an OpenAI-compatible chat decider for the CLI
  gate.py           # write gate: top-k similar → decider → update|supersede|create|discard
  gate_write.py     # the notes the gate authors: create (free path) and mark_superseded
  discard_log.py    # the discard log's write side: JSONL beside the notes, rotated
  discards.py       # its read side: list / get / restore, and doctor's count
  cluster.py        # complete-linkage clusters under a distance ceiling
  merge.py          # the merger contract: MergedNote, merge_prompt, parse_merged
  consolidate.py    # the consolidation pass over those clusters, dry-run by default
  watch.py          # watchfiles + per-path debounce + hash dirty-check → index_paths
  cli/
    __init__.py     # vault reindex|doctor|search|watch|consolidate|discards — arg parsing
    commands.py     # the subcommands that need more than a line
    usage.py        # help text and report formatting
tests/
  conftest.py       # make_vault / write_note / vec_of, VAULT_* env cleared per test
  fakes.py          # stub embedder, recording transport, fixed decider
  test_lines.py     # every source module under 200 lines, or listed with a reason
  test_smoke.py     # the bootstrap research above, kept runnable
  test_*.py         # one file (or a few) per module
scripts/check.sh    # the done-check: ruff check, ruff format --check, mypy strict, pytest
pyproject.toml      # uv project; runtime deps pinned as above
```

`indexer.py` has one write path, `index_paths(paths)`: hash-diff those paths,
write what changed, purge the rows whose file is gone. `reindex` passes every
path the files *or* the index know about (so a row with no file is a deletion);
the watcher passes the handful that just changed. A watched vault and a rebuilt
one cannot drift apart, because only one function ever writes an index row.

What is at a path is decided by an `lstat`, before it is read (`note_entry`, one
function so the indexer and `get` cannot drift): **only a regular file is a
note**, and a path that is gone, is a directory, or has become a symlink counts
as a deletion. The scan applies the same rule (it skips both), so
a path that survives only the read would be a row the scan never lists again —
`doctor` would report it missing forever while a repair happily re-read it,
indexing a symlink's target from *outside* the vault. A directory would be worse
than wrong: `IsADirectoryError` out of `reindex` and `doctor` alike, with no way left to
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
- **Auto-qualify on a new stem collision** (#29). At the moment an index pass
  *creates* a collision — a newly indexed note's stem matches exactly one note
  the index already knew about (a rename is not a collision: the old row is
  gone in the same pass) — every bare `[[stem]]` link in the vault still
  unambiguously means that incumbent, and only this moment can know it: once
  both are indexed, nothing records which was first. `index_paths` rewrites
  those links to the incumbent's path-qualified form, mechanically, no LLM —
  detection happens before the write transaction, no file I/O happens inside
  it (a rollback cannot unwrite a file), and the rewrites land after commit as
  textual body edits (`qualify_links`) under the gate's check-and-write rails,
  capped (default 500 per collision), then re-entered through `index_paths`
  (bounded: the rewritten notes are not new, so no further detection). Between
  the commit and that re-entry those edges briefly read `to_id = null` — an
  accepted window. The outcome rides on `IndexStats.qualified`; the CLI prints
  it and the watcher logs it. The invariant is **never guesses**, not *never
  ambiguous*: a root incumbent (no qualified form), an incumbent whose path
  cannot survive a wikilink round trip (`[`, `]` or `|` in a directory name
  would destroy the links it rewrites), an incumbent whose *file* is gone (the
  row alone is not truth — a move the watcher sees in two passes is a rename,
  not a collision), both notes new in the same pass (no incumbent — picking
  one would be the forbidden first-match resolution), a linker edited
  mid-flight (hash mismatch, never clobbered), a linker unreadable or
  unwritable, and the cap's remainder are all reported and left to `doctor`'s
  ambiguous report — `rewritten` + `skipped` account for every linker. A
  failure in the post-rewrite re-entry rides on `IndexStats.index_error`
  instead of raising (the files have already changed; the next pass recovers
  the rows). `doctor --rebuild` and any cold first index see every note as
  new, so they are structurally no-ops here.
- Frontmatter: `type`, `created`, `updated`, optional `superseded_by`
  (**vault-relative path** of the superseding note), plus free keys. Written by
  us, editable by humans. `parse_note` never raises: a file whose frontmatter is
  unterminated, non-mapping, invalid YAML, or over the alias-expansion node
  budget (YAML aliases re-expand — a 250-byte billion-laughs note would
  otherwise become a megabytes-wide index row) still indexes with the whole file
  as its body and a `malformed_frontmatter` flag for `doctor` to report; so does
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

Embed query → vec0 KNN with over-fetch (`k = 3×N`) and FTS5 BM25 `LIMIT 3×N`.
That cut is taken *before* the supersede and scope filters run, so it can be
eaten entirely by rows those filters then drop. When fewer than N hits survive
**and** a signal filled its cut with rows that passed its own cutoff, the query
is asked again over the entire index, which cannot starve because nothing is
left outside it. Both conditions matter: the width needed to reach past a crowd
is a property of the vault rather than a constant, but a *short* cut after the
cutoff means the query is exhausted, not crowded — each ceiling bounds the very
quantity its signal is ordered by, so a row rejected inside the cut has no
better twin outside it. Without that second test the write gate, which must set
a cutoff and whose candidates are usually novel, would widen on every write to
re-derive the same empty answer. The wide pass is a full scan (~6ms at 1k notes,
~28ms at 5k) and only runs when both conditions hold. One limit survives, on the
KNN side alone: vec0 refuses `k` above 4096 (every `k` is clamped to it, since
3×N crosses it at N=1366), so on a larger vault a thinly scoped agent can still
be crowded out of the *vector* signal — by the whole index rather than by 3×N
of it. The FTS side has no such ceiling and always widens to the whole index. **Relevance cutoffs apply per signal,
before fusion** — cosine-distance ceiling on the KNN side, BM25 ceiling on the
FTS side — because RRF scores are ordinal (top hit always scores 1/61 no matter
how bad it is); a threshold on the fused score cannot filter irrelevance. RRF
(`score = Σ 1/(60+rank)`, FULL OUTER JOIN) then only orders the survivors; cap at
N. Superseded notes are filtered from the over-fetched set before ranking. When
nothing survives the cutoffs, the result is empty — callers (the write gate
especially) must treat that as "no similar notes exist", not an error.
One-hop wikilink expansion is an opt-in second pass, never an LLM graph walk:
neighbours (either direction) of the survivors are appended below every direct
hit, capped at N of their own, so `expand_links` returns at most 2N.

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
  `.md`; `ledger/q3` names nothing — and returns the parsed note or None. It
  reads the **file**, never the index row, so a stale, missing or half-written
  row cannot change the answer; that is what makes it safe for another package
  to delete its own parser. The argument is canonicalized first, into the same
  form the scan stores (`relative` + forward slashes): `./ledger/q3.md`,
  `ledger//q3.md`, a Windows-joined `ledger\q3.md` and an absolute path inside
  the vault are all the one note, and the `path` handed back is the identity a
  caller may store — it cannot vary with how the caller spelled it. Only a regular file is a note, so a directory or a
  symlink at the path is None exactly like an absent one. The path's *parent* is
  put through the same `confined_path` rail the write gate uses, so an escape, a
  dot-directory or a symlinked directory on the way down **raises** — a caller
  that built such a path has a bug, and silence would hide it. The leaf needs no
  confinement of its own: a symlink there is already None, and a path cannot
  escape the root through its last segment alone.
- **`list(prefix?)`** returns vault-relative paths from the index rows, sorted.
  Derived data is legitimate here — paths are precisely what the scan rebuilds,
  and a caller that wants content calls `get`. An optional `prefix` names a
  namespace and matches on **segment boundaries**, the same rule as § Scopes and
  context: `ledger` and `ledger/` both match `ledger/q3.md`, neither matches
  `ledger-archive/q3.md`. It is the note set, not the search set — superseded
  notes are listed and no cutoff applies.

Both take a `VaultContext` as an optional trailing parameter, and under a
`ScopePolicy` it decides what they answer: `get` returns None for a note the
agent may not read (exactly like an absent one) and `list` is filtered to the
readable set. With no policy in force the parameter is inert — there is nothing
to enforce — but it is refused rather than ignored when one is (§ Scopes and
context).

## Embedding

`Embedder` is a protocol — `model: str`, `dims: int`, `async embed(texts:
list[str]) -> list[list[float]]` — injected. Vectors are L2-normalized on insert and on query (so cosine distance
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
  pass `api_key` — a local gateway may want one) and the key travels.
- **No key is required to reach localhost**; any other endpoint refuses to
  construct without one. A configured endpoint is validated as an http(s) URL
  with a host, and the error names the setting that holds the bad value.
- **"Start Ollama" is only said about that endpoint.** When nothing answers
  there the first request fails with "no embedder configured: start Ollama
  (`ollama pull all-minilm`) or configure a remote provider", the original
  failure attached as its `__cause__` — one attempt, no retry, and no fallback to a
  remote provider, which would ship note bodies off the machine to fix a daemon
  that is merely not running. An endpoint the caller chose (a vLLM on `:8000`)
  surfaces its own error instead, and a timeout means something *is* listening
  and is reported as itself.

The CLI is a caller like any other: every command builds that same defaulted
`FetchEmbedder`, and `--lexical` substitutes `TokenOverlapEmbedder` for a machine
with no daemon (and for the suite, so CI needs no Ollama). Either way a bad
configuration or an unreachable endpoint reaches the user as the one sentence it
was written as, and exit 1 — never a stack, and never raw: an error now quotes a
provider's response body, so it goes through `term.py` like every other
untrusted string the vault prints (`safe`/`printable` — control characters
become `?`, so nothing can redraw the terminal's last line).

**A model swap does not destroy anything until its replacements exist.**
Staleness is *detected* read-only (`vectors_stale`) before `embed` is called, and
the drop-and-recreate (`reset_vectors`) runs inside the write transaction that
files the new vectors. Embedding is a network call that fails for ordinary
reasons — the daemon is not running, `--lexical` and the default were swapped —
and the old order left the vault with an empty `vectors` table, an empty
`vector_meta` and nothing recording that a re-embed was owed: `search` would
then quietly answer on FTS alone, at exit 0. Now a failed swap rolls back whole,
and `search` keeps refusing stale vectors until a pass has actually replaced
them. (`doctor --rebuild` is safe for the same reason — it forces every note dirty and
lands in that same transaction, so a failure never leaves an empty table claiming
to be current.)

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
   `Cutoffs()` is refused at runtime, not just discouraged — at least one ceiling
   must be set, or the mandate is only a type. Note bodies reaching the prompt are
   data, not instruction: any line that could pass for one of the prompt's
   delimiters is indented so a note cannot close its own fence;
2. `decider(DeciderInput(candidate, similar))` → `Decision(action=update|supersede|create|discard, target=...)`
   — decider is an injected `async def` (the caller wires an LLM; tests use fakes).
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
     `serialize_note` is only for notes the gate authors from scratch. "Has a
     usable block" is one predicate shared with `parse_note`: a fenced block
     whose YAML the parser cannot read is *not* a block, and gets a fresh one
     prepended, or a `superseded_by` patched into it would be a line nothing
     ever reads.
   - `create` writes a new file; `update` rewrites the target body, bumps
     `updated` and sets *or clears* the provenance keys to match the call
     (textual patches, per the rule above — the patcher takes a `None` value as
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
     loses information — together with the `similar` set the decider saw
     (`[{path, hash, score}]`, `[]` when none), which is the justification a
     later staleness pass (#34) reads and cannot be retrofitted. That log is
     durable history, so it lives beside the notes
     and not in the disposable `.vault/` index directory — a `--rebuild` or an
     `rm -rf .vault` must not take it with them. It is a dot-file, so the scan
     never indexes it, and `doctor` moves a log left in the old location once.
     Bounded but never deleted (#33): at 5 MiB it rotates to `.discarded.N.log`,
     and the first write to a fresh log adds `.discarded.log*` to the vault's
     `.gitignore` — once; a line the user removed stays removed — so refused
     note bodies never ride into git history. The read side is `discards.py`
     (`vault discards list | show <n> | restore <n>`, and a count in `doctor`'s
     report): `restore` feeds the candidate back through `propose`, so
     re-admission re-runs search+decide against *current* vault state — the
     gate stays the only write door. The CLI's `restore` wires `fetch_decider`
     (`decide.py`), FetchEmbedder's chat twin: OpenAI-compatible, configured by
     `VAULT_DECIDE_*`, endpoint defaulting to the local Ollama, model always
     explicit — the one CLI command that runs a model, because restoring
     without re-deciding would bypass the gate.

**The closing pass, and the freshness window.** A `propose` ends by re-indexing,
so the index never lags a write we made ourselves. That pass is deliberately the
*whole* vault and not just the paths written: what it buys is the next call's
search seeing a note a human edited behind our back, and without it the gate
re-creates notes that already exist — files are truth, and a human editing one
is the normal case, not a corner. It is also the expensive part of a `propose`
(measured at ~86% of one, since dirtiness is decided by content hash and every
note is therefore re-read), and a caller that writes in bursts pays for the
whole vault once per note.

`GateOptions.freshness` is the seconds that view may be stale. Zero, the
default, walks on every write exactly as before; a positive window lets the
writes inside it share one walk (measured ~4x on a burst of ten over 5k notes).
Only edits made *outside* the vault API go unseen for the window — a note the
gate wrote itself is indexed before `propose` returns whatever the setting — so
the window trades a bounded blindness to outside edits for the cost of finding
them. The clock is per handle and lives on `Vault`, not in `GateOptions`: it is
a property of this handle's view of the files, not of the gate's policy.

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
place at all is appended to `<root>/.discarded.log` before it raises. Losing the
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

```python
@dataclass(frozen=True)
class VaultContext:
    agent: str
    source: str | None = None
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

**`ScopePolicy`** — the optional `scopes=` of `open()`; **absent means
allow-all**, so existing single-agent callers change nothing. Present, it is
an allowlist, and it fails closed: an agent with no entry is refused with a
`VaultError`, not a silently empty result — silence is how an orchestrator typo
makes an agent re-create the memory it thinks it lost — and a call without a
`VaultContext` raises for the same reason. An empty policy `{}` therefore
denies everyone: `{}` and `None` sit on opposite sides of the
fail-open/fail-closed line, deliberately.

```python
class ScopeRule(TypedDict):  # plain dicts: {"prefix": "ledger/", "write": False}
    prefix: str
    read: NotRequired[bool]
    write: NotRequired[bool]


ScopePolicy = Mapping[str, list[ScopeRule]]  # agent name → rules
```

A `prefix` names a namespace subtree and matches on **segment boundaries
only**: every non-empty prefix is normalized to a trailing `/` at `open()`,
and `ledger/` matches `ledger/q3.md` but not `ledger-archive/q3.md` — raw
`startswith` would grant across sibling namespaces, which is precisely what
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
would grant. Every rule is checked to be `{prefix: str, read?: bool,
write?: bool}`, and a prefix that is not the canonical form of a path
(`./ledger`, `ledger//sub`, `ledger/../x`) is refused rather than normalized:
stored paths are canonical, so such a prefix matches nothing, and a deny rule
that matches nothing is a deny that never fires.

Resolution itself lives in `scope.py` — validated and normalized once at
`open()`, then one `may(permission, path)` every enforcement point calls, plus
the same rules compiled to a SQL `case` for the two filters that have to run
inside the query. Two spellings of one rule, held to the same answers by the
suite; the alternative was the prefix rule reimplemented in `vault.py`,
`search.py` and `gate.py`.

Enforcement points, all inside the library so no caller re-implements them:

- `search` — the scope filter runs over the **over-fetched** set (alongside
  the supersede filter, before RRF caps at N), so a scoped agent gets **up
  to** N readable hits, and "up to" is the cap and not a hedge: a scoped agent
  is not thinned by the notes it may not read. It would be under a fixed
  over-fetch — a crowd of unreadable notes fills the cut and the agent gets
  **zero** hits, reading them as "nothing similar exists" — and no constant
  factor fixes that, since the width required is `(crowd+1)/N` and grows with
  the vault. § Retrieval covers how the cut is widened instead. The one-hop
  `expand_links` pass is its own enforcement point: neighbour rows pass the
  same read filter before they are appended, or a scoped agent would read
  forbidden titles one wikilink away;
- `get` / `list` — the read check. An unreadable `get` returns None, exactly
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
  re-checks each hit in Python before reading its body off disk — that is where
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

### One shared memory, and how to carve exceptions in it

**The vault is one memory, not one memory per agent.** A fact the clerk learns
is *the* fact: one note, which any agent may later refine in place. That is the
default and it needs no configuration — `open()` without a `scopes=` policy
already gives every agent the whole vault, and the concurrency rails make
shared writing safe (every update is check-and-write against the file's hash,
every create claims its filename with a link that fails rather than
overwrites).

Per-agent partitioning is what a policy is *not* for. If every agent could only
write under `agents/<self>/`, a fact the clerk learned would be visible to all
but owned by the clerk: another agent refining it gets the cross-namespace write
refused, falls back to an in-namespace `create`, and the vault holds two
drifting copies of one fact. Namespaces are for *topics* — `customers/`,
`ledger/` — not for authorship. Authorship is already recorded, in the
`vault_agent` frontmatter the gate stamps on every note it writes.

What a policy *is* for is the narrow exception: a subtree some agent must not
write. The shape to reach for, and the one read and write were resolved
separately to allow:

```python
{
    "clerk": [
        {"prefix": "", "read": True, "write": True},  # the shared memory
        {"prefix": "ledger", "write": False},  # ...except the numbers
    ]
}
```

**Reads should stay wide.** Narrowing them buys nothing defensive — scopes are
advisory containment, and anything with filesystem access reads the notes
anyway. What it costs is real: the gate decides against the similar notes the
search returns, so an agent that cannot *see* a fact proposes a second copy of
it. Narrowing reads to sharpen retrieval is solving a ranking problem with a
permission, and it belongs in ranking.

**The decider is the rail, and that is accepted.** One shared memory means the
decider chooses among every agent's notes on every propose, and nothing but its
judgement keeps one agent's proposal off a note another agent depends on. A
per-call restriction on which note a single `propose` may target would not help:
the target is *deliberately* shared, so confining a call to its caller's
namespace defeats the model rather than protects it — and `ScopePolicy` could
not express it anyway, being agent-keyed and fixed at `open()`. A caller that
asked to update a specific note checks the returned `GateResult.path` is the one
it asked for. That is the whole guarantee, and it is enough: a wrong landing is
a bad edit to a versioned text file, not a loss.

### Not built: per-agent memory

Recorded so it is not re-derived, and deliberately **not designed here**. The
system above is memory of the *world*, and its mechanism suits that: an agent
proposes a fact, a decider decides where it lands, and it may be merged into an
existing note or discarded outright. An agent's memory of *itself* — a
scratchpad, notes worth reloading every session, how to drive a particular tool,
a workflow it has settled on — wants the opposite mechanism: the agent names the
path, the note lands there, no decider judges it and nothing discards it. Those
are two systems that would share one file tree, not one system with two
namespaces.

The gap in the current surface is exactly one thing: `propose` is the only write
path. Direct reads already exist (`get`, `list`). Whether that second system
gets built, whether its notes are embedded at all (a scratchpad rewritten thirty
times a session is thirty embeddings, and agent chatter dilutes the brain's
retrieval), and whether the gate may target the agent's own subtree, are all
open. One idea worth keeping if it is: when the same fact turns up in two
agents' own memories, that is evidence it is not about either of them — the
consolidation pass promoting it into the shared memory is the natural home for
that rule.

## Consolidation pass

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
- **A write run survives a bad cluster.** Once a merge has landed, a later
  cluster's failure (merger error, no free filename) must not discard the
  report of what landed: on a write run, per-cluster errors are collected into
  the report's `errors` field (scrubbed via `printable`; when the merged note
  was created before the raise, the entry carries its `path` — the file is
  live in search and nothing else names it) and the run continues, and the
  closing reindex runs regardless so the index never lags the landed writes —
  and when the reindex itself fails, that rides on the report's `index_error`
  instead of discarding it. A dry run still raises — nothing has landed that a
  report would need to account for. One failure stays loud even on a write
  run: a member path failing confinement is a tampered index, not a cluster
  error, and aborts the run.
- **Per-run action cap**, counted in clusters whose merger *ran* — merged or
  errored after the call, both spent their model call (default single digits);
  an error before the call (an unreadable member) burns no slot, or broken
  clusters would starve real ones run after run. On hitting it the run stops
  and reports the remainder. A pass that wants to rewrite half the vault is
  evidence the ceiling is wrong, and the cap turns that evidence into a short
  report instead of a long mess.

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
`--rebuild` rewrites every row from the files, in the live `index.db`. It
deliberately does *not* build a temp file and rename it over: the rename is
atomic, but it replaces the inode, and any process that already had the index
open goes on writing rows into a file nobody will open again. Nor does it clear
the old rows first — that would publish an empty index for the whole embedding
window, and a `propose` landing in that window judges against an empty vault and
writes a duplicate note to disk, which no later `doctor` can undo. It passes
`force` to `index_paths` instead: the hash check is skipped so every note is
dirty, and the one write transaction that already swaps the vectors replaces
every row and purges what the files no longer have. A failed rebuild rolls back
whole, and no reader ever sees a half-rebuilt index. Doctor also carries
the one migration the vault has: a discard log still sitting in `.vault/` is
appended to `<root>/.discarded.log` and removed, once, before anything else
touches `.vault/`, and the report says so. It runs only on a **repairing** run
(the default, or `--rebuild`); `repair: false` is a report, and a report does
not move files.

**A directory the scan cannot read.** `os.walk` reports an unreadable directory
as an empty one, so notes under it are invisible with nothing raised. That is the
mirror of the index side, where EACCES on an already-indexed note raises rather
than reading as a deletion — and the two are answered differently on purpose.
Purging known-good rows because we could not look is destructive, so it raises;
declining to add rows we never had is only a gap in visibility, and raising there
would make one unreadable directory anywhere in the tree fail every `reindex`
**and** `doctor --rebuild`, leaving the vault no way back. So the scan reports
instead: `scan_vault` returns those directories alongside the paths,
`IndexStats.unreadable` and `DoctorReport.unreadable` carry them, and both the
pass summary and the doctor report name them — an operator is told the vault was
only partly seen rather than reading the counts as the whole of it. This covers
directories, which `os.walk` hides; an unreadable *file* still raises out of
`read_raw`, and so out of `reindex` and `doctor`, because there the failure is
visible and reporting it would mean deciding whether an unreadable note counts as
one — a question with no answer the index can act on.

`vault watch` — `watchfiles` (recursive) on the root, acting only on `.md` paths
outside dot-directories, so the index's own writes under `.vault/` cannot feed
the watcher its own tail. Debounce is **per path** (~250ms): an editor writing
one file continuously delays that file, never every other change queued behind
it; paths that come due together are indexed in one pass, and a pass in flight
makes later ones queue rather than run concurrently against the same database.
Each pass is `index_paths`, so the hash check — not the event — decides whether a
note is reindexed and re-embedded, and a delete is just a path whose file is
gone. The CLI reindexes once before watching, so edits made while nothing was
watching are not missed.

Three failure modes, all of which land in the same place. A transient error (a
provider blip, a locked database, a `watchfiles` error) is logged and the
watcher keeps going — it never raises at the caller mid-run, and never takes the
process down, including when the caller's own `on_error` raises. A directory
rename reports only the directory, so notes moved inside it are missed. A
`close()` drops paths still inside their debounce window and anything queued
behind the pass in flight. None of these is data loss: the files are the truth,
and `doctor` rebuilds every row from them.

`close()` is synchronous and returns an awaitable for the pass in flight,
because that pass still holds the database handle: `await watcher.close()` before closing the database is what keeps a
shutdown from racing a write. Queued paths are dropped rather than drained —
nobody is waiting for them, and doctor knows where they are.

## Concurrency

The intended deployment is one writing process — a library caller holding one
handle, or `vault watch`. What happens when that is not true was left unsaid,
which is worse than a documented limitation: a caller cannot obey a rule nobody
wrote down. So the boundary is drawn here, and the parts that can hold without a
lock do.

**Safe against any number of writers.** Every write to an existing note is
check-and-write against the file's content hash, re-read at the moment of
writing: the gate's update path, `mark_superseded` and `qualify` all refuse and
report rather than overwrite a file that changed under them, which is the same
mechanism that protects a human editing in Obsidian. `write_atomic` renames a
finished temp file over the target, so no reader ever sees half a note. A note
the gate authors claims its filename with `os.link`, which fails rather than
overwrites — of two writers racing for `acme.md`, one gets it and the other
moves to `acme-2.md` still holding its content. Discard-log entries are one
`os.write` to a file opened `O_APPEND`, which POSIX makes atomic. A buffered
writer would not do: an entry holds a whole candidate body, so it easily passes
the 8KiB buffer, and an entry split across writes interleaves with another
writer's and ruins both lines.

**Safe because SQLite is.** The index is WAL with a 5s `busy_timeout`, so
readers never block writers and separate processes share one index file. Write
transactions are `begin immediate`: a deferred transaction takes the write lock
only at its first write, and SQLite refuses *that* upgrade outright rather than
waiting, so the lock is taken at the top where the timeout applies.

**Safe by not moving the file.** `doctor --rebuild` used to rename a fresh index
over `index.db`, which stranded any handle another process already held: it went
on writing rows to the replaced inode, into a file nobody would open again. The
rebuild now rewrites every row in the live database, inside the one transaction
it was already taking, so there is no second inode to strand a handle on and no
advisory lock for a watcher and a library handle to agree about. Coordination is
avoided rather than implemented, which is the cheaper answer when it is available.

**Not corruption, just waste.** Two reindex passes racing do redundant work and
converge: every write is a per-path upsert, and the hash decides. One edge is not
quite convergence — a pass decides which rows are `gone` *before* the embed await
and purges after it, so a note another process creates in that window has its
fresh row purged and stays unindexed until the next watcher event or `doctor`.
The file is untouched, which is why this is waste and not loss.

## Testing / evals

`uv run pytest` runs everything; done-check: `./scripts/check.sh` (= `ruff
check`, `ruff format --check`, `mypy` strict over `src` and `tests`, `pytest`).
Fixture vaults live under `tmp-test/` in the repo (pytest's basetemp), and every
test starts with the `VAULT_*` environment cleared, so a developer's own key or
endpoint can never decide one. `tests/test_lines.py` holds every source module
under 200 lines unless it is listed there with a reason. CI (`.github/workflows/check.yml`) runs the check on every PR and
push to main; the agent workflow also runs it before merge. Deterministic-first evals (spec §8): retrieval
(exact identifier hits via FTS, overlap paraphrase via vector path, fusion beats
either alone on a seeded vault), write-gate behaviors per action incl. the
mid-flight-edit abort, doctor idempotence (delete DB → rebuild → same results),
watcher re-embed on edit, and a model swap end to end (reopen with another
embedder → search refuses stale vectors → doctor re-embeds → search works).
LLM-judge evals only where a rubric is unavoidable — none needed for MVP.

Real filesystem events are timed by the OS, so the watcher is tested at two
levels: its event core driven directly (`watcher.touch(path)` is the same entry
point `watchfiles` calls, and `idle()` resolves when the queue has drained), plus
one end-to-end pass that edits, creates and deletes real files under a real
`watchfiles` and waits for the index rows to catch up.

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
(#21); 9. CLI real embedder (#22); 10. scope enforcement (#23);
11. the consolidation pass (#31) — `vault consolidate` reports clusters, the
library API is the write path (a merger is injected, like a decider).

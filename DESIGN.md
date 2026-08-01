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
  note.ts    # parse/serialize a note: frontmatter (Bun.YAML), wikilinks, sha256 hash
  db.ts      # open DB (WAL, busy_timeout), load sqlite-vec, schema/migrations
  embed.ts   # Embedder interface + deterministic test embedder + fetch-based API embedder
  indexer.ts # scan vault dir, hash-diff, upsert notes/fts/vectors, rewrite edges
  search.ts  # hybrid: vec KNN + FTS5 BM25 → pre-fusion cutoffs → RRF ordering
  gate.ts    # write gate: top-k similar → decider → update|supersede|create|discard
  watch.ts   # fs.watch + debounce + hash dirty-check → reindex changed files
  vault.ts   # Vault facade (public API)
  cli.ts     # vault doctor|reindex|search|watch
```

## Data model

- A note = one `.md` file under the vault root (subdirs = namespaces). The scan
  skips dot-directories (`.vault/`, `.git/`, `.obsidian/`, …) and does not follow
  symlinks.
- Identity = vault-relative path; wikilink slug = filename stem. Filename stems
  must be vault-wide unique: `[[acme]]` resolves only when exactly one stem
  matches; zero or multiple matches leave `to_id` null and `doctor` reports the
  broken/ambiguous link. A rename/move is a delete + create (identity is the
  path); `doctor` reports the resulting broken edges.
- Frontmatter: `type`, `created`, `updated`, optional `superseded_by`
  (**vault-relative path** of the superseding note), plus free keys. Written by
  us, editable by humans. `parseNote` never throws: a file whose frontmatter is
  unterminated, non-mapping, or invalid YAML still indexes with the whole file as
  its body and a `malformedFrontmatter` flag for `doctor` to report. Title =
  frontmatter `title` ?? first `# ` heading ?? filename stem.
- Wikilinks (`[[slug]]`, `[[slug|alias]]` — slug only, deduped) and the fallback
  heading are found by regex over the body, not a markdown parse: links and
  headings inside code fences count. Deliberate MVP simplification — a spurious
  edge is visible in `doctor`, and no note is ever lost to a parse failure.
- DB at `<vault>/.vault/index.db`, opened in WAL mode with `busy_timeout=5000`
  (watcher, CLI, and library callers share it). Never committed to the vault's
  own git. Tables:
  - `notes(id, path unique, title, type, hash, frontmatter, superseded_by, mtime)`
  - `edges(from_id, to_slug, to_id nullable, unique(from_id, to_slug))` —
    reindexing a note deletes its edges by `from_id` and reinserts. `to_id is
    null` ⇒ broken or ambiguous link; backlinks/orphans are trivial SQL.
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
One-hop wikilink expansion is an opt-in second pass, never an LLM graph walk.

## Embedding

`Embedder = { model: string; dims: number; embed(texts: string[]): Promise<Float32Array[]> }`
— injected. Vectors are L2-normalized on insert and on query (so cosine distance
is well-defined regardless of provider). Ships: `TokenOverlapEmbedder`
(deterministic bag-of-tokens, for tests/evals — exercises the plumbing, not
semantics) and `FetchEmbedder` (OpenAI-compatible `/v1/embeddings`; API key read
from env only, never persisted to DB/frontmatter and never echoed in error
messages; note that whole note bodies leave the machine on every embed — callers
choose the provider accordingly). Notes embedded whole — no chunking.

## Write gate

Every programmatic write goes through `vault.propose(candidate)`:

1. hybrid-search top-k similar notes, capturing each hit's content hash;
2. `decider({candidate, similar})` → `{action: update|supersede|create|discard, target?}`
   — decider is an injected async fn (the caller wires an LLM; tests use fakes).
   A prompt template + strict response parser ship here;
3. apply, with two safety rails:
   - **Check-and-write:** before touching a target file, re-hash it; if it
     changed since step 1 (human edit mid-flight), abort the apply and re-run
     the gate once against fresh state; on a second mismatch, fall back to
     `create`. Nothing is ever clobbered silently. Applies to `update` rewrites
     and to `supersede`'s frontmatter edit of the old note.
   - **Path confinement:** created paths are `<namespace>/<slug>.md` where slug
     is a single slugified segment (`[a-z0-9-]+`, no dots, no separators) derived
     from the candidate title; the joined path is resolved and asserted to be
     under the vault root; symlinked targets are rejected. LLM-derived strings
     never name a raw filesystem path.
   - `create` writes a new file; `update` rewrites the target body; `supersede`
     writes the new note, adds `superseded_by` (vault-relative path) to the old
     note's frontmatter plus a forward wikilink; `discard` appends the candidate
     as a JSONL line to `.vault/discarded.log` so a wrong LLM call never silently
     loses information.

Human edits bypass the gate by definition (files are truth); the watcher +
`doctor` pick them up.

## Doctor / watcher

`vault doctor` — report + repair: rebuild stale index rows (hash mismatch), remove
rows for deleted files, drop-and-re-embed on embedding model/dims change, list
broken/ambiguous links, orphans, and duplicate stems. `--rebuild` reindexes from
scratch into a temp DB file, then atomically renames it over `index.db` (safe
against a concurrently running watcher). `vault watch` — `fs.watch` (recursive) +
debounce; hash check decides re-embed. Watcher failure is never data loss: doctor
rebuilds everything from files.

## Testing / evals

`bun test` runs everything; done-check: `bun run check` (= `bun test && tsc
--noEmit`). No CI is configured — the check is enforced by the agent workflow
(every PR runs it before merge). Deterministic-first evals (spec §8): retrieval
(exact identifier hits via FTS, overlap paraphrase via vector path, fusion beats
either alone on a seeded vault), write-gate behaviors per action incl. the
mid-flight-edit abort, doctor idempotence (delete DB → rebuild → same results),
watcher re-embed on edit. LLM-judge evals only where a rubric is unavoidable —
none needed for MVP.

## Build slices (issues)

1. Note model: parse/serialize frontmatter + wikilinks + hash.
2. DB + indexer + doctor core: schema, scan/upsert, edges, broken/orphan queries.
3. Hybrid search: embedder interface, vec+FTS+RRF, pre-fusion cutoffs, supersede
   filtering, over-fetch.
4. Write gate: decider contract, apply paths incl. check-and-write + path
   confinement, supersede chain, discard log.
5. Watcher, embedding-model/dims-swap re-embed, CLI polish, README.

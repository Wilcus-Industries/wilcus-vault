# @wilcus/vault — design

Files-are-truth memory vault per the wilcus-agents SPEC (§ @wilcus/vault). Markdown
notes on disk are the source of truth; SQLite holds only derived, disposable index
data. Standalone: zero wilcus deps, MIT.

Verified in bootstrap research (test/hybrid-smoke.test.ts): sqlite-vec
0.1.7-alpha.2 (pinned — pre-1.0) loads under `bun:sqlite` on Linux via
`sqliteVec.load(db)`; FTS5 and `FULL OUTER JOIN` are available in Bun's bundled
SQLite, so hybrid RRF fusion is one SQL statement; `Bun.YAML` parses frontmatter —
no YAML dependency.

## Layout

```
src/
  note.ts    # parse/serialize a note: frontmatter (Bun.YAML), wikilinks, sha256 hash
  db.ts      # open DB, load sqlite-vec, schema/migrations
  embed.ts   # Embedder interface + deterministic test embedder + fetch-based API embedder
  indexer.ts # scan vault dir, hash-diff, upsert notes/edges/fts/vectors; full rebuild
  search.ts  # hybrid: vec KNN + FTS5 BM25 → RRF; optional one-hop wikilink expansion
  gate.ts    # write gate: top-k similar → decider → update|supersede|create|discard
  watch.ts   # fs.watch + debounce + hash dirty-check → reindex changed files
  vault.ts   # Vault facade (public API)
  cli.ts     # vault doctor|reindex|search|watch
```

## Data model

- A note = one `.md` file anywhere under the vault root (subdirs = namespaces).
  Identity = path; wikilink slug = filename stem.
- Frontmatter: `type`, `created`, `updated`, optional `superseded_by`, plus free
  keys. Written by us, editable by humans.
- DB at `<vault>/.vault/index.db` (never committed to the vault's own git):
  - `notes(id, path unique, title, type, hash, frontmatter, superseded_by, mtime)`
  - `edges(from_id, to_slug, to_id nullable)` — `to_id is null` ⇒ broken link;
    backlinks/orphans are trivial SQL.
  - `vectors` vec0 virtual table (`note_id`, `emb float[dims]`) +
    `vector_meta(note_id, model)` — model id per vector; model swap ⇒ deterministic
    full re-embed.
  - `notes_fts` FTS5 (`title`, `body`).

## Retrieval

Embed query → vec0 KNN top-N; FTS5 BM25 top-N (query sanitized for FTS syntax);
fuse via RRF (`score = Σ 1/(60+rank)`); apply score threshold + result cap.
Superseded notes are excluded by default. One-hop wikilink expansion is an opt-in
second pass, never an LLM graph walk.

## Embedding

`Embedder = { model: string; dims: number; embed(texts: string[]): Promise<Float32Array[]> }`
— injected. Ships: `TokenOverlapEmbedder` (deterministic bag-of-tokens, for tests/
evals — exercises the plumbing, not semantics) and `FetchEmbedder` (OpenAI-compatible
`/v1/embeddings` endpoint, env-configured). Notes embedded whole — no chunking.

## Write gate

Every programmatic write goes through `vault.propose(candidate)`:

1. hybrid-search top-k similar notes;
2. `decider({candidate, similar})` → `{action: update|supersede|create|discard, target?}`
   — decider is an injected async fn (the caller wires an LLM; tests use fakes).
   A prompt template + strict response parser ship here;
3. apply: `create` writes a new file; `update` rewrites target body;
   `supersede` writes the new note, adds `superseded_by` to the old note's
   frontmatter and a forward wikilink — nothing is deleted or overwritten silently.

Human edits bypass the gate by definition (files are truth); the watcher +
`doctor` pick them up.

## Doctor / watcher

`vault doctor` — report + repair: rebuild stale index rows (hash mismatch), remove
rows for deleted files, re-embed on embedding-model change, list broken links and
orphans. `--rebuild` drops the DB and reindexes from scratch. `vault watch` —
`fs.watch` (recursive) + debounce; hash check decides re-embed. Watcher failure
is never data loss: doctor rebuilds everything from files.

## Testing / evals

`bun test` runs everything; done-check: `bun test && bunx tsc --noEmit`.
Deterministic-first evals (spec §8): retrieval (exact identifier hits via FTS,
overlap paraphrase via vector path, fusion beats either alone on a seeded vault),
write-gate behaviors per action, doctor idempotence (delete DB → rebuild → same
results), watcher re-embed on edit. LLM-judge evals only where a rubric is
unavoidable — none needed for MVP.

## Build slices (issues)

1. Note model: parse/serialize frontmatter + wikilinks + hash.
2. DB + indexer + doctor core: schema, scan/upsert, edges, broken/orphan queries.
3. Hybrid search: embedder interface, vec+FTS+RRF, thresholds, supersede filtering.
4. Write gate: decider contract, apply paths, supersede chain.
5. Watcher, embedding-model-swap re-embed, CLI polish, README.

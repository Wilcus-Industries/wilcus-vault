# @wilcus/vault

Files-are-truth markdown memory vault. Read DESIGN.md before changing anything —
it is the architecture contract. Spec source: wilcus-agents SPEC § @wilcus/vault.

## Done-check

```
bun run check
```

Both green before any PR. TDD: failing test first.

## Rules

- Bun only: `bun:sqlite`, `Bun.YAML`, `Bun.CryptoHasher` — no Node-ecosystem
  substitutes for what Bun ships.
- Runtime deps: `sqlite-vec` (pinned exact — pre-1.0) and nothing else without a
  design reason recorded in DESIGN.md.
- Files are truth: no code path may treat the DB as authoritative; every index
  row must be rebuildable from the `.md` files.
- No chunking; notes embed whole. Numbers/ledgers never live in prose notes.
- Deciders and embedders are injected interfaces — never hardcode a provider.

## Workflow

- One issue per PR; branch `issue-<N>`; worktree under `.claude/worktrees/issue-<N>`.
- Every PR gets a review-agent pass before merge.

# wilcus-vault

Files-are-truth markdown memory vault. Read DESIGN.md before changing anything —
it is the architecture contract. Spec source: wilcus-agents SPEC § @wilcus/vault.

## Done-check

```
./scripts/check.sh
```

ruff check, ruff format --check, mypy (strict), pytest. All green before any PR.
TDD: failing test first.

## Rules

- Python 3.12+, stdlib first: `sqlite3`, `hashlib`, `urllib`, `asyncio` — no
  third-party substitute for what the stdlib ships.
- Runtime deps: `sqlite-vec` (pinned exact — pre-1.0), `PyYAML`, `watchfiles`, and
  nothing else without a design reason recorded in DESIGN.md.
- Files are truth: no code path may treat the DB as authoritative; every index
  row must be rebuildable from the `.md` files.
- No chunking; notes embed whole. Numbers/ledgers never live in prose notes.
- Deciders and embedders are injected interfaces — never hardcode a provider.
- Source modules stay under 200 lines (`tests/test_lines.py` enforces it).
- Comments explain the code, not the design doc.

## Workflow

- One issue per PR; branch `issue-<N>`; worktree under `.claude/worktrees/issue-<N>`.
- Every PR gets a review-agent pass before merge.

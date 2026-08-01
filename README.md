# @wilcus/vault

Files-are-truth markdown memory vault for AI agents: atomic notes, wikilink graph,
hybrid semantic + keyword search (sqlite-vec + FTS5 + reciprocal rank fusion), and a
write gate so nothing is silently overwritten. Open the vault in Obsidian or any
editor — the SQLite index is derived and disposable (`vault doctor --rebuild`).

Status: MVP under construction. See DESIGN.md.

```
bun run src/cli.ts reindex [--vault <dir>]           # index new and changed notes
bun run src/cli.ts doctor [--rebuild] [--vault <dir>] # check and repair the index
```

The index lives at `<vault>/.vault/index.db` and is safe to delete — `doctor
--rebuild` recreates it from the files.

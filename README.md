# @wilcus/vault

Files-are-truth markdown memory vault for AI agents: atomic notes, wikilink graph,
hybrid semantic + keyword search (sqlite-vec + FTS5 + reciprocal rank fusion), and a
write gate so nothing is silently overwritten. Open the vault in Obsidian or any
editor — the SQLite index is derived and disposable (`vault doctor --rebuild`).

Status: MVP under construction. See DESIGN.md.

```
bun run src/cli.ts reindex [--vault <dir>]            # index new and changed notes
bun run src/cli.ts doctor [--rebuild] [--vault <dir>] # check and repair the index
bun run src/cli.ts search <query> [--vault <dir>]     # hybrid search over the index
```

The CLI embeds locally with the deterministic bag-of-tokens embedder, so nothing
leaves the machine. A library caller can inject `FetchEmbedder` instead
(OpenAI-compatible `/v1/embeddings`, configured from `VAULT_EMBED_API_KEY`,
`VAULT_EMBED_ENDPOINT`, `VAULT_EMBED_MODEL` and `VAULT_EMBED_DIMS`) — note that
whole note bodies then leave the machine on every embed.

The index lives at `<vault>/.vault/index.db` and is safe to delete — `doctor
--rebuild` recreates it from the files.

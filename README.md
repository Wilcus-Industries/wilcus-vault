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
await vault.search("acme renewal");
await vault.propose({ title: "Acme renewal 2026", type: "customer", namespace: "customers", body });
await vault.doctor();
vault.close();
```

`propose` is the write gate: it hybrid-searches for similar notes, asks the
decider to `update`, `supersede`, `create` or `discard`, and applies that with
two rails — it re-hashes a target immediately before writing (a human edit
mid-flight aborts the apply, re-runs the gate once, then falls back to `create`)
and confines every path it writes to the vault root. Writes land through a temp
file renamed into place. A discarded candidate — or one the gate cannot place —
is appended whole to `.vault/discarded.log`. Notes it did not author are patched
textually, never re-serialized, so comments and `01234` survive.

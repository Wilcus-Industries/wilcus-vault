# @wilcus/vault

Files-are-truth markdown memory vault for AI agents: atomic notes, wikilink graph,
hybrid semantic + keyword search (sqlite-vec + FTS5 + reciprocal rank fusion), and a
write gate so nothing is silently overwritten. Open the vault in Obsidian or any
editor — the SQLite index is derived and disposable (`vault doctor --rebuild`).

Status: MVP under construction. See DESIGN.md.

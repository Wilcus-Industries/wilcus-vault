"""The CLI's help text and report formatting."""

from ..doctor import DoctorReport
from ..indexer import IndexStats
from ..term import safe

USAGE = """vault <command> [options]

  reindex             index new and changed notes
  doctor [--rebuild]  check and repair the index (--rebuild: from scratch)
  search <query>      hybrid search: one line per hit — score, path, title
  watch               index every change as it is saved, until interrupted
  consolidate         report near-duplicate clusters (needs --ceiling)
  discards list       the discard log, newest first — one line per entry
  discards show <n>   one entry in full, as JSON
  discards restore <n>  re-propose entry n through the write gate

  --vault <dir>       vault root (default: the current directory)
  --lexical           embed offline, without a provider (see below)
  --ceiling <d>       cosine distance two notes must be within to cluster
  --help, -h          this text
  --                  end of flags, so a search query may start with a dash

vault consolidate is report-only: one line per cluster — widest internal
distance, then the member paths, with the ones that span namespaces flagged
(those are never merged). Merging itself needs an injected merger, so it
lives in the library API; this is the pass that tells you which ceiling your
embedder calls a duplicate.

vault discards reviews <root>/.discarded.log — every candidate the write
gate refused, kept whole. Entries are numbered newest-first across the
size-rotated .discarded.N.log files, so a rotation can renumber between a
list and a restore: show <n> before restoring. restore feeds the candidate
back through the gate against the current vault, which needs --ceiling and
a chat model: set VAULT_DECIDE_MODEL (and VAULT_DECIDE_ENDPOINT /
VAULT_DECIDE_API_KEY for a provider that is not the local Ollama). The
log's first write adds .discarded.log* to the vault's .gitignore, once,
so refused note bodies never ride into git history.

Notes embed on a local Ollama unless VAULT_EMBED_* names another
OpenAI-compatible provider: run `ollama pull all-minilm` once, or pass
--lexical for a deterministic bag-of-tokens embedder that needs nothing
running — no network, and no semantics either. They are different vector
spaces: switching costs a full re-embed, which reindex, doctor and watch do
on their next pass and search refuses to go without.

Exit code 0 on success, 1 on error — and 1 from doctor when it found links
only a human can fix: broken (nothing to point at) or ambiguous (a bare
[[stem]] several notes answer to — qualify it as [[folder/stem]])."""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def summary(s: IndexStats) -> str:
    out = f"{s.added} new, {s.updated} changed, {s.removed} removed, {s.unchanged} unchanged"
    if s.reembedded:
        out += " (re-embedded: model or dims changed)"
    for q in s.qualified:
        # The stem and target are filenames, not our text, so they are scrubbed.
        if q.target is None:
            out += safe(
                f"; [[{q.stem}]] went ambiguous (no qualified form — "
                f"{_plural(len(q.skipped), 'linking note')} left for doctor)"
            )
        else:
            out += safe(
                f"; [[{q.stem}]] qualified to [[{q.target}]] in {_plural(len(q.rewritten), 'note')}"
            )
            if q.skipped:
                out += f" ({len(q.skipped)} skipped — see doctor)"
    if s.unreadable:
        # Their notes are invisible this pass, so the counts above undercount the vault.
        out += safe(f"; could not read: {', '.join(s.unreadable)}")
    if s.index_error is not None:
        out += safe(f"; re-index of rewritten notes failed: {s.index_error}")
    return out


def print_report(r: DoctorReport) -> None:
    lines = [
        f"reindexed {len(r.stale)} stale, purged {len(r.missing)} deleted"
        + (", re-embedded all notes (model or dims changed)" if r.reembedded else "")
    ]
    if r.migrated_discard_log:
        lines.append("moved .vault/discarded.log -> .discarded.log")
    if r.discards["entries"] > 0:
        lines.append(
            f"discard log: {r.discards['entries']} entries ({r.discards['recent']} recent)"
        )
    lines += [f"broken link:    {b.from_path} -> [[{b.slug}]]" for b in r.broken_links]
    # The candidates are the fix: qualify the link with one of these paths.
    lines += [
        f"ambiguous link: {a.from_path} -> [[{a.slug}]] ({', '.join(a.candidates)})"
        for a in r.ambiguous_links
    ]
    lines += [f"malformed frontmatter: {p}" for p in r.malformed]
    lines += [f"orphan: {p}" for p in r.orphans]
    lines += [f"unreadable directory: {p} (its notes are invisible)" for p in r.unreadable]
    print("\n".join(safe(line) for line in lines))

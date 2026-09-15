"""The CLI's help text and report formatting."""

from ..doctor import DoctorReport
from ..gate_write import GateResult
from ..indexer import IndexStats
from ..term import safe

USAGE = """vault <command> [options]

  reindex             index new and changed notes
  doctor [--rebuild]  check and repair the index (--rebuild: from scratch)
  search <query>      hybrid search: one line per hit — score, path, title
  watch               index every change as it is saved, until interrupted
  propose             write the note on stdin through the write gate
  get <path>          print one note's file
  list [prefix]       note paths, one per line, optionally under a namespace
  consolidate         report near-duplicate clusters (needs --ceiling)
  discards list       the discard log, newest first — one line per entry
  discards show <n>   one entry in full, as JSON
  discards restore <n>  re-propose entry n through the write gate
  init --layout swarm --roster <file>
                      set the vault up as a swarm's tiered memory

  --vault <dir>       vault root (default: the current directory)
  --agent <name>      who propose, get, list, search and discards act for
  --namespace <ns>    where propose may create the note (default: the root)
  --lexical           embed offline, without a provider (see below)
  --ceiling <d>       cosine distance two notes must be within to cluster,
                      or to count as similar in the write gate
  --layout swarm      the layout init sets up; swarm is the only one
  --roster <file>     the roster init reads (see below)
  --help, -h          this text
  --                  end of flags, so a search query may start with a dash

propose, get, list, search and discards act for an agent. A
.vault-policy.json at the vault's root scopes them: agent name ->
[{prefix, read?, write?}] rules, as JSON. With one, every call needs
--agent and answers only what that agent may touch, discards runs only for
an agent that may read the whole vault (the log spans all of it), and a
--vault inside that vault is refused. With none, any agent may do anything.
A policy file that cannot be read or is not a valid policy is an error,
never allow-all. reindex, doctor, watch and consolidate never read it.

vault init --layout swarm --roster <file> sets the vault up as a swarm's
tiered memory. The roster is JSON: role -> {"kind": "orchestrator",
"manager", "doer" or "worker", "manager": "<role>"}, where only a worker
needs a manager, it must be a manager row, and other keys are ignored. A
role name is used verbatim as its --agent and its directory, so a name that
is not one path segment is refused; any invalid row is refused before
anything is written. init runs only on a directory that is not there yet,
is empty, or holds a .vault-policy.json of its own (a re-init), and never
inside a scoped vault. A policy scopes everything below it, so init will
not turn a project root above a swarm into a vault that locks the swarm
out, and for the same reason it will not convert an existing unscoped
vault. init creates shared/, plus roles/<role>/ and proposals/<role>/ for
each manager and doer, keeping what already exists, and writes
.vault-policy.json from the roster, replacing any before it. The
orchestrator reads and writes everything; a manager or doer reads shared/
and writes its own two directories; a worker reads shared/ and its
manager's roles/, and writes nothing.

vault propose reads a note's markdown on stdin, takes its title and type
from its frontmatter or first # heading (no other frontmatter key is
kept), reindexes, and puts it through the write gate, which needs --ceiling
and a chat model (see discards restore below). It prints the action and
path, then (fell back) if the gate had to create the note instead. vault
get prints a note's file, or exits 1 with "no note at <path>" when there is
none the agent may read. list, propose and discards restore report their
reindex on stderr, so stdout holds only the answer.

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
only a human can fix (broken: nothing to point at; ambiguous: a bare
[[stem]] several notes answer to — qualify it as [[folder/stem]]) or when
doctor's own reindex hit an error it could not absorb (a confinement or
permission failure, or a failed re-entry)."""


def gate_line(r: GateResult) -> str:
    """What the write gate did, as propose and discards restore print it."""
    path = f"  {r.path}" if r.path is not None else ""
    return safe(f"{r.action}{path}{' (fell back)' if r.fell_back else ''}")


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
        out += safe(f"; reindex incomplete: {s.index_error}")
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
    if r.index_error is not None:
        lines.append(f"reindex incomplete: {r.index_error}")
    print("\n".join(safe(line) for line in lines))

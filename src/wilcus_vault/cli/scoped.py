"""The commands an agent runs: propose, get, list and search, under the vault's policy."""

import sys

from ..db import db_path, open_db
from ..decide import fetch_decider
from ..decision import Candidate
from ..embed import Embedder
from ..gate import GateOptions
from ..indexer import read_raw
from ..note import parse_note
from ..scope import VaultContext
from ..search import SearchOptions
from ..search_sql import Cutoffs
from ..term import VaultError, safe
from ..vault import Vault, open
from .commands import Args
from .policy import load_policy
from .usage import USAGE, gate_line, summary


def _open(args: Args, embedder: Embedder, gate: GateOptions | None = None) -> Vault:
    # open() validates the policy's rules before it opens a handle.
    return open(args.root, embedder, gate=gate, scopes=load_policy(args.root))


def _ctx(args: Args) -> VaultContext | None:
    # No --agent under a policy is refused by the library, in its own words.
    return None if args.agent is None else VaultContext(args.agent)


async def cmd_propose(args: Args, embedder: Embedder, rest: list[str]) -> int:
    if rest:
        raise VaultError(f"propose reads the note on stdin, not from arguments\n\n{USAGE}")
    if args.ceiling is None:
        raise VaultError(f"propose needs --ceiling\n\n{USAGE}")
    # No path, so no filename to fall back on: the title has to be in the note.
    note = parse_note(sys.stdin.read(), "")
    if note.malformed_frontmatter:
        raise VaultError(
            "propose: the note's frontmatter is malformed "
            "(unterminated, not YAML, or a title or type that is not a string)"
        )
    if note.title.strip() == "":
        raise VaultError("propose: the note has no title — give it a # heading or a title: key")
    candidate = Candidate(note.title, note.body, note.type, args.namespace)
    gate = GateOptions(fetch_decider(), Cutoffs(distance_ceiling=args.ceiling))
    vault = _open(args, embedder, gate)
    try:
        # On stderr, so stdout is only the line saying where the note went.
        print(f"indexed {summary(await vault.reindex())}", file=sys.stderr)
        print(gate_line(await vault.propose(candidate, _ctx(args))))
    finally:
        vault.close()
    return 0


async def cmd_get(args: Args, embedder: Embedder, rest: list[str]) -> int:
    if len(rest) != 1:
        raise VaultError(f"get needs one note path\n\n{USAGE}")
    vault = _open(args, embedder)
    try:
        note = await vault.get(rest[0], _ctx(args))
        # A Note holds the parse, not the file's text, so the file is read again for it.
        text = None if note is None else read_raw(vault.root, note.path)
    finally:
        vault.close()
    if text is None:
        # The same answer whether the note is absent or unreadable to this agent.
        raise VaultError(f"no note at {rest[0]}")
    # Tabs, newlines and CRLF endings print raw; a lone CR would redraw a printed line.
    print("\r\n".join(safe(line, keep="\n\t") for line in text.split("\r\n")), end="")
    return 0


async def cmd_list(args: Args, embedder: Embedder, rest: list[str]) -> int:
    if len(rest) > 1:
        raise VaultError(f"list takes at most one namespace\n\n{USAGE}")
    vault = _open(args, embedder)
    try:
        # Reindexed first, so a note written by hand since the last pass is listed.
        print(f"indexed {summary(await vault.reindex())}", file=sys.stderr)
        paths = vault.list(rest[0] if rest else None, _ctx(args))
    finally:
        vault.close()
    for path in paths:
        print(safe(path))
    return 0


async def cmd_search(args: Args, embedder: Embedder, query: str) -> int:
    if query == "":
        raise VaultError(f"search needs a query\n\n{USAGE}")
    vault = _open(args, embedder)
    try:
        # No cutoffs: the ceiling that means "irrelevant" is a property of the
        # embedder, unknowable here, so the CLI shows the ranking and lets the reader judge.
        hits = await vault.search(query, SearchOptions(ctx=_ctx(args)))
    finally:
        vault.close()
    if not hits and not _indexed(args.root):
        print("vault is not indexed (run vault reindex)")
        return 0
    lines = [safe(f"{h.score:.4f}  {h.path} — {h.title}") for h in hits]
    print("\n".join(lines) if lines else "no matches")
    return 0


def _indexed(root: str) -> bool:
    """Is any note indexed? Asked after the search, so a policy's refusal comes first."""
    db = open_db(db_path(root))
    try:
        return db.execute("select 1 from notes limit 1").fetchone() is not None
    finally:
        db.close()

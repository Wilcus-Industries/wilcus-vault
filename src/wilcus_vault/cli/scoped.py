"""The commands under the vault's policy: propose, promote, get, list, search, discards."""

import json
import sys
from collections.abc import Awaitable, Callable

from ..db import db_path, open_db
from ..decide import fetch_decider
from ..decision import Candidate
from ..discards import entry_to_json, get_discard, list_discards, restore_discard
from ..embed import Embedder
from ..gate import GateOptions
from ..gate_write import GateResult
from ..indexer import read_raw
from ..note import parse_note
from ..paths import canonical_path
from ..scope import VaultContext, compile_scopes, scope_for
from ..search import SearchOptions
from ..search_sql import Cutoffs
from ..term import VaultError, safe
from ..vault import Vault, open
from .commands import Args
from .policy import load_policy
from .usage import USAGE, gate_line, summary

GateWrite = Callable[[Vault, VaultContext | None], Awaitable[GateResult]]


def _open(args: Args, embedder: Embedder, gate: GateOptions | None = None) -> Vault:
    # open() validates the policy's rules before it opens a handle.
    return open(args.root, embedder, gate=gate, scopes=load_policy(args.root))


def _ctx(args: Args) -> VaultContext | None:
    # No --agent under a policy is refused by the library, in its own words.
    return None if args.agent is None else VaultContext(args.agent)


async def _through_gate(args: Args, embedder: Embedder, command: str, write: GateWrite) -> int:
    """A write as --agent: reindexed first, so the chat model decides among the notes
    within --ceiling as they are on disk now."""
    if args.ceiling is None:
        raise VaultError(f"{command} needs --ceiling\n\n{USAGE}")
    gate = GateOptions(fetch_decider(), Cutoffs(distance_ceiling=args.ceiling))
    vault = _open(args, embedder, gate)
    try:
        # On stderr, so stdout is only the line saying where the note went.
        print(f"indexed {summary(await vault.reindex())}", file=sys.stderr)
        print(gate_line(await write(vault, _ctx(args))))
    finally:
        vault.close()
    return 0


async def cmd_propose(args: Args, embedder: Embedder, rest: list[str]) -> int:
    if rest:
        raise VaultError(f"propose reads the note on stdin, not from arguments\n\n{USAGE}")
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
    return await _through_gate(args, embedder, "propose", lambda v, ctx: v.propose(candidate, ctx))


async def cmd_promote(args: Args, embedder: Embedder, rest: list[str]) -> int:
    if len(rest) != 1:
        raise VaultError(f"promote needs one proposal path\n\n{USAGE}")
    # The layout is the CLI's, not the library's: proposals/ in, shared/ out. Checked
    # in the canonical form the library reads, or `proposals/../shared/x.md` would pass.
    path = canonical_path(args.root, rest[0])
    if not path.startswith("proposals/"):
        raise VaultError(f"promote: {safe(rest[0])} is not under proposals/")
    return await _through_gate(
        args, embedder, "promote", lambda v, ctx: v.promote(path, "shared", ctx)
    )


async def cmd_discards(args: Args, embedder: Embedder, rest: list[str]) -> int:
    _reads_whole_vault(args)
    sub = rest[0] if rest else None
    if sub == "list":
        entries, malformed = list_discards(args.root)
        lines = [
            safe(
                f"[{e.n}] {e.at}  {e.candidate.title} — "
                f"{e.decision.action if e.decision else e.reason or '?'}"
            )
            for e in entries
        ]
        if malformed:
            lines.append(f"({malformed} unreadable line{'' if malformed == 1 else 's'} skipped)")
        print("\n".join(lines) if lines else "no discards")
        return 0
    n = int(rest[1]) if len(rest) > 1 and rest[1].isascii() and rest[1].isdigit() else 0
    if sub not in ("show", "restore") or n < 1:
        raise VaultError(f"discards needs list, show <n> or restore <n>\n\n{USAGE}")
    if sub == "show":
        entry = get_discard(args.root, n)
        if entry is None:
            raise VaultError(f"discards: no entry {n} — run vault discards list")
        print(json.dumps(entry_to_json(entry), indent=2))  # JSON escapes control characters
        return 0
    # Back through the gate against the vault as it is now, like any propose.
    return await _through_gate(
        args, embedder, "discards restore", lambda v, ctx: restore_discard(v, n, ctx)
    )


def _reads_whole_vault(args: Args) -> None:
    """The discard log holds candidates from every namespace, so under a policy only an
    agent that may read all of the vault reviews it or restores from it."""
    policy = load_policy(args.root)
    if policy is not None and not scope_for(compile_scopes(policy), _ctx(args)).reads_all():
        raise VaultError(
            f'discards: "{safe(args.agent or "")}" may not read the whole vault, '
            "and the discard log holds candidates from all of it"
        )


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

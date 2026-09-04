"""`vault <command>`: reindex, doctor, search, watch, consolidate, discards."""

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass

from .cli_usage import USAGE, print_report, summary
from .cluster import clusters
from .db import db_path, open_db
from .decide import fetch_decider
from .discards import entry_to_json, get_discard, list_discards, restore_discard
from .doctor import DoctorOptions, doctor
from .embed import Embedder, TokenOverlapEmbedder
from .fetch_embedder import FetchEmbedder
from .gate import GateOptions
from .indexer import reindex
from .search import Cutoffs, hybrid_search
from .term import VaultError, printable, safe
from .vault import open
from .watch import WatchOptions, watch

COMMANDS = ("reindex", "doctor", "search", "watch", "consolidate", "discards")


@dataclass
class Args:
    root: str
    words: list[str]
    rebuild: bool = False
    lexical: bool = False
    ceiling: float | None = None
    help: bool = False


def parse_args(argv: list[str]) -> Args:
    """Flags are consumed wherever they appear; everything else is a positional
    word, the first of which is the command. `--` ends flag parsing."""
    args = Args(root=os.getcwd(), words=[])
    literal = False
    it = iter(argv)
    for arg in it:
        if literal or (not arg.startswith("--") and arg != "-h"):
            args.words.append(arg)
        elif arg == "--":
            literal = True
        elif arg in ("--help", "-h"):
            args.help = True
        elif arg == "--rebuild":
            args.rebuild = True
        elif arg == "--lexical":
            args.lexical = True
        elif arg == "--ceiling":
            value = next(it, None)
            try:
                ceiling = float(value) if value is not None and not value.startswith("--") else None
            except ValueError:
                ceiling = None
            # A nonsense ceiling would reach the query as NaN and match nothing.
            if ceiling is None or not 0 <= ceiling <= 2:
                raise VaultError(f"--ceiling needs a cosine distance in 0..2\n\n{USAGE}")
            args.ceiling = ceiling
        elif arg == "--vault":
            root = next(it, None)
            if root is None or root.startswith("--"):
                raise VaultError(f"--vault needs a directory\n\n{USAGE}")
            args.root = root
        else:
            raise VaultError(f"unknown flag {arg}\n\n{USAGE}")
    return args


async def main(argv: list[str]) -> int:
    try:
        return await run(argv)
    except Exception as e:
        # A stack trace is not an error message: the user sees the sentence it
        # was written as, and never escape codes the terminal would act on.
        print(printable(e), file=sys.stderr)
        return 1


async def run(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.help:
        print(USAGE)
        return 0
    if not args.words or args.words[0] not in COMMANDS:
        message = (
            USAGE if not args.words else printable(f"unknown command {args.words[0]}\n\n{USAGE}")
        )
        print(message, file=sys.stderr)
        return 1
    command, rest = args.words[0], args.words[1:]
    # A bad embedder configuration raises here, before a database is opened.
    embedder: Embedder = TokenOverlapEmbedder() if args.lexical else FetchEmbedder()
    if command == "search":
        return await _search(args.root, embedder, " ".join(rest))
    if command == "reindex":
        db = open_db(db_path(args.root))
        try:
            print(f"indexed {summary(await reindex(db, args.root, embedder))}")
        finally:
            db.close()
        return 0
    if command == "consolidate":
        return await _consolidate(args, embedder)
    if command == "discards":
        return await _discards(args, embedder, rest)
    if command == "watch":
        return await _watch(args.root, embedder)
    report = await doctor(args.root, embedder, DoctorOptions(rebuild=args.rebuild))
    print_report(report)
    # Drift is repaired; links a human has to fix are not, so the exit code says so.
    return 0 if not report.broken_links and not report.ambiguous_links else 1


async def _search(root: str, embedder: Embedder, query: str) -> int:
    if query == "":
        raise VaultError(f"search needs a query\n\n{USAGE}")
    db = open_db(db_path(root))
    try:
        if db.execute("select 1 from notes limit 1").fetchone() is None:
            print("vault is not indexed (run vault reindex)")
            return 0
        # No cutoffs: the ceiling that means "irrelevant" is a property of the
        # embedder, unknowable here, so the CLI shows the ranking and lets the reader judge.
        hits = await hybrid_search(db, embedder, query)
        lines = [safe(f"{h.score:.4f}  {h.path} — {h.title}") for h in hits]
        print("\n".join(lines) if lines else "no matches")
    finally:
        db.close()
    return 0


async def _consolidate(args: Args, embedder: Embedder) -> int:
    # Report-only: a merge needs an injected merger, and the CLI wires no model.
    if args.ceiling is None:
        raise VaultError(f"consolidate needs --ceiling\n\n{USAGE}")
    db = open_db(db_path(args.root))
    try:
        print(f"indexed {summary(await reindex(db, args.root, embedder))}")
        found = clusters(db, args.ceiling)
        lines = [
            safe(
                f"{c.distance:.4f}  "
                f"{'cross-namespace  ' if c.namespace is None else ''}{' '.join(c.members)}"
            )
            for c in found
        ]
        print("\n".join(lines) if lines else "no clusters under that ceiling")
    finally:
        db.close()
    return 0


async def _discards(args: Args, embedder: Embedder, rest: list[str]) -> int:
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
    n = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else 0
    if sub not in ("show", "restore") or n < 1:
        raise VaultError(f"discards needs list, show <n> or restore <n>\n\n{USAGE}")
    if sub == "show":
        entry = get_discard(args.root, n)
        if entry is None:
            raise VaultError(f"discards: no entry {n} — run vault discards list")
        print(
            json.dumps(entry_to_json(entry), indent=2)
        )  # JSON escaping keeps control chars off the terminal
        return 0
    # restore: back through the gate against current vault state, so it needs a
    # ceiling (like propose's cutoffs) and a chat model.
    if args.ceiling is None:
        raise VaultError(f"discards restore needs --ceiling\n\n{USAGE}")
    gate = GateOptions(fetch_decider(), Cutoffs(distance_ceiling=args.ceiling))
    vault = open(args.root, embedder, gate=gate)
    try:
        print(f"indexed {summary(await vault.reindex())}")
        r = await restore_discard(vault, n)
        path = f"  {r.path}" if r.path is not None else ""
        print(safe(f"{r.action}{path}{' (fell back)' if r.fell_back else ''}"))
    finally:
        vault.close()
    return 0


async def _watch(root: str, embedder: Embedder) -> int:
    db = open_db(db_path(root))
    try:
        # Edits made while nothing was watching are picked up here, not missed.
        print(f"indexed {summary(await reindex(db, root, embedder))}")
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)

        def on_change(paths: list[str], stats: object) -> None:
            print(safe(f"{' '.join(paths)} — {summary(stats)}"))  # type: ignore[arg-type]

        watcher = watch(db, root, embedder, WatchOptions(on_change=on_change))
        print(f"watching {os.path.abspath(root)} — press ctrl-c to stop")
        await stopped.wait()
        await watcher.close()  # the pass in flight still owns the database handle
    finally:
        db.close()
    return 0


def entrypoint() -> None:
    sys.exit(asyncio.run(main(sys.argv[1:])))

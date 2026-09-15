"""`vault <command>`: reindex, doctor, search, watch, propose, get, list, consolidate, discards,
init."""

import asyncio
import os
import sys
from pathlib import Path

from ..db import db_path, open_db
from ..doctor import DoctorOptions, doctor
from ..embed import Embedder, TokenOverlapEmbedder
from ..fetch_embedder import FetchEmbedder
from ..indexer import reindex
from ..term import VaultError, printable
from .commands import Args, cmd_consolidate, cmd_watch
from .init import cmd_init
from .scoped import cmd_discards, cmd_get, cmd_list, cmd_propose, cmd_search
from .usage import USAGE, print_report, summary

COMMANDS = (
    "reindex",
    "doctor",
    "search",
    "watch",
    "propose",
    "get",
    "list",
    "consolidate",
    "discards",
    "init",
)
VALUED = {  # flag -> Args field
    "--vault": "root",
    "--agent": "agent",
    "--namespace": "namespace",
    "--layout": "layout",
    "--roster": "roster",
}


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
        elif arg in VALUED:
            value = next(it, None)
            # A flag swallowed as the value would index the wrong directory, or
            # act as an agent named `--lexical`.
            if value is None or value.startswith("--"):
                raise VaultError(f"{arg} needs a value\n\n{USAGE}")
            setattr(args, VALUED[arg], value)
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
    # Resolved once, so the policy lookup and the vault opened are one directory even
    # through a symlink: `link/..` is the link target's parent, not the lexical one.
    args.root = str(Path(args.root).resolve())
    if command == "init":
        return cmd_init(args, rest)  # embeds nothing, so no embedder configuration can fail it
    # A bad embedder configuration raises here, before a database is opened.
    embedder: Embedder = TokenOverlapEmbedder() if args.lexical else FetchEmbedder()
    if command == "search":
        return await cmd_search(args, embedder, " ".join(rest))
    if command == "propose":
        return await cmd_propose(args, embedder, rest)
    if command == "get":
        return await cmd_get(args, embedder, rest)
    if command == "list":
        return await cmd_list(args, embedder, rest)
    if command == "reindex":
        db = open_db(db_path(args.root))
        try:
            print(f"indexed {summary(await reindex(db, args.root, embedder))}")
        finally:
            db.close()
        return 0
    if command == "consolidate":
        return await cmd_consolidate(args, embedder)
    if command == "discards":
        return await cmd_discards(args, embedder, rest)
    if command == "watch":
        return await cmd_watch(args.root, embedder)
    report = await doctor(args.root, embedder, DoctorOptions(rebuild=args.rebuild))
    print_report(report)
    # Drift is repaired; links a human has to fix, and a reindex that hit an
    # error it could not absorb, are not — so the exit code says so.
    failed = report.broken_links or report.ambiguous_links or report.index_error is not None
    return 1 if failed else 0


def entrypoint() -> None:
    sys.exit(asyncio.run(main(sys.argv[1:])))

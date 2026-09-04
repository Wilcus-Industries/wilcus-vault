"""`vault <command>`: reindex, doctor, search, watch, consolidate, discards."""

import asyncio
import os
import sys

from .cli_commands import Args, cmd_consolidate, cmd_discards, cmd_search, cmd_watch
from .cli_usage import USAGE, print_report, summary
from .db import db_path, open_db
from .doctor import DoctorOptions, doctor
from .embed import Embedder, TokenOverlapEmbedder
from .fetch_embedder import FetchEmbedder
from .indexer import reindex
from .term import VaultError, printable

COMMANDS = ("reindex", "doctor", "search", "watch", "consolidate", "discards")


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
        return await cmd_search(args.root, embedder, " ".join(rest))
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
    # Drift is repaired; links a human has to fix are not, so the exit code says so.
    return 0 if not report.broken_links and not report.ambiguous_links else 1


def entrypoint() -> None:
    sys.exit(asyncio.run(main(sys.argv[1:])))

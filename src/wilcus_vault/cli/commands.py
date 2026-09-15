"""The maintenance subcommands that need more than a line: consolidate and watch."""

import asyncio
import os
import signal
from dataclasses import dataclass

from ..cluster import clusters
from ..db import db_path, open_db
from ..embed import Embedder
from ..indexer import IndexStats, reindex
from ..term import VaultError, safe
from ..watch import WatchOptions, watch
from .usage import USAGE, summary


@dataclass
class Args:
    root: str
    words: list[str]
    rebuild: bool = False
    lexical: bool = False
    ceiling: float | None = None
    agent: str | None = None
    namespace: str | None = None
    help: bool = False


async def cmd_consolidate(args: Args, embedder: Embedder) -> int:
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


def pass_line(paths: list[str], stats: IndexStats) -> str | None:
    """The line `vault watch` prints for a pass, or None when nothing changed: an
    editor rewriting identical bytes is not news."""
    if not (stats.added or stats.updated or stats.removed):
        return None
    return safe(f"{' '.join(paths)} — {summary(stats)}")


async def cmd_watch(root: str, embedder: Embedder) -> int:
    db = open_db(db_path(root))
    try:
        # Edits made while nothing was watching are picked up here, not missed.
        print(f"indexed {summary(await reindex(db, root, embedder))}")
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)

        def on_change(paths: list[str], stats: IndexStats) -> None:
            if (line := pass_line(paths, stats)) is not None:
                print(line)

        watcher = watch(db, root, embedder, WatchOptions(on_change=on_change))
        print(f"watching {os.path.abspath(root)} — press ctrl-c to stop")
        await stopped.wait()
        await watcher.close()  # the pass in flight still owns the database handle
    finally:
        db.close()
    return 0

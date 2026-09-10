"""Keep the index warm while a human edits the vault.

A convenience, never a source of truth: every pass is the same hash-diff the
indexer runs, and anything the watcher misses is found again by doctor. Nothing
here may raise at the caller mid-run.
"""

import asyncio
import contextlib
import os
import sqlite3
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from watchfiles import Change, awatch

from .embed import Embedder
from .indexer import IndexStats, index_paths, is_note_path, reindex
from .term import printable


def _note_events(_change: Change, path: str) -> bool:
    """Only `.md` paths reach Python at all.

    The index writes under `.vault/`, and in WAL mode it writes often — every
    one of those is an event the watcher would wake for and then discard. Kept
    coarse on purpose: `touch` still applies the real `is_note_path` rule, this
    only stops the index feeding the watcher its own tail.
    """
    return path.endswith(".md")


def _log_error(error: BaseException) -> None:
    print(f"vault watch: {printable(error)}", file=sys.stderr)


@dataclass(frozen=True)
class WatchOptions:
    debounce: float = 0.25  # quiet period per path before it is indexed, in seconds
    on_change: Callable[[list[str], IndexStats], None] | None = None  # after every pass
    on_error: Callable[[BaseException], None] = _log_error  # failures are logged, never raised


class Watcher:
    """Debounce per path, index the paths that come due together in one pass, and
    queue new ones behind a pass in flight rather than running two at once."""

    def __init__(
        self, db: sqlite3.Connection, root: str | Path, embedder: Embedder, options: WatchOptions
    ) -> None:
        self._db = db
        self._root = Path(root)
        self._embedder = embedder
        self._options = options
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._ready: set[str] = set()
        self._running: asyncio.Task[None] | None = None
        self._closed = False
        self._stop = asyncio.Event()
        self._fs_task = asyncio.create_task(self._watch_fs())

    def touch(self, rel: str) -> None:
        """Feed in a vault-relative path as if the filesystem had reported it."""
        # One filter for both entry points; it also stops the index's own writes
        # under `.vault/` from feeding the watcher its own tail.
        if self._closed or not is_note_path(rel):
            return
        if timer := self._timers.pop(rel, None):
            timer.cancel()
        loop = asyncio.get_running_loop()
        self._timers[rel] = loop.call_later(self._options.debounce, self._due, rel)

    async def idle(self) -> None:
        """Resolves once every debounced path has been indexed and nothing is in flight."""
        while self._timers or self._ready or self._running is not None:
            await asyncio.sleep(0.001)

    def close(self) -> Awaitable[None]:
        """Stop watching now; pending paths are dropped (doctor finds them). The
        returned awaitable completes when the pass in flight, which still holds
        the database, is done, so closing the database after it is safe."""
        self._closed = True
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        self._ready.clear()
        self._stop.set()
        return asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        if self._running is not None:
            await self._running
        await self._fs_task

    def _due(self, rel: str) -> None:
        self._timers.pop(rel, None)
        self._ready.add(rel)
        if self._running is None:
            self._running = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        try:
            # Yield once so paths due in the same tick join this pass.
            await asyncio.sleep(0)
            while self._ready and not self._closed:
                paths = sorted(self._ready)
                self._ready.clear()
                try:
                    stats = await index_paths(self._db, self._root, self._embedder, paths)
                    if stats.reembedded:
                        # A model swap invalidated every vector: re-embed the rest
                        # of the vault too, but keep this pass's qualify report.
                        full = await reindex(self._db, self._root, self._embedder)
                        qualified = stats.qualified + full.qualified
                        stats = replace(full, reembedded=True, qualified=qualified)
                    if self._options.on_change is not None:
                        self._options.on_change(paths, stats)
                except Exception as e:
                    self._report(e)
        finally:
            # Whatever escaped, the watcher must not believe a pass is still in
            # flight: that would wedge every later change and hang idle().
            self._running = None

    async def _watch_fs(self) -> None:
        try:
            async for changes in awatch(
                self._root,
                watch_filter=_note_events,
                stop_event=self._stop,
                debounce=50,
                step=25,
            ):
                for _, path in changes:
                    self.touch(os.path.relpath(path, self._root))
        except Exception as e:
            self._report(e)

    def _report(self, error: BaseException) -> None:
        """Report a failure and carry on. A reporter that raises has nowhere left to go."""
        with contextlib.suppress(Exception):
            self._options.on_error(error)


def watch(
    db: sqlite3.Connection,
    root: str | Path,
    embedder: Embedder,
    options: WatchOptions | None = None,
) -> Watcher:
    """Watch the vault and index what changes, until `close()`. Needs a running event loop."""
    return Watcher(db, root, embedder, options or WatchOptions())

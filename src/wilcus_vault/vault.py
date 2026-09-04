"""The public API. Embedder, decider and merger are injected; the vault never
hardcodes a provider."""

import os
import sqlite3
from pathlib import Path

from .consolidate import ConsolidateOptions, ConsolidateReport, ConsolidateRun
from .consolidate import consolidate as run_consolidate
from .db import db_path, open_db
from .decision import Candidate
from .doctor import DoctorOptions, DoctorReport
from .doctor import doctor as run_doctor
from .embed import Embedder
from .gate import GateOptions, GateResult
from .gate import propose as run_gate
from .indexer import IndexStats, is_note_path, note_entry, read_note
from .indexer import reindex as reindex_vault
from .note import Note
from .paths import confined_path
from .scope import (
    CompiledPolicy,
    ScopePolicy,
    VaultContext,
    compile_scopes,
    normalize_prefix,
    scope_for,
)
from .search import SearchHit, SearchOptions, hybrid_search
from .term import VaultError
from .watch import Watcher, WatchOptions
from .watch import watch as watch_vault


class Vault:
    """A vault handle. The index is created on demand; the files are the truth."""

    def __init__(
        self,
        root: str | Path,
        embedder: Embedder,
        gate: GateOptions | None = None,  # required by propose
        consolidate: ConsolidateOptions | None = None,  # required by consolidate
        # Per-agent namespace rules. Absent means allow-all; present, it is an
        # allowlist that fails closed and every call then needs a VaultContext.
        scopes: ScopePolicy | None = None,
    ) -> None:
        self.root: Path = Path(os.path.abspath(root))
        self._embedder = embedder
        self._gate = gate
        self._consolidate = consolidate
        # Validated before anything is opened, so a contradictory policy costs no handle.
        self._policy: CompiledPolicy | None = compile_scopes(scopes)
        self._db: sqlite3.Connection = open_db(db_path(self.root))

    async def search(self, query: str, options: SearchOptions | None = None) -> list[SearchHit]:
        """Hybrid search. Under a scope policy the answer is up to N readable hits."""
        opts = options or SearchOptions()
        scope = scope_for(self._policy, opts.ctx)
        return await hybrid_search(self._db, self._embedder, query, opts, scope)

    async def get(self, path: str, ctx: VaultContext | None = None) -> Note | None:
        """One note by its identity, the vault-relative path including `.md`.

        Read from the file, so a stale index row cannot change the answer. None
        when nothing is there, or when the agent may not read it: a scope is not
        an existence oracle. A path that escapes the vault or runs through a
        hidden or symlinked directory raises.
        """
        scope = scope_for(self._policy, ctx)
        # Canonicalized into the form the scan stores, so `./x.md`, `a//x.md`
        # and an absolute path inside the vault all name one note.
        norm = os.path.relpath(os.path.join(self.root, path), self.root).replace("\\", "/")
        if "\0" in norm:
            return None
        # The parent is confined, not the leaf: a symlink where the note should
        # be is already "not a note", and a path cannot escape through its last segment.
        confined_path(self.root, os.path.dirname(norm))
        if not scope.may("read", norm):
            return None
        if not is_note_path(norm) or note_entry(self.root, norm) is None:
            return None
        return read_note(self.root, norm)

    def list(self, prefix: str | None = None, ctx: VaultContext | None = None) -> list[str]:
        """Vault-relative paths of every note, sorted. `prefix` names a namespace and
        matches on segment boundaries. Superseded notes are listed: this is the
        note set, not the search set."""
        scope = scope_for(self._policy, ctx)
        under = normalize_prefix(prefix)
        paths = [r["path"] for r in self._db.execute("select path from notes order by path")]
        return [p for p in paths if p.startswith(under) and scope.may("read", p)]

    async def propose(self, candidate: Candidate, ctx: VaultContext | None = None) -> GateResult:
        """The write gate: search, decide, apply. `ctx` names the calling agent and
        is stamped as provenance on every note the gate authors."""
        if self._gate is None:
            raise VaultError("vault: propose needs a gate — Vault(..., gate=GateOptions(...))")
        scope = scope_for(self._policy, ctx)
        return await run_gate(self._db, self.root, self._embedder, candidate, self._gate, scope)

    async def consolidate(self, run: ConsolidateRun) -> ConsolidateReport:
        """The consolidation pass. An operator operation like doctor, unscoped:
        `run.ctx` is provenance only. Dry-run unless `write=True`."""
        if self._consolidate is None:
            raise VaultError(
                "vault: consolidate needs a merger — "
                "Vault(..., consolidate=ConsolidateOptions(...))"
            )
        return await run_consolidate(self._db, self.root, self._embedder, self._consolidate, run)

    async def reindex(self) -> IndexStats:
        return await reindex_vault(self._db, self.root, self._embedder)

    async def doctor(self, options: DoctorOptions | None = None) -> DoctorReport:
        report = await run_doctor(self.root, self._embedder, options)
        # A rebuild renamed a fresh index over the old file; our handle still
        # points at the replaced inode, so take the new one.
        if options is not None and options.rebuild:
            self._db.close()
            self._db = open_db(db_path(self.root))
        return report

    def watch(self, options: WatchOptions | None = None) -> Watcher:
        """Keep the index up to date as the files change, until `close()`."""
        return watch_vault(self._db, self.root, self._embedder, options)

    def close(self) -> None:
        self._db.close()


def open(
    root: str | Path,
    embedder: Embedder,
    *,
    gate: GateOptions | None = None,
    consolidate: ConsolidateOptions | None = None,
    scopes: ScopePolicy | None = None,
) -> Vault:
    """Open a vault at `root`."""
    return Vault(root, embedder, gate, consolidate, scopes)

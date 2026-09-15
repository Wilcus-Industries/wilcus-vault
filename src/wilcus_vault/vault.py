"""The public API. Embedder, decider and merger are injected; the vault never
hardcodes a provider."""

import math
import os
import sqlite3
import time
from pathlib import Path

from .consolidate import ConsolidateOptions, ConsolidateReport, ConsolidateRun
from .consolidate import consolidate as run_consolidate
from .db import db_path, open_db
from .decision import Candidate
from .doctor import DoctorOptions, DoctorReport
from .doctor import doctor as run_doctor
from .embed import Embedder
from .gate import GateOptions
from .gate import propose as run_gate
from .gate_write import GateResult
from .indexer import IndexStats, is_note_path, note_entry, read_note
from .indexer import reindex as reindex_vault
from .note import Note
from .paths import canonical_path, confined_path
from .promote import PromoteResult
from .promote import promote as run_promote
from .scope import (
    CompiledPolicy,
    Scope,
    ScopePolicy,
    VaultContext,
    compile_scopes,
    normalize_prefix,
    scope_for,
)
from .search import SearchOptions, hybrid_search
from .search_sql import SearchHit
from .term import VaultError, printable, safe
from .watch import Watcher, WatchOptions
from .watch import watch as watch_vault


class Vault:
    """A vault handle. The index is created on demand; the files are the truth."""

    def __init__(
        self,
        root: str | Path,
        embedder: Embedder,
        gate: GateOptions | None = None,  # required by propose and promote
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
        self._walked = -math.inf  # when the gate last re-read the files

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
        hidden or symlinked directory raises, and so does a path we could not
        read — an unreadable directory in the way, an I/O error. "I cannot say"
        is not "there is nothing there", and it must not answer None.
        """
        scope = scope_for(self._policy, ctx)
        norm = canonical_path(self.root, path)
        if "\0" in norm:
            return None
        # The parent is confined, not the leaf: a symlink where the note should
        # be is already "not a note", and a path cannot escape through its last segment.
        confined_path(self.root, os.path.dirname(norm))
        if not scope.may("read", norm):
            return None
        try:
            if not is_note_path(norm) or note_entry(self.root, norm) is None:
                return None
            return read_note(self.root, norm)
        except OSError as e:
            # As a VaultError, like every other refusal this surface raises.
            raise VaultError(f"vault: cannot read {safe(path)}: {printable(e)}") from e

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
        return await self._gate_write(candidate, scope_for(self._policy, ctx))

    async def promote(
        self, path: str, namespace: str, ctx: VaultContext | None = None
    ) -> PromoteResult:
        """One note through the write gate into `namespace`, judged against that
        namespace alone, then removed — or kept, if it changed while the gate ran. The
        agent must be able to read and write the note; `ctx.source` defaults to its path."""
        note = await self.get(path, ctx)
        if note is None:  # absent, or not this agent's to read: one answer, as with get
            raise VaultError(f"promote: no note at {safe(path)}")
        scope = scope_for(self._policy, ctx)
        return await run_promote(
            self._db, self.root, self._embedder, note, namespace, scope, self._gate_write
        )

    async def _gate_write(
        self, candidate: Candidate, scope: Scope, exclude: str | None = None
    ) -> GateResult:
        """The gate as propose and promote run it; `exclude` is never shown as similar."""
        if self._gate is None:
            raise VaultError(
                "vault: propose and promote need a gate — Vault(..., gate=GateOptions(...))"
            )
        # The clock lives here, not in the gate: freshness is a property of this
        # handle's view of the files, and a burst of writes shares one walk.
        now = time.monotonic()
        walk = now - self._walked >= self._gate.freshness
        result = await run_gate(
            self._db, self.root, self._embedder, candidate, self._gate, scope, walk, exclude
        )
        if walk:
            self._walked = now
        return result

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
        # A rebuild works in the live index file, so this handle stays valid.
        return await run_doctor(self.root, self._embedder, options)

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

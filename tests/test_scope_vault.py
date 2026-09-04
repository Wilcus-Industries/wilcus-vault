"""Per-agent scopes enforced through the vault: search, get, list and propose.
Deterministic embedder, fake deciders, no network."""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace

import pytest
from conftest import MakeVault
from test_scope_policy import POLICY

from wilcus_vault.db import db_path, open_db
from wilcus_vault.decision import Candidate, DeciderInput, Decision
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.gate import GateOptions
from wilcus_vault.scope import ScopePolicy, VaultContext
from wilcus_vault.search import SearchOptions, hybrid_search
from wilcus_vault.search_sql import Cutoffs, SearchHit
from wilcus_vault.term import VaultError
from wilcus_vault.vault import Vault, open

CUTOFFS = Cutoffs(distance_ceiling=0.9, bm25_ceiling=0)
NOTES = VaultContext("core/notes", "task-42")
SCHEDULER = VaultContext("core/scheduler")

VAULT = {
    "notes/acme-renewal.md": "---\ntype: customer\n---\n# Acme renewal\n\n"
    "The Acme renewal closes in March at the agreed renewal pricing.\n",
    "notes/support-rota.md": "# Support rota\n\nWho carries the pager each week.\n",
    "notes/hub.md": "# Renewal hub\n\n"
    "The renewal hub: [[secret/plans]] and [[notes/support-rota]].\n",
    "ledger/q3.md": "# Q3 ledger\n\nQ3 revenue by account, including the Acme renewal line.\n",
    "ledger-archive/q3.md": "# Archived Q3\n\nSuperseded Acme renewal numbers.\n",
    "secret/plans.md": "# Secret plans\n\nThe Acme renewal pricing nobody else may read.\n",
    "root-note.md": "# Root note\n\nAt the vault root, about the Acme renewal.\n",
}

CANDIDATE = Candidate(
    title="Acme renewal 2026",
    type="customer",
    namespace="notes",
    body="The Acme renewal closes in March 2026 at the agreed renewal pricing.\n",
)

Decider = Callable[[DeciderInput], Awaitable[Decision]]
OpenVault = Callable[..., Awaitable[Vault]]


async def _create(_input: DeciderInput) -> Decision:
    return Decision("create")


@pytest.fixture
async def open_vault(
    make_vault: MakeVault, embedder: TokenOverlapEmbedder
) -> AsyncIterator[OpenVault]:
    opened: list[Vault] = []

    async def make(
        scopes: ScopePolicy | None, decider: Decider = _create, files: dict[str, str] = VAULT
    ) -> Vault:
        v = open(make_vault(files), embedder, gate=GateOptions(decider, CUTOFFS), scopes=scopes)
        opened.append(v)
        await v.reindex()
        return v

    yield make
    for v in opened:
        v.close()


def paths(hits: list[SearchHit]) -> list[str]:
    return [h.path for h in hits]


async def test_no_policy_is_allow_all(open_vault: OpenVault) -> None:
    v = await open_vault(None)
    anyone = VaultContext("anyone")
    assert "secret/plans.md" in paths(await v.search("acme renewal"))
    assert "secret/plans.md" in paths(await v.search("acme renewal", SearchOptions(ctx=anyone)))
    assert (await v.get("secret/plans.md")).title == "Secret plans"  # type: ignore[union-attr]
    assert (await v.get("secret/plans.md", anyone)).title == "Secret plans"  # type: ignore[union-attr]
    assert "secret/plans.md" in v.list()
    assert v.list("secret", anyone) == ["secret/plans.md"]
    # an agent no policy ever named may still write, because there is no policy
    r = await v.propose(replace(CANDIDATE, namespace="ledger"), anyone)
    assert (r.action, r.path) == ("create", "ledger/acme-renewal-2026.md")


async def test_search_filter_measures_a_prefix_the_way_sqlite_does(open_vault: OpenVault) -> None:
    # `"🔒secret/"` is 8 characters and 9 UTF-16 code units: a length measured in
    # the wrong unit makes the deny arm miss and the root allow answer instead.
    v = await open_vault(
        {
            "a": [
                {"prefix": "", "read": True, "write": False},
                {"prefix": "🔒secret/", "read": False},
            ]
        },
        _create,
        {
            "🔒secret/plans.md": "# Secret plans\n\n"
            "The Acme renewal pricing nobody else may read.\n",
            "notes/acme-renewal.md": "# Acme renewal\n\nThe Acme renewal closes in March.\n",
        },
    )
    a = VaultContext("a")
    assert paths(await v.search("acme renewal pricing", SearchOptions(n=10, ctx=a))) == [
        "notes/acme-renewal.md"
    ]
    assert v.list(None, a) == ["notes/acme-renewal.md"]


async def test_a_policy_in_force_fails_closed(open_vault: OpenVault) -> None:
    v = await open_vault(POLICY)
    # Silence is how an orchestrator typo makes an agent re-create the memory it
    # thinks it lost, so every one of these raises instead.
    with pytest.raises(VaultError, match="VaultContext"):
        await v.search("acme")
    with pytest.raises(VaultError, match="VaultContext"):
        await v.get("notes/hub.md")
    with pytest.raises(VaultError, match="VaultContext"):
        v.list()
    with pytest.raises(VaultError, match="VaultContext"):
        await v.propose(CANDIDATE)

    typo = VaultContext("core/typo")
    with pytest.raises(VaultError, match="has no scope"):
        await v.search("acme", SearchOptions(ctx=typo))
    with pytest.raises(VaultError, match="has no scope"):
        await v.get("notes/hub.md", typo)
    with pytest.raises(VaultError, match="has no scope"):
        v.list(None, typo)
    with pytest.raises(VaultError, match="has no scope"):
        await v.propose(CANDIDATE, typo)

    # `{}` and None sit on opposite sides of the fail-closed line.
    empty = await open_vault({})
    with pytest.raises(VaultError, match="has no scope"):
        await empty.get("notes/hub.md", NOTES)


async def test_hybrid_search_refuses_a_context_only_the_facade_can_resolve(
    open_vault: OpenVault, embedder: TokenOverlapEmbedder
) -> None:
    v = await open_vault(POLICY)
    db = open_db(db_path(v.root))
    try:
        # A direct caller handing `ctx` to the search function would otherwise be
        # answered allow-all: the silent grant the allowlist exists to prevent.
        with pytest.raises(VaultError, match=r"call vault\.search"):
            await hybrid_search(db, embedder, "acme", SearchOptions(ctx=NOTES))
    finally:
        db.close()


async def test_search_returns_only_readable_hits(open_vault: OpenVault) -> None:
    v = await open_vault(POLICY)
    query = "acme renewal pricing"
    # Unscoped, the vault answers with notes from every namespace.
    unscoped = paths(await v.search(query, SearchOptions(n=10, ctx=SCHEDULER)))
    assert {"secret/plans.md", "ledger-archive/q3.md", "root-note.md"} <= set(unscoped)

    scoped = paths(await v.search(query, SearchOptions(n=10, ctx=NOTES)))
    assert scoped
    assert all(p.startswith(("notes/", "ledger/")) for p in scoped)
    assert "ledger/q3.md" in scoped  # readable, just not writable


async def test_scope_filter_runs_before_the_cap(open_vault: OpenVault) -> None:
    # Three short, dense notes the agent cannot read outrank the one it can on
    # both signals; the over-fetch (3×N) is what still leaves room for it.
    files = {
        "notes/acme-renewal.md": "# Acme renewal\n\nThe Acme renewal closes in March. Contract, "
        "notice periods, uplift caps,\nescalation ladder, finance sign off, procurement "
        "thresholds, quarterly review cadence,\nsupport handover, pager rota, and where an "
        "approved pricing exception is recorded.\n",
    }
    for n in (1, 2, 3):
        files[f"secret/plan-{n}.md"] = "# Acme renewal pricing\n\nAcme renewal pricing.\n"
    v = await open_vault(POLICY, _create, files)

    query = "acme renewal pricing"
    assert paths(await v.search(query, SearchOptions(n=1, ctx=SCHEDULER))) == ["secret/plan-1.md"]
    # "up to N": the readable note is outside the 3×1 over-fetch, so a scoped
    # agent sees fewer hits. Accepted, and documented.
    assert await v.search(query, SearchOptions(n=1, ctx=NOTES)) == []
    # Widen N and the over-fetch reaches past the crowd to the note it may read.
    assert paths(await v.search(query, SearchOptions(n=3, ctx=NOTES))) == ["notes/acme-renewal.md"]


async def test_expand_links_neighbours_pass_the_same_read_filter(open_vault: OpenVault) -> None:
    v = await open_vault(POLICY)

    # Cutoffs keep the direct set to the hub note itself, so both of its
    # neighbours are inside the expansion's own cap of N.
    async def expansions(ctx: VaultContext) -> list[str]:
        opts = SearchOptions(n=2, cutoffs=CUTOFFS, expand_links=True, ctx=ctx)
        return paths([h for h in await v.search("renewal hub", opts) if h.expansion])

    # A forbidden note is one wikilink away from a hit the agent may read.
    assert "secret/plans.md" in await expansions(SCHEDULER)
    scoped = await expansions(NOTES)
    assert "notes/support-rota.md" in scoped
    assert "secret/plans.md" not in scoped


async def test_get_an_unreadable_note_is_none_exactly_like_an_absent_one(
    open_vault: OpenVault,
) -> None:
    v = await open_vault(POLICY)
    # A scope is not an existence oracle: these two answers must be identical.
    assert await v.get("secret/plans.md", NOTES) is None
    assert await v.get("secret/nothing-here.md", NOTES) is None
    assert (await v.get("secret/plans.md", SCHEDULER)).title == "Secret plans"  # type: ignore[union-attr]
    assert (await v.get("ledger/q3.md", NOTES)).title == "Q3 ledger"  # type: ignore[union-attr]


async def test_list_is_filtered_to_what_the_agent_may_read(open_vault: OpenVault) -> None:
    v = await open_vault(POLICY)
    assert v.list(None, NOTES) == [
        "ledger/q3.md",
        "notes/acme-renewal.md",
        "notes/hub.md",
        "notes/support-rota.md",
    ]
    assert v.list("secret", NOTES) == []
    assert "secret/plans.md" in v.list(None, SCHEDULER)

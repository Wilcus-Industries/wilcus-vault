"""Retrieval evals: deterministic, seeded vaults, the TokenOverlapEmbedder standing
in for semantics. No network."""

import sqlite3
from pathlib import Path

import pytest
from conftest import MakeVault

from wilcus_vault.db import db_path, open_db
from wilcus_vault.embed import Embedder, TokenOverlapEmbedder, Vector
from wilcus_vault.indexer import reindex
from wilcus_vault.scope import Scope
from wilcus_vault.search import SearchOptions, fts_query, hybrid_search
from wilcus_vault.search_sql import Cutoffs, SearchHit

EMBEDDER = TokenOverlapEmbedder()

# The eval vault, seeded so the query "acme renewal pricing" has a *different*
# top hit on each signal. "acme" and "renewal" sit in half of the ten notes, so
# BM25 gives them no IDF and the keyword ranking turns entirely on the rare
# term "pricing"; the vector ranking turns on token purity instead. That pulls
# the two signals apart: `renewal-stub` is the vector top (nothing but query
# tokens) and nowhere near the BM25 top, `pricing-model` is the BM25 top (the
# rare term, three times) and nowhere near the vector top, and `acme-renewal`,
# the note a human wants, is second on both and first on neither.
EVAL_VAULT: dict[str, str] = {
    "ledger/invoice-2031.md": "---\ntype: ledger\n---\n# Invoice INV-2031\n\n"
    "Annual license invoice INV-2031 issued to Acme Corp, net 30 days.\n",
    "notes/account-history.md": "# Account history\n\nThe Acme account has run since 2019; "
    "the last renewal was signed without changes and the paperwork sits with legal.\n",
    "notes/acme-contacts.md": "# Acme contacts\n\nDay to day contacts at Acme: "
    "the procurement lead owns the Acme renewal thread.\n",
    "notes/acme-renewal.md": "---\ntype: customer\n---\n# Acme renewal\n\nThe Acme renewal "
    "closes in March. Renewal pricing is agreed; Acme signs the order form. "
    "See [[support-rota]].\n",
    "notes/discount-approvals.md": "# Discount approvals\n\nWho signs off a discount, "
    "the escalation ladder, the finance review window, and where an approved pricing "
    "exception is recorded afterwards.\n",
    "notes/globex.md": "# Globex\n\nVendor policy for the Globex account: procurement, "
    "security review, invoicing cadence.\n",
    "notes/pricing-model.md": "# Pricing model\n\nPricing bands, pricing exceptions, discount "
    "approval, procurement thresholds, escalation path, finance sign off, quarterly review "
    "cadence.\n",
    "notes/renewal-playbook.md": "# Renewal playbook\n\nThe renewal playbook: notice periods, "
    "uplift caps, and the renewal calendar.\n",
    "notes/renewal-stub.md": "# Acme renewal\n\nAcme renewal.\n",
    "notes/support-rota.md": "# Support rota\n\nWho carries the pager each week, "
    "and how the handover works.\n",
}


async def indexed(
    make_vault: MakeVault, files: dict[str, str], embedder: Embedder = EMBEDDER
) -> tuple[Path, sqlite3.Connection]:
    root = make_vault(files)
    db = open_db(db_path(root))
    await reindex(db, root, embedder)
    return root, db


def paths(hits: list[SearchHit]) -> list[str]:
    return [h.path for h in hits]


async def test_eval_an_exact_identifier_is_the_top_hit(make_vault: MakeVault) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    hits = await hybrid_search(db, EMBEDDER, "INV-2031")
    assert hits[0].path == "ledger/invoice-2031.md"
    db.close()


async def test_eval_a_reworded_query_finds_the_note_through_the_vector_path(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(
        make_vault,
        {
            "notes/onboarding.md": "# Customer onboarding\n\nSend the welcome email, create the "
            "shared workspace, then schedule a call to kick off the work.\n",
            "notes/email-policy.md": "# Email policy\n\nRetention rules for email archives.\n",
            "notes/globex.md": EVAL_VAULT["notes/globex.md"],
        },
    )
    # reordered and padded: no phrase in common, plenty of tokens in common
    hits = await hybrid_search(
        db, EMBEDDER, "how do we kick off a new customer: workspace, welcome call, email"
    )
    assert hits[0].path == "notes/onboarding.md"

    # A hyphenated term is one FTS phrase ("kick off" adjacent, in that order)
    # and two embedder tokens, so this query has no keyword match at all: only
    # the vector side can carry it.
    vec_only = await hybrid_search(db, EMBEDDER, "customer-workspace")
    assert (vec_only[0].path, vec_only[0].fts_rank, vec_only[0].vec_rank) == (
        "notes/onboarding.md",
        None,
        1,
    )
    db.close()


async def test_eval_fusion_beats_either_signal_alone(make_vault: MakeVault) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    # n=1 ⇒ each signal over-fetches 3, so a note outside a signal's top 3 is
    # absent from it entirely: exactly what a single-signal search would miss.
    hits = await hybrid_search(db, EMBEDDER, "acme renewal pricing", SearchOptions(n=1))
    assert paths(hits) == ["notes/acme-renewal.md"]

    both = await hybrid_search(db, EMBEDDER, "acme renewal pricing", SearchOptions(n=5))
    target = next(h for h in both if h.path == "notes/acme-renewal.md")
    # second-best on both signals: neither ranking alone puts it first
    assert target.vec_rank == 2
    assert target.fts_rank == 2
    assert next(h for h in both if h.vec_rank == 1).path == "notes/renewal-stub.md"
    assert next(h for h in both if h.fts_rank == 1).path == "notes/pricing-model.md"
    # and it still wins the fusion, by a clear margin over both single-signal tops
    assert both[0].path == "notes/acme-renewal.md"
    assert target.score > both[1].score
    db.close()


async def test_eval_a_superseded_note_is_filtered_out_of_both_signals(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(
        make_vault,
        {
            "notes/old-terms.md": "---\nsuperseded_by: notes/new-terms.md\n---\n"
            "# Acme payment terms\n\nAcme pays net 60 on the annual invoice.\n",
            "notes/new-terms.md": "# Acme payment terms 2026\n\n"
            "Acme pays net 30 on the annual invoice.\n",
        },
    )
    hits = await hybrid_search(db, EMBEDDER, "acme payment terms net")
    assert paths(hits) == ["notes/new-terms.md"]
    db.close()


async def test_eval_cutoffs_make_a_garbage_query_return_nothing(make_vault: MakeVault) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    garbage = "zzqqxx wwvvuu"
    # without cutoffs the vector side still returns its k nearest, however far
    assert len(await hybrid_search(db, EMBEDDER, garbage)) > 0
    # with them, nothing survives, and an empty result is a valid outcome
    opts = SearchOptions(cutoffs=Cutoffs(distance_ceiling=0.5, bm25_ceiling=-1))
    assert await hybrid_search(db, EMBEDDER, garbage, opts) == []
    db.close()


async def test_each_cutoff_narrows_its_own_signal_in_the_direction_it_says(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    q = "acme renewal pricing"

    # BM25 alone: FTS5's rank is negative and lower is better, so -1 keeps only
    # the strongest keyword hit. A reversed comparison would keep the rest.
    keyword = await hybrid_search(
        db, EMBEDDER, q, SearchOptions(n=10, cutoffs=Cutoffs(bm25_ceiling=-1))
    )
    assert [h.path for h in keyword if h.fts_rank is not None] == ["notes/pricing-model.md"]
    assert len([h for h in keyword if h.vec_rank is not None]) > 1  # vector side untouched

    # Cosine distance alone: 0.3 keeps the two nearest (0.184, 0.265) and drops
    # the rest, while FTS keeps returning rows.
    vector = await hybrid_search(
        db, EMBEDDER, q, SearchOptions(n=10, cutoffs=Cutoffs(distance_ceiling=0.3))
    )
    assert sorted(h.path for h in vector if h.vec_rank is not None) == [
        "notes/acme-renewal.md",
        "notes/renewal-stub.md",
    ]
    assert len([h for h in vector if h.fts_rank is not None]) > 2  # FTS side untouched
    db.close()


async def test_eval_fts5_syntax_in_the_query_is_text_not_syntax(make_vault: MakeVault) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    hostile = [
        '"a OR b"',
        "NEAR(acme renewal, 2)",
        "acme AND NOT globex",
        "acme*",
        "body:acme",
        "^acme",
        '"""',
        "((((",
        "-",
        "{acme}",
        'acme" OR "renewal',
    ]
    for q in hostile:
        assert isinstance(await hybrid_search(db, EMBEDDER, q), list)
    # every term is quoted, embedded quotes are doubled, and operators are terms
    assert fts_query("acme OR renewal") == '"acme" OR "OR" OR "renewal"'
    assert fts_query('a"b') == '"a""b"'
    assert fts_query("  ***  ") is None
    # a whole note body is a legitimate query (the write gate passes one), so the
    # keyword side is capped at the first 32 distinct terms
    long = fts_query(" ".join(f"w{i} w{i}" for i in range(200)))
    assert long is not None
    assert len(long.split(" OR ")) == 32
    assert long.startswith('"w0" OR "w1" OR "w2"')
    db.close()


async def test_a_query_the_cutoffs_exhausted_does_not_pay_for_a_wider_one(
    make_vault: MakeVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short answer is only worth widening when the cut was full.

    Each ceiling bounds the quantity its own signal is ordered by, so a row it
    rejected inside the cut has no better twin outside it — widening re-derives
    the same empty answer over the whole index. That is the write gate's every
    call: it must set a cutoff, and a novel candidate matches nothing.
    """
    import wilcus_vault.search_sql as search_sql

    root, db = await indexed(make_vault, EVAL_VAULT)
    cuts: list[int] = []
    real = search_sql._fuse

    def counted(
        conn: sqlite3.Connection,
        vector: Vector | None,
        match: str | None,
        n: int,
        cutoffs: Cutoffs,
        scope: Scope,
        cut: int,
    ) -> list[SearchHit]:
        cuts.append(cut)
        return real(conn, vector, match, n, cutoffs, scope, cut)

    monkeypatch.setattr(search_sql, "_fuse", counted)

    tight = Cutoffs(distance_ceiling=0.001, bm25_ceiling=-99999.0)
    assert await hybrid_search(db, EMBEDDER, "zebra xylophone", SearchOptions(5, tight)) == []
    # One pass at 3×N, and no second one. That a genuinely crowded-out query
    # *does* widen is pinned where it is visible to a caller, in
    # test_scope_vault.py: that test fails outright without the widening.
    assert cuts == [15], "an exhausted query widened anyway"
    db.close()

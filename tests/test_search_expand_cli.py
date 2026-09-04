"""Link expansion, and `vault search` on the command line."""

import re
import sqlite3

import pytest
from conftest import MakeVault
from fakes import stub_embedder
from test_search import EMBEDDER, EVAL_VAULT, indexed, paths

from wilcus_vault.cli import main
from wilcus_vault.search import SearchOptions, hybrid_search
from wilcus_vault.term import VaultError


async def test_expand_links_appends_one_hop_neighbours_below_every_direct_hit(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(
        make_vault,
        {
            "notes/acme-renewal.md": EVAL_VAULT["notes/acme-renewal.md"],  # links to support-rota
            "notes/support-rota.md": EVAL_VAULT["notes/support-rota.md"],
            "notes/globex.md": EVAL_VAULT["notes/globex.md"],
        },
    )
    plain = await hybrid_search(db, EMBEDDER, "acme renewal march", SearchOptions(n=1))
    assert paths(plain) == ["notes/acme-renewal.md"]

    expanded = await hybrid_search(
        db, EMBEDDER, "acme renewal march", SearchOptions(n=1, expand_links=True)
    )
    assert paths(expanded) == ["notes/acme-renewal.md", "notes/support-rota.md"]
    e = expanded[1]
    assert (e.expansion, e.score, e.vec_rank, e.fts_rank) == (True, 0, None, None)
    # an expansion never outranks a direct hit
    assert [h for h in expanded if not h.expansion][-1].score > expanded[1].score
    db.close()


async def test_expand_links_walks_a_path_qualified_link_past_a_shared_stem(
    make_vault: MakeVault,
) -> None:
    # the neighbour is reached by exact path; a bare [[globex]] here would be
    # ambiguous, resolve to nothing, and expand to nothing (the test above is
    # the bare-stem half of this)
    _, db = await indexed(
        make_vault,
        {
            "notes/acme-renewal.md": "---\ntype: customer\n---\n# Acme renewal\n\nThe Acme "
            "renewal closes in March. Renewal pricing is agreed; Acme signs the order form. "
            "See [[vendors/globex]].\n",
            "vendors/globex.md": EVAL_VAULT["notes/globex.md"],
            "customers/globex.md": "# Globex the customer\n\n"
            "A different account with the same stem.\n",
        },
    )
    expanded = await hybrid_search(
        db, EMBEDDER, "acme renewal march", SearchOptions(n=1, expand_links=True)
    )
    assert paths(expanded) == ["notes/acme-renewal.md", "vendors/globex.md"]
    assert expanded[1].expansion is True
    db.close()


async def test_vault_search_prints_ranked_hits(
    make_vault: MakeVault, capsys: pytest.CaptureFixture[str]
) -> None:
    root, db = await indexed(make_vault, EVAL_VAULT)
    db.close()
    r = str(root)
    # --lexical throughout: it is the embedder `indexed` built these vectors
    # with, and it keeps the suite off any provider
    assert await main(["search", "INV-2031", "--lexical", "--vault", r]) == 0
    assert "ledger/invoice-2031.md" in capsys.readouterr().out

    # the flag and its value are consumed wherever they sit, and never become
    # query words: this must find the same note as the line above
    assert await main(["search", "--vault", r, "--lexical", "INV-2031"]) == 0
    assert "ledger/invoice-2031.md" in capsys.readouterr().out

    assert await main(["search", "🙂", "--lexical", "--vault", r]) == 0  # no signal either side
    assert capsys.readouterr().out == "no matches\n"

    # an unindexed vault is not the same answer as "nothing matched"
    empty_root, empty = await indexed(make_vault, {})
    empty.close()
    assert await main(["search", "acme", "--lexical", "--vault", str(empty_root)]) == 0
    assert re.search(r"not indexed.*reindex", capsys.readouterr().out)

    assert await main(["search", "--lexical", "--vault", r]) == 1  # no query at all
    capsys.readouterr()
    # an error is a message, not a stack trace
    assert await main(["search", "acme", "--sideways", "--lexical", "--vault", r]) == 1
    assert "--sideways" in capsys.readouterr().err


class _WedgedFts:
    """A connection whose FTS MATCH always fails: the quoting should make this
    unreachable, and the fallback is what keeps a tokenizer/Unicode skew from
    taking the whole search down."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    def execute(self, sql: str, params: object = ()) -> sqlite3.Cursor:
        if "notes_fts match ?" in sql:
            raise sqlite3.OperationalError('fts5: syntax error near "x"')
        return self._db.execute(sql, params)  # type: ignore[arg-type]


async def test_an_fts5_error_drops_the_keyword_signal_instead_of_failing(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    wedged = _WedgedFts(db)
    hits = await hybrid_search(wedged, EMBEDDER, "acme renewal pricing")  # type: ignore[arg-type]
    assert len(hits) > 0
    assert all(h.fts_rank is None for h in hits)
    db.close()


async def test_a_token_less_note_has_no_vector_row_and_stays_findable(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(
        make_vault,
        {"notes/cjk.md": "# 圏点\n\n日本語 🙂\n", "notes/globex.md": EVAL_VAULT["notes/globex.md"]},
    )
    assert db.execute("select count(*) as c from vectors").fetchone()["c"] == 1
    # FTS-only note, and a query the embedder tokenizes to nothing: no vector
    # signal on either end, no NaN distances, still a hit
    hits = await hybrid_search(db, EMBEDDER, "日本語")
    assert paths(hits) == ["notes/cjk.md"]
    assert (hits[0].vec_rank, hits[0].fts_rank) == (None, 1)
    db.close()


async def test_n_caps_the_result_and_an_empty_vault_returns_nothing(
    make_vault: MakeVault,
) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    assert len(await hybrid_search(db, EMBEDDER, "acme", SearchOptions(n=2))) == 2
    with pytest.raises(VaultError, match="positive integer"):
        await hybrid_search(db, EMBEDDER, "acme", SearchOptions(n=0))
    db.close()

    _, empty = await indexed(make_vault, {})
    assert await hybrid_search(empty, EMBEDDER, "acme") == []
    empty.close()


async def test_searching_with_a_different_embedder_is_refused(make_vault: MakeVault) -> None:
    _, db = await indexed(make_vault, EVAL_VAULT)
    mismatch = re.compile(r"token-overlap-v1.*other-model.*doctor", re.S)
    with pytest.raises(VaultError, match=mismatch):
        await hybrid_search(db, stub_embedder("other-model", 256), "acme")
    db.close()

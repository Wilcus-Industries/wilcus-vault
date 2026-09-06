import math
import sqlite3

import pytest
from conftest import MakeVault

from wilcus_vault.db import db_path, open_db, reset_vectors, to_blob, vectors_stale
from wilcus_vault.embed import TokenOverlapEmbedder, Vector, l2_normalize


def _count(db: sqlite3.Connection, table: str) -> int:
    return int(db.execute(f"select count(*) from {table}").fetchone()[0])


def _vectors_sql(db: sqlite3.Connection) -> str:
    return str(db.execute("select sql from sqlite_master where name='vectors'").fetchone()[0])


def test_open_db_creates_index_in_wal_with_busy_timeout_and_vec(make_vault: MakeVault) -> None:
    root = make_vault({})
    db = open_db(db_path(root))
    assert (root / ".vault" / "index.db").exists()
    assert db.execute("pragma journal_mode").fetchone()[0] == "wal"
    assert db.execute("pragma busy_timeout").fetchone()[0] == 5000
    assert db.execute("select vec_version()").fetchone()[0].startswith("v0.1.7")
    db.close()


def test_schema_tables_and_unique_edges(make_vault: MakeVault) -> None:
    db = open_db(db_path(make_vault({})))
    names = [
        r["name"]
        for r in db.execute("select name from sqlite_master where type in ('table','view')")
    ]
    for name in ("notes", "edges", "notes_fts", "vector_meta"):
        assert name in names
    assert "vectors" not in names  # created lazily, from the embedder's dims

    db.execute(
        "insert into notes (path, slug, title, hash, frontmatter, mtime) "
        "values ('a.md','a','A','h','{}',0)"
    )
    db.execute("insert into edges (from_id, to_slug) values (1, 'b')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("insert into edges (from_id, to_slug) values (1, 'b')")
    db.close()


def test_vectors_table_created_lazily_with_dims_and_cosine(make_vault: MakeVault) -> None:
    db = open_db(db_path(make_vault({})))
    e = TokenOverlapEmbedder(8)
    assert vectors_stale(db, e) is False
    reset_vectors(db, e, False)
    sql = _vectors_sql(db)
    assert "float[8]" in sql
    assert "distance_metric=cosine" in sql
    db.close()


def test_model_or_dims_change_drops_vec0_table_and_meta(make_vault: MakeVault) -> None:
    db = open_db(db_path(make_vault({})))
    reset_vectors(db, TokenOverlapEmbedder(8), False)
    db.execute("insert into vectors (note_id, emb) values (1, ?)", (to_blob([0.25] * 8),))
    db.execute("insert into vector_meta (note_id, model, dims) values (1, 'token-overlap-v1', 8)")

    # same embedder: left alone
    same = TokenOverlapEmbedder(8)
    assert vectors_stale(db, same) is False
    reset_vectors(db, same, False)
    assert _count(db, "vectors") == 1

    # different dims: reported stale, and *only* reported, so a caller that goes
    # on to fail before it has replacements still has the vectors it started with
    wider = TokenOverlapEmbedder(16)
    assert vectors_stale(db, wider) is True
    assert _count(db, "vectors") == 1

    # acting on it is the separate, destructive step
    reset_vectors(db, wider, True)
    assert _count(db, "vectors") == 0
    assert _count(db, "vector_meta") == 0
    assert "float[16]" in _vectors_sql(db)
    db.close()


def _cos(x: Vector, y: Vector) -> float:
    return sum(a * b for a, b in zip(l2_normalize(x), y, strict=True))


async def test_token_overlap_embedder_deterministic_sized_overlap_sensitive() -> None:
    e = TokenOverlapEmbedder(64)
    assert e.dims == 64
    a, b, c = await e.embed(["acme corp invoice", "acme corp invoice", "zebra habitat"])
    assert len(a) == 64
    assert a == b
    assert _cos(a, l2_normalize(b)) == pytest.approx(1, abs=1e-5)
    assert _cos(a, l2_normalize(c)) < 0.5


def test_l2_normalize_unit_length_and_zero_vector() -> None:
    v = l2_normalize([3, 4])
    assert v[0] == pytest.approx(0.6, abs=1e-6)
    assert v[1] == pytest.approx(0.8, abs=1e-6)
    assert math.hypot(*v) == pytest.approx(1)
    assert l2_normalize([0, 0]) == [0, 0]


def test_transaction_takes_the_write_lock_up_front(make_vault: MakeVault) -> None:
    """A deferred `begin` defers the conflict to the first write, where SQLite
    refuses the upgrade outright rather than waiting out `busy_timeout`. Taking
    the lock at the top makes a second writer queue instead of fail."""
    from wilcus_vault.db import transaction

    path = db_path(make_vault({}))
    a, b = open_db(path), open_db(path)
    b.execute("pragma busy_timeout = 0")
    try:
        with transaction(a):
            a.execute(
                "insert into notes (path, slug, title, hash, frontmatter, mtime)"
                " values ('a.md', 'a', 'A', 'h', '{}', 0)"
            )
            # the begin itself must be what is refused, not a later write
            with pytest.raises(sqlite3.OperationalError, match="locked"), transaction(b):
                pass
    finally:
        a.close()
        b.close()

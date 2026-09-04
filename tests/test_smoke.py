"""Bootstrap research: sqlite-vec loads, FTS5 works, RRF fusion across both is one
SQL statement, and YAML parses frontmatter the way the vault needs.

The corpus is built so each side of the FULL OUTER JOIN has rows the other lacks
(doc 2: vector-only; docs 3/4: FTS-only). A LEFT JOIN would fail this.
"""

import sqlite3

import sqlite_vec
import yaml

from wilcus_vault.db import to_blob
from wilcus_vault.note import _Loader


def test_sqlite_vec_knn_fts5_rrf_hybrid_in_one_query() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    (vec_version,) = db.execute("select vec_version()").fetchone()
    assert vec_version == "v0.1.7-alpha.2"  # pinned pre-1.0

    db.execute("create virtual table v using vec0(id integer primary key, emb float[4])")
    db.execute("create virtual table f using fts5(id unindexed, body)")
    docs: list[tuple[int, list[float], str]] = [
        (1, [1, 0, 0, 0], "acme corp invoice INV-001"),
        (2, [0.95, 0.05, 0, 0], "globex vendor policy"),
        (3, [0.9, 0.1, 0, 0], "acme renewal decision meeting notes with extra detail words"),
        (4, [0, 0, 1, 0], "acme acme acme"),
    ]
    for doc_id, emb, body in docs:
        db.execute("insert into v (id, emb) values (?, ?)", (doc_id, to_blob(emb)))
        db.execute("insert into f (id, body) values (?, ?)", (doc_id, body))

    # RRF: k=60. KNN top-2 = {1, 2}; FTS "acme" = {1, 3, 4}: asymmetric by construction.
    rows = db.execute(
        """with vecq as (
             select id, row_number() over (order by distance) as r
             from v where emb match ? and k = 2
           ),
           ftsq as (
             select id, row_number() over (order by rank) as r
             from f where f match ? limit 3
           )
           select coalesce(vecq.id, ftsq.id) as id,
                  coalesce(1.0/(60+vecq.r),0) + coalesce(1.0/(60+ftsq.r),0) as score
           from vecq full outer join ftsq on vecq.id = ftsq.id
           order by score desc""",
        (to_blob([1, 0, 0, 0]), "acme"),
    ).fetchall()

    ids = [r[0] for r in rows]
    assert len(ids) == 4  # union of both sides
    # Doc 1 is the only doc on BOTH signals: worst case 1/61+1/63 ≈ .0323 beats
    # any single-signal doc's best case 1/61 ≈ .0164, a strict winner.
    assert rows[0][0] == 1
    assert rows[0][1] > rows[1][1] * 1.5
    assert 2 in ids  # vector-only side survived the outer join
    assert 4 in ids  # FTS-only side survived the outer join
    db.close()


def test_yaml_parses_frontmatter() -> None:
    fm = yaml.load("type: customer\ntags: [a, b]\nsuperseded_by: null", Loader=_Loader)
    assert fm == {"type": "customer", "tags": ["a", "b"], "superseded_by": None}
    # Dates stay strings: the vault never wants a datetime out of frontmatter.
    assert yaml.load("created: 2026-01-02", Loader=_Loader) == {"created": "2026-01-02"}

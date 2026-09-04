"""Wikilink resolution: bare stems against the note set, qualified targets by exact path."""

from conftest import MakeVault
from indexer_common import embedder, id_of, one, open_index

from wilcus_vault.indexer import reindex


async def test_reindex_re_resolves_every_edge_even_when_no_file_changed(
    make_vault: MakeVault,
) -> None:
    # An index written under an older resolution rule holds `to_id` values that
    # rule produced, and nothing on disk has to change for them to be wrong:
    # `[[one/dup]]` was unresolvable before this rule and is exact under it. The
    # whole-vault pass is the only place that can notice.
    root = make_vault({"hub.md": "# Hub\n\n[[one/dup]]\n", "one/dup.md": "# Dup one\n"})
    db = open_index(root)
    await reindex(db, root, embedder)
    db.execute("update edges set to_id = null")  # an index built before the rule
    stats = await reindex(db, root, embedder)
    assert (stats.added, stats.updated, stats.unchanged) == (0, 0, 2)
    assert one(db, "select n.path from edges e join notes n on n.id = e.to_id") == "one/dup.md"
    db.close()


async def test_bare_stem_zero_or_several_matches_leave_to_id_null(make_vault: MakeVault) -> None:
    root = make_vault(
        {
            "a.md": "# A\n\n[[dup]] and [[nowhere]]\n",
            "one/dup.md": "# Dup one\n",
            "two/dup.md": "# Dup two\n",
        }
    )
    db = open_index(root)
    await reindex(db, root, embedder)
    assert one(db, "select count(*) from edges where to_id is not null") == 0

    # removing the duplicate makes the link resolvable on the next pass, even
    # though the linking note itself never changed
    (root / "two" / "dup.md").unlink()
    await reindex(db, root, embedder)
    resolved = one(
        db, "select n.path from edges e join notes n on n.id = e.to_id where e.to_slug='dup'"
    )
    assert resolved == "one/dup.md"
    assert one(db, "select to_id from edges where to_slug='nowhere'") is None
    db.close()


async def test_path_qualified_links_resolve_by_exact_path_past_a_duplicated_stem(
    make_vault: MakeVault,
) -> None:
    root = make_vault(
        {
            "hub.md": "# Hub\n\n[[customers/acme]], [[vendors/acme]], [[acme]], "
            "[[customers/ghost]], [[../outside/secret]]\n",
            "customers/acme.md": "# Acme the customer\n",
            "vendors/acme.md": "# Acme the vendor\n",
        }
    )
    db = open_index(root)
    await reindex(db, root, embedder)
    hub = id_of(db, "hub.md")

    edges = db.execute(
        "select to_slug, to_id from edges where from_id = ? order by to_slug", (hub,)
    ).fetchall()
    assert [tuple(e) for e in edges] == [
        # never path-joined to the filesystem: resolution is SQL against known
        # note paths, so a traversing target simply matches nothing
        ("../outside/secret", None),
        ("acme", None),  # bare stem, two candidates ⇒ unresolved
        ("customers/acme", id_of(db, "customers/acme.md")),
        ("customers/ghost", None),  # qualified, but no such note
        ("vendors/acme", id_of(db, "vendors/acme.md")),
    ]

    # the bare stem resolves again once only one note carries it, and the
    # qualified links are untouched by that
    (root / "vendors" / "acme.md").unlink()
    await reindex(db, root, embedder)
    by_slug = "select to_id from edges where from_id = ? and to_slug = ?"
    assert one(db, by_slug, hub, "acme") == id_of(db, "customers/acme.md")
    assert one(db, by_slug, hub, "vendors/acme") is None
    db.close()


async def test_a_qualified_link_is_a_path_not_a_stem(make_vault: MakeVault) -> None:
    root = make_vault(
        {
            "hub.md": "# Hub\n\n[[customers/acme.md]] [[/customers/acme]] [[./customers/acme]] "
            "[[acme]]\n",
            "customers/acme.md": "# Acme\n",
        }
    )
    db = open_index(root)
    await reindex(db, root, embedder)
    # only the bare stem resolves here: the qualified forms are matched against
    # `notes.path` minus its `.md`, exactly, with no normalization
    rows = db.execute("select to_slug from edges where to_id is not null").fetchall()
    assert [r["to_slug"] for r in rows] == ["acme"]
    db.close()

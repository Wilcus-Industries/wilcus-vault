"""Re-embedding on a model or dims change, and notes that get no vector at all."""

import pytest
from conftest import MakeVault
from fakes import stub_embedder
from indexer_common import FIXTURE, embedder, id_of, one, open_index

from wilcus_vault.embed import TokenOverlapEmbedder, Vector
from wilcus_vault.indexer import index_paths, reindex, scan_vault


async def test_an_embedding_model_change_re_embeds_every_note(make_vault: MakeVault) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    await reindex(db, root, embedder)
    stats = await reindex(db, root, TokenOverlapEmbedder(48))
    assert (stats.updated, stats.unchanged, stats.reembedded) == (3, 0, True)
    assert one(db, "select count(*) from vectors") == 3
    assert one(db, "select count(*) from vector_meta where dims = 48") == 3
    db.close()


async def test_a_new_model_at_the_same_dims_still_re_embeds_everything(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    await reindex(db, root, embedder)
    stats = await reindex(db, root, stub_embedder("other-model-v1", 32))
    assert (stats.updated, stats.unchanged, stats.reembedded) == (3, 0, True)
    assert one(db, "select count(*) from vector_meta where model='other-model-v1'") == 3
    assert one(db, "select count(*) from vectors") == 3
    db.close()


async def test_a_model_swap_that_fails_to_embed_leaves_the_old_vectors(
    make_vault: MakeVault,
) -> None:
    root = make_vault(FIXTURE)
    db = open_index(root)
    await reindex(db, root, embedder)
    stored = "select note_id, emb from vectors order by note_id"
    before = [tuple(r) for r in db.execute(stored)]

    # The swap's replacement vectors never arrive: an Ollama that is not
    # running, which `--lexical` <-> default switching makes an ordinary event.
    async def down(texts: list[str]) -> list[Vector]:
        raise RuntimeError("no embedder configured: start Ollama")

    with pytest.raises(RuntimeError, match="no embedder configured"):
        await reindex(db, root, stub_embedder("other-model-v1", 32, down))
    # Nothing may have been thrown away for a re-embed that did not happen: an
    # emptied vectors table with emptied meta is a vault that searches on FTS
    # alone and says nothing about it.
    assert [tuple(r) for r in db.execute(stored)] == before
    assert one(db, "select count(*) from vector_meta where model = ?", embedder.model) == 3

    # and the swap still works once the embedder does
    stats = await reindex(db, root, stub_embedder("other-model-v1", 32))
    assert (stats.updated, stats.unchanged, stats.reembedded) == (3, 0, True)
    assert one(db, "select count(*) from vector_meta where model='other-model-v1'") == 3
    db.close()


async def test_a_note_with_no_tokens_gets_no_vector_row_and_stays_idempotent(
    make_vault: MakeVault,
) -> None:
    root = make_vault(
        {"cjk.md": "# 圏点\n\n日本語 🙂\n", "notes/acme.md": FIXTURE["notes/acme.md"]}
    )
    db = open_index(root)
    assert (await reindex(db, root, embedder)).added == 2
    note_id = id_of(db, "cjk.md")
    # an all-zero vector has no direction: cosine distance against it is NaN
    assert one(db, "select count(*) from vectors where note_id = ?", note_id) == 0
    assert one(db, "select count(*) from vector_meta where note_id = ?", note_id) == 1
    assert one(db, "select count(*) from notes_fts where rowid = ?", note_id) == 1
    # must not look half-indexed on the next pass (measured on `index_paths`:
    # `reindex` re-resolves edges on every run by design)
    before = db.total_changes
    stats = await index_paths(db, root, embedder, scan_vault(root))
    assert (stats.unchanged, stats.updated) == (2, 0)
    assert db.total_changes == before
    db.close()

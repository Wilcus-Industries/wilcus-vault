"""Doctor and the embedder: failed rebuilds, races, and model or dims changes."""

import os
import re

import pytest
from conftest import MakeVault
from fakes import stub_embedder
from test_doctor import GRAPH, embedder, snapshot

from wilcus_vault.db import db_path, open_db
from wilcus_vault.doctor import DoctorOptions, doctor
from wilcus_vault.embed import TokenOverlapEmbedder, Vector
from wilcus_vault.term import VaultError
from wilcus_vault.vault import open as open_vault


async def test_a_failed_rebuild_leaves_a_stale_index_the_next_run_finishes(
    make_vault: MakeVault,
) -> None:
    """The rebuild works in the live file, so a failure part-way through leaves a
    partly filled index rather than the old one. That is the price of never
    replacing the inode: stale is recoverable from the files, stranded is not."""
    root = make_vault(GRAPH)
    await doctor(root, embedder)

    async def down(texts: list[str]) -> list[Vector]:
        raise RuntimeError("provider down")

    boom = stub_embedder("boom-v1", 32, down)
    with pytest.raises(RuntimeError, match="provider down"):
        await doctor(root, boom, DoctorOptions(rebuild=True))
    assert os.listdir(root / ".vault") == ["index.db"]  # no temp database either way
    assert await doctor(root, embedder, DoctorOptions(rebuild=True)) is not None
    assert len(snapshot(root)[0]) == 5  # the next run rebuilds it from the files


async def test_survives_a_file_deleted_mid_run(make_vault: MakeVault) -> None:
    root = make_vault(GRAPH)

    async def racy_embed(texts: list[str]) -> list[Vector]:
        (root / "notes" / "lonely.md").unlink(missing_ok=True)
        return [[1.0] * 32 for _ in texts]

    racy = stub_embedder("race-v1", 32, racy_embed)
    await doctor(root, racy)  # must not raise
    assert (await doctor(root, racy)).missing == ["notes/lonely.md"]
    assert (await doctor(root, racy, DoctorOptions(repair=False))).missing == []


async def test_model_or_dims_change_drops_vec0_and_reembeds(make_vault: MakeVault) -> None:
    root = make_vault(GRAPH)
    await doctor(root, embedder)
    report = await doctor(root, TokenOverlapEmbedder(64))
    assert report.reembedded is True

    db = open_db(db_path(root))
    assert db.execute("select count(*) from vector_meta where dims = 64").fetchone()[0] == 5
    ddl = db.execute("select sql from sqlite_master where name='vectors'").fetchone()["sql"]
    assert "float[64]" in ddl
    db.close()


async def test_reopen_with_another_embedder_doctor_reembeds_and_search_works(
    make_vault: MakeVault,
) -> None:
    root = make_vault(GRAPH)
    before = open_vault(root, embedder)
    await before.reindex()
    assert "notes/globex.md" in [h.path for h in await before.search("globex vendor")]
    before.close()

    # a different model *and* width: the vec0 table's dims live in its DDL
    swapped = stub_embedder("swapped-v1", 64, TokenOverlapEmbedder(64).embed)
    after = open_vault(root, swapped)
    # vectors from another model are not comparable, so search refuses until doctor runs
    with pytest.raises(VaultError, match=re.escape("run vault doctor")):
        await after.search("globex")

    assert (await after.doctor()).reembedded is True
    db = open_db(db_path(root))
    ddl = db.execute("select sql from sqlite_master where name='vectors'").fetchone()["sql"]
    assert "float[64]" in ddl
    assert db.execute("select count(*) from vectors").fetchone()[0] == 5
    by_model = "select count(*) from vector_meta where model='swapped-v1' and dims=64"
    assert db.execute(by_model).fetchone()[0] == 5
    assert db.execute("select count(*) from vector_meta where dims<>64").fetchone()[0] == 0
    db.close()

    # and the vault is usable again through the new embedder
    assert "notes/globex.md" in [h.path for h in await after.search("globex vendor")]
    after.close()

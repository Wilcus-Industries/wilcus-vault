"""Shared fixtures. Fixture vaults live under tmp-test/ (pytest's basetemp, gitignored)."""

import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from wilcus_vault.db import from_blob
from wilcus_vault.embed import TokenOverlapEmbedder, Vector

# The whole provider configuration a developer might have lying around. Cleared
# for every test, so a real key or endpoint can never decide one.
ENV_NAMES = (
    "VAULT_EMBED_API_KEY",
    "VAULT_EMBED_ENDPOINT",
    "VAULT_EMBED_MODEL",
    "VAULT_EMBED_DIMS",
    "VAULT_DECIDE_API_KEY",
    "VAULT_DECIDE_ENDPOINT",
    "VAULT_DECIDE_MODEL",
)


def write_note(root: str | Path, rel: str, body: str) -> None:
    abs_path = Path(root) / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(body, encoding="utf-8")


def vec_of(db: sqlite3.Connection, note_id: int) -> Vector:
    """A note's stored vector, as floats."""
    row = db.execute("select emb from vectors where note_id = ?", (note_id,)).fetchone()
    return from_blob(row["emb"])


MakeVault = Callable[[dict[str, str]], Path]


@pytest.fixture
def make_vault(tmp_path: Path) -> MakeVault:
    """A fresh vault directory holding exactly `files` (vault-relative path -> text)."""

    def make(files: dict[str, str]) -> Path:
        root = tmp_path / f"v-{uuid.uuid4().hex[:8]}"
        root.mkdir()
        for rel, body in files.items():
            write_note(root, rel, body)
        return root

    return make


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def embedder() -> TokenOverlapEmbedder:
    return TokenOverlapEmbedder()

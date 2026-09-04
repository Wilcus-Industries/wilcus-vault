"""The corpus and helpers the write-gate tests share."""

from pathlib import Path
from typing import Any

from wilcus_vault.decision import Candidate, Decider
from wilcus_vault.embed import TokenOverlapEmbedder
from wilcus_vault.gate import GateOptions
from wilcus_vault.note import parse_note
from wilcus_vault.scope import VaultContext
from wilcus_vault.search_sql import Cutoffs
from wilcus_vault.vault import Vault, open

EMBEDDER = TokenOverlapEmbedder()
# Loose enough that a related note survives; the point is that they are passed.
CUTOFFS = Cutoffs(distance_ceiling=0.9, bm25_ceiling=0)
NOTHING_SIMILAR = Cutoffs(distance_ceiling=0.5, bm25_ceiling=-1)

OLD_NOTE = """---
type: customer
id: 01234 # legacy account number, must survive a gate write
rate: 1.0
updated: 2020-01-01
---
# Acme renewal

The Acme renewal closes in March. See [[support-rota]].
"""

VAULT = {
    "notes/acme-renewal.md": OLD_NOTE,
    "notes/support-rota.md": "# Support rota\n\nWho carries the pager each week.\n",
}

CANDIDATE = Candidate(
    title="Acme renewal 2026",
    body="The Acme renewal closes in March 2026 at the agreed renewal pricing.\n",
    type="customer",
    namespace="notes",
)

CTX = VaultContext(agent="core/scheduler", source="task-42")


def read(root: str | Path, rel: str) -> str:
    return (Path(root) / rel).read_text(encoding="utf-8")


def fm(root: str | Path, rel: str) -> dict[str, Any]:
    """A note's frontmatter as the parser reads it: `create` serializes, `update`
    patches textually, and provenance has to arrive either way."""
    return parse_note(read(root, rel), rel).frontmatter


async def open_gate(root: Path, decider: Decider, cutoffs: Cutoffs = CUTOFFS) -> Vault:
    v = open(root, EMBEDDER, gate=GateOptions(decider=decider, cutoffs=cutoffs))
    await v.reindex()
    return v

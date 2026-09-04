"""Shared setup for the consolidation evals: an exact-vector embedder, a fake
merger, and the three-note corpus the linkage rule is judged on."""

from pathlib import Path
from typing import Any

from conftest import MakeVault
from fakes import stub_embedder

from wilcus_vault import ConsolidateOptions, MergedNote, MergeInput, Merger, Vault, open
from wilcus_vault.embed import Vector
from wilcus_vault.note import parse_note
from wilcus_vault.scope import VaultContext

# Exact vectors by marker word, so the distances are arithmetic rather than
# whatever a lexical embedder happens to produce: d(ALPHA,BETA) = 0.10,
# d(BETA,GAMMA) = 0.20, d(ALPHA,GAMMA) = 0.54, the chain single linkage would
# collapse and complete linkage must not. DELTA and EPSILON are the other two
# axes, a full 1.0 from ALPHA and from each other; EPSILON is also where text
# carrying no marker at all lands.
MARKERS: dict[str, Vector] = {
    "ALPHA": [1, 0, 0],
    "BETA": [0.9, 0.43589, 0],
    "GAMMA": [0.45847, 0.88871, 0],
    "DELTA": [0, 1, 0],
    "EPSILON": [0, 0, 1],
}


async def marker_embed(texts: list[str]) -> list[Vector]:
    return [next((list(v) for m, v in MARKERS.items() if m in t), [0, 0, 1]) for t in texts]


embedder = stub_embedder("marker-v1", 3, marker_embed)

NOTES = {
    "notes/alpha.md": "# Alpha\n\nALPHA marks the first note.\n",
    "notes/beta.md": "# Beta\n\nBETA marks its near neighbour.\n",
    "notes/gamma.md": "# Gamma\n\nGAMMA marks the far one.\n",
}

# Three pairs on three axes: every pair is identical to itself and a full 1.0
# from the other two, so the ceiling finds exactly three clusters.
PAIRS = {
    "notes/a1.md": "# A one\n\nALPHA one.\n",
    "notes/a2.md": "# A two\n\nALPHA two.\n",
    "notes/d1.md": "# D one\n\nDELTA one.\n",
    "notes/d2.md": "# D two\n\nDELTA two.\n",
    "notes/e1.md": "# E one\n\nEPSILON one.\n",
    "notes/e2.md": "# E two\n\nEPSILON two.\n",
}

CTX = VaultContext(agent="core/librarian", source="task-9")


def fake_merger() -> tuple[Merger, list[MergeInput]]:
    """A merger that answers with one fixed note, and records what it was shown."""
    seen: list[MergeInput] = []

    async def merger(input: MergeInput) -> MergedNote:
        seen.append(input)
        return MergedNote("Merged note", "ALPHA and BETA in one note.\n", "customer")

    return merger, seen


def open_vault(make_vault: MakeVault, files: dict[str, str], merger: Merger) -> Vault:
    """No reindex here on purpose: the pass indexes before it discovers, so a
    vault whose index is empty or stale still consolidates against the files."""
    return open(make_vault(files), embedder, consolidate=ConsolidateOptions(merger))


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def fm(root: Path, rel: str) -> dict[str, Any]:
    return parse_note(read(root, rel), rel).frontmatter

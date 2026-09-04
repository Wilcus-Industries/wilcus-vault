"""Files-are-truth markdown memory vault for AI agents."""

from .cluster import Cluster, clusters
from .consolidate import (
    ConsolidateOptions,
    ConsolidateReport,
    ConsolidateRun,
    Merge,
    MergeError,
)
from .decide import fetch_decider
from .decision import (
    Action,
    Candidate,
    Decider,
    DeciderInput,
    Decision,
    SimilarNote,
    gate_prompt,
    parse_decision,
)
from .discards import (
    DiscardedSimilar,
    DiscardEntry,
    count_discards,
    get_discard,
    list_discards,
    restore_discard,
)
from .doctor import AmbiguousLink, DoctorOptions, DoctorReport, LinkProblem
from .embed import Embedder, TokenOverlapEmbedder, Vector
from .fetch_embedder import FetchEmbedder
from .gate import GateOptions
from .gate_write import GateResult
from .indexer import IndexStats
from .merge import MergedNote, MergeInput, Merger, merge_prompt, parse_merged
from .note import Note
from .paths import slugify
from .qualify import Qualified
from .scope import Permission, ScopePolicy, ScopeRule, VaultContext
from .search import SearchOptions
from .search_sql import Cutoffs, SearchHit
from .term import VaultError
from .vault import Vault, open
from .watch import Watcher, WatchOptions

__all__ = [
    "Action",
    "AmbiguousLink",
    "Candidate",
    "Cluster",
    "ConsolidateOptions",
    "ConsolidateReport",
    "ConsolidateRun",
    "Cutoffs",
    "Decider",
    "DeciderInput",
    "Decision",
    "DiscardEntry",
    "DiscardedSimilar",
    "DoctorOptions",
    "DoctorReport",
    "Embedder",
    "FetchEmbedder",
    "GateOptions",
    "GateResult",
    "IndexStats",
    "LinkProblem",
    "Merge",
    "MergeError",
    "MergeInput",
    "MergedNote",
    "Merger",
    "Note",
    "Permission",
    "Qualified",
    "ScopePolicy",
    "ScopeRule",
    "SearchHit",
    "SearchOptions",
    "SimilarNote",
    "TokenOverlapEmbedder",
    "Vault",
    "VaultContext",
    "VaultError",
    "Vector",
    "Watcher",
    "WatchOptions",
    "clusters",
    "count_discards",
    "fetch_decider",
    "gate_prompt",
    "get_discard",
    "list_discards",
    "merge_prompt",
    "open",
    "parse_decision",
    "parse_merged",
    "restore_discard",
    "slugify",
]

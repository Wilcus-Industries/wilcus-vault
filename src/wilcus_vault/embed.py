"""The Embedder contract and the deterministic embedder used by tests and evals."""

import math
import re
from typing import Protocol

Vector = list[float]

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model: str  # identity of the vector space; a change forces a full re-embed
    dims: int

    async def embed(self, texts: list[str]) -> list[Vector]: ...


def l2_normalize(v: Vector) -> Vector:
    """Scale to unit length so cosine distance is well-defined whatever the provider returns."""
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else list(v)


def fnv1a(s: str) -> int:
    """FNV-1a, 32-bit. Fixed by spec, so stored vectors never shift."""
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class TokenOverlapEmbedder:
    """Bag of tokens hashed into `dims` buckets. Lexical only, deterministic everywhere."""

    model = "token-overlap-v1"

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    async def embed(self, texts: list[str]) -> list[Vector]:
        out = []
        for text in texts:
            v = [0.0] * self.dims
            for token in _TOKEN.findall(text.lower()):
                v[fnv1a(token) % self.dims] += 1
            out.append(v)
        return out

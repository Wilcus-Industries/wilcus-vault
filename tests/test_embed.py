import math

import pytest

from wilcus_vault.embed import TokenOverlapEmbedder, l2_normalize


def test_l2_normalize_leaves_a_zero_vector_alone() -> None:
    # it has no direction
    assert l2_normalize([0, 0, 0]) == [0, 0, 0]
    assert math.hypot(*l2_normalize([3, 4])) == pytest.approx(1, abs=1e-6)


async def test_token_overlap_embedder_stays_deterministic_across_instances() -> None:
    [a] = await TokenOverlapEmbedder(16).embed(["Acme renewal"])
    [b] = await TokenOverlapEmbedder(16).embed(["acme   RENEWAL!"])
    assert a == b

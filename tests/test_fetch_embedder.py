"""FetchEmbedder over the wire: request shape, batching, redaction, response checks."""

import json
from typing import cast
from urllib.error import URLError

import pytest
from fakes import StubTransport, stub_transport

from wilcus_vault.fetch_embedder import FetchEmbedder
from wilcus_vault.http import Transient
from wilcus_vault.term import VaultError

KEY = "sk-test-do-not-log-me"


async def test_posts_an_openai_shaped_request_and_returns_vectors_in_input_order() -> None:
    t = stub_transport(4)
    e = FetchEmbedder(
        api_key=KEY, dims=4, model="m-1", endpoint="https://x/v1/embeddings", transport=t
    )
    assert e.model == "m-1"
    assert e.dims == 4

    out = await e.embed(["a", "bb", "ccc"])
    assert out == [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    assert len(t.calls) == 1
    assert t.calls[0].url == "https://x/v1/embeddings"
    assert t.calls[0].body == {"model": "m-1", "input": ["a", "bb", "ccc"]}
    assert t.calls[0].headers["authorization"] == f"Bearer {KEY}"
    assert t.calls[0].timeout > 0  # timeout, not a hang


async def test_batches_by_count_and_by_request_size() -> None:
    t = stub_transport(2)
    e = FetchEmbedder(api_key=KEY, dims=2, transport=t, batch_size=2)
    assert len(await e.embed(["a", "b", "c", "d", "e"])) == 5
    assert [c.body["input"] for c in t.calls] == [["a", "b"], ["c", "d"], ["e"]]

    t.calls.clear()
    small = FetchEmbedder(api_key=KEY, dims=2, transport=t, batch_size=100, max_chars=6)
    # whole notes embed whole (no chunking), so a text over the budget goes alone
    assert len(await small.embed(["aaaa", "bb", "c" * 20, "d"])) == 4
    assert [c.body["input"] for c in t.calls] == [["aaaa", "bb"], ["c" * 20], ["d"]]


async def test_never_leaks_the_api_key_in_errors_or_when_inspected() -> None:
    t = stub_transport(4, lambda _: (401, f"bad key {KEY}"))
    e = FetchEmbedder(api_key=KEY, dims=4, transport=t)
    with pytest.raises(VaultError) as info:
        await e.embed(["x"])
    message = str(info.value)
    assert "401" in message
    assert KEY not in message
    assert "[redacted]" in message
    assert KEY not in repr(e)
    assert KEY not in str(e)


async def test_redacts_the_key_before_truncating_a_long_error_body() -> None:
    # the key straddles the 200-character cut: truncating first would leak its head
    t = stub_transport(4, lambda _: (500, "x" * 190 + KEY + "y" * 50))
    e = FetchEmbedder(api_key=KEY, dims=4, transport=t)
    with pytest.raises(VaultError) as info:
        await e.embed(["x"])
    assert "500" in str(info.value)
    assert KEY[:8] not in str(info.value)


async def test_rejects_vectors_that_are_not_all_finite_numbers() -> None:
    # JSON has no NaN literal, so a provider's NaN arrives as null
    def embedder(bad: str) -> FetchEmbedder:
        body = f'{{"data":[{{"index":0,"embedding":{bad}}}]}}'
        t = stub_transport(4, lambda _: (200, body))
        return FetchEmbedder(api_key=KEY, dims=4, model="m-1", transport=t)

    for bad in ['[1, "2", 3, 4]', "[1, null, 3, 4]", "[1, 2, 3, 1e999]"]:
        e = embedder(bad)
        # a non-finite element survives l2_normalize and would poison the vec0 index
        with pytest.raises(VaultError, match=r"(?s)m-1.*finite numbers"):
            await e.embed(["x"])


async def test_rejects_a_response_whose_indices_are_not_one_per_text() -> None:
    def embedder(body: str) -> FetchEmbedder:
        t = stub_transport(2, lambda _: (200, body))
        return FetchEmbedder(api_key=KEY, dims=2, model="m-1", transport=t)

    vec = '"embedding":[1,2]'
    # duplicated index: one text would silently get another text's vector
    for body in (
        f'{{"data":[{{"index":0,{vec}}},{{"index":0,{vec}}}]}}',
        f'{{"data":[{{"index":0,{vec}}},{{"index":7,{vec}}}]}}',
        f'{{"data":[{{{vec}}},{{{vec}}}]}}',
    ):
        with pytest.raises(VaultError, match=r"(?s)m-1.*index"):
            await embedder(body).embed(["a", "b"])


async def test_reports_a_non_json_body_instead_of_throwing_a_parse_error() -> None:
    t = stub_transport(4, lambda _: (200, f"<html>gateway timeout {KEY}</html>"))
    e = FetchEmbedder(api_key=KEY, dims=4, model="m-1", transport=t)
    with pytest.raises(VaultError, match=r"(?s)m-1.*non-JSON") as info:
        await e.embed(["x"])
    assert KEY not in str(info.value)


async def test_quotes_a_keyless_provider_error_body_verbatim() -> None:
    # redacting the empty string would splice [redacted] between every character
    t = stub_transport(384, lambda _: (404, "model 'all-minilm' not found"))
    e = FetchEmbedder(transport=t)
    with pytest.raises(VaultError, match="404 model 'all-minilm' not found"):
        await e.embed(["x"])


async def test_rejects_a_response_that_does_not_match_the_request() -> None:
    short = FetchEmbedder(
        api_key=KEY,
        dims=4,
        model="m-1",
        transport=stub_transport(4, lambda _: (200, json.dumps({"data": []}))),
    )
    with pytest.raises(VaultError, match=r"(?s)m-1.*0 vectors for 2"):
        await short.embed(["a", "b"])

    # 2-wide vectors for a 4-dim embedder
    narrow = FetchEmbedder(api_key=KEY, dims=4, model="m-1", transport=stub_transport(2))
    with pytest.raises(VaultError, match=r"(?s)m-1.*width 2.*expected 4"):
        await narrow.embed(["a"])


def _flaky(dims: int, failures: list[tuple[int, str]]) -> StubTransport:
    """A transport that serves `failures` in order, then embeds normally.
    A failure of status 0 is raised as a timeout rather than answered."""
    served = 0

    def reply(payload: dict[str, object]) -> tuple[int, str]:
        nonlocal served
        if served < len(failures):
            status, text = failures[served]
            served += 1
            if status == 0:
                raise TimeoutError(text)
            return status, text
        texts = cast(list[str], payload["input"])
        data = [{"index": i, "embedding": [float(len(t))] * dims} for i, t in enumerate(texts)]
        return 200, json.dumps({"data": data})

    return stub_transport(dims, reply)


async def test_a_transient_provider_failure_is_retried_not_thrown_away() -> None:
    """A rebuild re-embeds the whole vault in one pass; one 503 must not lose it."""
    t = _flaky(2, [(503, "upstream busy"), (429, "slow down")])
    e = FetchEmbedder(api_key=KEY, dims=2, transport=t, backoff=0)
    assert await e.embed(["a", "bb"]) == [[1.0, 1.0], [2.0, 2.0]]
    assert len(t.calls) == 3


async def test_a_timeout_is_retried_too() -> None:
    t = _flaky(2, [(0, "read timed out")])
    e = FetchEmbedder(api_key=KEY, dims=2, transport=t, backoff=0)
    assert await e.embed(["a"]) == [[1.0, 1.0]]
    assert len(t.calls) == 2


async def test_a_rejected_request_is_not_retried() -> None:
    """A bad key or an unknown model fails the same way however many times it is asked."""
    t = _flaky(2, [(401, "bad key"), (401, "bad key")])
    e = FetchEmbedder(api_key=KEY, dims=2, transport=t, backoff=0)
    with pytest.raises(VaultError):
        await e.embed(["a"])
    assert len(t.calls) == 1


async def test_the_retry_budget_is_finite_and_the_last_failure_is_the_one_raised() -> None:
    """Three 503s exhaust the default budget: the third is raised, not swallowed,
    and no fourth request is made."""
    t = _flaky(2, [(503, "one"), (503, "two"), (503, "the last one")])
    e = FetchEmbedder(api_key=KEY, dims=2, transport=t, backoff=0)
    with pytest.raises(Transient, match="the last one"):
        await e.embed(["a"])
    assert len(t.calls) == 3


async def test_a_dropped_connection_is_transient_but_a_refused_one_is_not() -> None:
    """The overload case and the nothing-is-listening case look alike and are not:
    a provider that accepted the connection and dropped it is worth asking again."""

    async def dropped(*_: object) -> tuple[int, str]:
        raise URLError(ConnectionResetError(104, "Connection reset by peer"))

    e = FetchEmbedder(api_key=KEY, dims=2, transport=dropped, backoff=0)
    with pytest.raises(Transient):
        await e.embed(["a"])

    async def refused(*_: object) -> tuple[int, str]:
        raise URLError(ConnectionRefusedError(111, "Connection refused"))

    local = FetchEmbedder(dims=2, transport=refused, backoff=0)
    with pytest.raises(VaultError, match="no embedder configured"):
        await local.embed(["a"])

"""Reusable fakes: an embedder that answers what a test wants, and a transport
that records calls and answers OpenAI-shaped JSON."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from wilcus_vault.embed import Vector

Embed = Callable[[list[str]], Awaitable[list[Vector]]]


class StubEmbedder:
    def __init__(self, model: str, dims: int, embed: Embed | None = None) -> None:
        self.model = model
        self.dims = dims
        self._embed = embed

    async def embed(self, texts: list[str]) -> list[Vector]:
        if self._embed is not None:
            return await self._embed(texts)
        return [[1.0] * self.dims for _ in texts]


def stub_embedder(model: str, dims: int, embed: Embed | None = None) -> StubEmbedder:
    return StubEmbedder(model, dims, embed)


@dataclass
class Call:
    url: str
    headers: dict[str, str]
    body: dict[str, object]
    timeout: float


# What a stub answers for one request: (status, response text), given the posted JSON.
Reply = Callable[[dict[str, object]], tuple[int, str]]


@dataclass
class StubTransport:
    dims: int
    reply: Reply | None = None
    calls: list[Call] = field(default_factory=list)

    async def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, str]:
        payload = json.loads(body)
        self.calls.append(Call(url, headers, payload, timeout))
        if self.reply is not None:
            return self.reply(payload)
        inputs = payload["input"]
        # Deliberately out of order: providers do not promise response order.
        data = [
            {"index": i, "embedding": [float(len(t))] * self.dims} for i, t in enumerate(inputs)
        ]
        return 200, json.dumps({"data": list(reversed(data))})


def stub_transport(dims: int, reply: Reply | None = None) -> StubTransport:
    return StubTransport(dims, reply)


def chat_reply(content: str) -> Reply:
    """A transport reply carrying one chat completion whose message is `content`."""
    return lambda _payload: (
        200,
        json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}),
    )

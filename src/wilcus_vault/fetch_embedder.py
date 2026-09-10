"""An embedder over an OpenAI-compatible `POST /v1/embeddings`.

Unconfigured it talks to a local Ollama and nothing leaves the machine. A remote
provider is an explicit choice, and one that must name its model and dims.
"""

import asyncio
import math
import os
import random
from http.client import RemoteDisconnected

from .embed import Vector
from .http import (
    Endpoint,
    Transient,
    Transport,
    default_transport,
    post_json,
    resolve_endpoint,
)
from .term import VaultError, printable

LOCAL_ENDPOINT = "http://localhost:11434/v1/embeddings"
LOCAL_MODEL = "all-minilm"
LOCAL_DIMS = 384
NO_EMBEDDER = (
    "no embedder configured: start Ollama (`ollama pull all-minilm`) or configure a remote provider"
)


class FetchEmbedder:
    def __init__(
        self,
        *,
        model: str | None = None,
        dims: int | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        max_chars: int = 96_000,
        timeout: float = 30.0,
        retries: int = 2,  # extra attempts per batch, for transient provider failures
        backoff: float = 0.5,  # seconds before the first retry, doubling after
        transport: Transport = default_transport,
    ) -> None:
        self._endpoint: Endpoint = resolve_endpoint(
            "FetchEmbedder", "VAULT_EMBED", LOCAL_ENDPOINT, endpoint, api_key
        )
        model = model if model is not None else os.environ.get("VAULT_EMBED_MODEL")
        if dims is None:
            dims = _env_dims()
        # The defaults describe the local model; a remote provider would be
        # posting note bodies under a model name it never heard of.
        missing = [name for name, value in (("model", model), ("dims", dims)) if value is None]
        if missing and not self._endpoint.local:
            env = "/".join(f"VAULT_EMBED_{m.upper()}" for m in missing)
            raise VaultError(
                f"FetchEmbedder: a remote endpoint needs {' and '.join(missing)} — "
                f"pass {'/'.join(missing)} or set {env}"
            )
        self.model = model if model is not None else LOCAL_MODEL
        self.dims = dims if dims is not None else LOCAL_DIMS
        if isinstance(self.dims, bool) or not isinstance(self.dims, int) or self.dims < 1:
            raise VaultError(f"FetchEmbedder: dims must be a positive integer, got {self.dims}")
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff
        self._transport = transport

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Embed in batches bounded by count and by characters; one long text goes alone."""
        out: list[Vector] = []
        batch: list[str] = []
        chars = 0
        for text in texts:
            full = len(batch) >= self._batch_size or chars + len(text) > self._max_chars
            if batch and full:
                out.extend(await self._post(batch))
                batch, chars = [], 0
            batch.append(text)
            chars += len(text)
        if batch:
            out.extend(await self._post(batch))
        return out

    async def _post(self, texts: list[str]) -> list[Vector]:
        """One batch, retried while the failure is the kind that asking again fixes.

        An index pass embeds every dirty note before it writes any of them, so a
        single 503 mid-rebuild otherwise rolls the whole pass back.
        """
        for attempt in range(self._retries):
            try:
                return self._vectors(await self._attempt(texts), texts)
            except (Transient, TimeoutError):
                # Jittered: a shared provider 503s every caller at once, and a fixed
                # delay would have them all come back in lockstep and do it again.
                await asyncio.sleep(self._backoff * 2**attempt * random.uniform(0.5, 1.5))
        return self._vectors(await self._attempt(texts), texts)

    async def _attempt(self, texts: list[str]) -> object:
        payload = {"model": self.model, "input": texts}
        try:
            reply = await post_json(
                self._transport,
                self._endpoint,
                payload,
                self._timeout,
                request="embedding",
                who=f"embedder {self.model}",
            )
        except (VaultError, TimeoutError):
            raise
        except Exception as e:
            # A connection the provider accepted and then dropped is the overload
            # case, not the absent case: worth asking again. One nobody accepted
            # means nothing is listening, which no amount of retrying fixes.
            if isinstance(getattr(e, "reason", e), RemoteDisconnected | ConnectionResetError):
                raise Transient(f"embedder {self.model}: {printable(e)}") from e
            # Nothing is listening on the default endpoint: say what to do, once.
            # No retry and no fallback to a cloud provider. A chosen endpoint
            # keeps its own error, since `ollama pull` is not the fix for it.
            if self._endpoint.defaulted:
                raise VaultError(NO_EMBEDDER) from e
            raise
        return reply

    def _vectors(self, reply: object, texts: list[str]) -> list[Vector]:
        data = reply.get("data") if isinstance(reply, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            count = len(data) if isinstance(data, list) else 0
            raise VaultError(
                f"embedder {self.model} returned {count} vectors for {len(texts)} texts"
            )
        # Response order is not promised: `index` maps each vector back to its
        # text, and every slot must be claimed exactly once.
        out: list[Vector | None] = [None] * len(texts)
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(out)
                or out[index] is not None
            ):
                raise VaultError(
                    f"embedder {self.model} returned an unusable vector index {index} "
                    f"for {len(texts)} texts"
                )
            out[index] = self._vector(item.get("embedding"))
        return [v for v in out if v is not None]

    def _vector(self, embedding: object) -> Vector:
        """A vector is only usable if it is `dims` wide and every element is finite."""
        if not isinstance(embedding, list) or len(embedding) != self.dims:
            width = len(embedding) if isinstance(embedding, list) else 0
            raise VaultError(
                f"embedder {self.model} returned a vector of width {width}, expected {self.dims}"
            )
        finite = all(
            isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
            for x in embedding
        )
        if not finite:
            raise VaultError(
                f"embedder {self.model} returned a vector that is not all finite numbers"
            )
        return [float(x) for x in embedding]


def _env_dims() -> int | None:
    raw = os.environ.get("VAULT_EMBED_DIMS")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise VaultError(f"FetchEmbedder: dims must be a positive integer, got {raw}") from None

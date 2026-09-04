"""The HTTP plumbing shared by the embedding and chat providers.

Both speak OpenAI-compatible JSON over POST, both default to a local Ollama, and
both follow the same key rules: no key is needed for localhost, a remote endpoint
must have one, and the default endpoint never picks up a key from the environment.
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .term import VaultError

# (url, headers, body, timeout_seconds) -> (status, response text)
Transport = Callable[[str, dict[str, str], bytes, float], Awaitable[tuple[int, str]]]

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


async def default_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, str]:
    """POST with urllib on a worker thread. A non-2xx status is a result, not an exception."""

    def post() -> tuple[int, str]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise e.reason from e
            raise

    return await asyncio.to_thread(post)


def is_local(endpoint: str, setting: str, who: str) -> bool:
    """Whether a configured endpoint is on this machine. Raises on a malformed URL.

    The message names the setting, never its value: a URL may carry credentials
    and this error gets printed and pasted into bug reports.
    """
    url = urlparse(endpoint)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise VaultError(f"{who}: invalid {setting} — expected an http(s) URL with a host")
    return url.hostname in _LOCAL_HOSTS


@dataclass(frozen=True)
class Endpoint:
    url: str
    key: str
    local: bool
    defaulted: bool  # nobody chose this endpoint; it is the built-in local default


def resolve_endpoint(
    who: str,
    env_prefix: str,
    default_url: str,
    endpoint: str | None = None,
    api_key: str | None = None,
) -> Endpoint:
    """Pick the endpoint and key from the arguments, then the environment, then the default.

    An ambient key belongs to whoever configured a remote provider, so the
    default endpoint never sends it. Only a chosen endpoint or an explicit key does.
    """
    setting = "endpoint" if endpoint is not None else f"{env_prefix}_ENDPOINT"
    chosen = endpoint if endpoint is not None else os.environ.get(f"{env_prefix}_ENDPOINT")
    local = chosen is None or is_local(chosen, setting, who)
    if api_key is not None:
        key = api_key
    else:
        key = "" if chosen is None else os.environ.get(f"{env_prefix}_API_KEY", "")
    if key == "" and not local:
        raise VaultError(f"{who}: no API key — pass api_key or set {env_prefix}_API_KEY")
    url = default_url if chosen is None else chosen
    return Endpoint(url=url, key=key, local=local, defaulted=url == default_url)


def redact(body: str, key: str) -> str:
    """Strip the key out of a provider's response, then cut it to a quotable length."""
    cleaned = body if key == "" else body.replace(key, "[redacted]")
    return cleaned[:200]


async def post_json(
    transport: Transport,
    endpoint: Endpoint,
    payload: dict[str, Any],
    timeout: float,
    *,
    request: str,
    who: str,
) -> Any:
    """POST `payload` and return the parsed JSON reply.

    A failed status or a non-JSON body becomes one readable error with the key
    redacted out of whatever the provider echoed back.
    """
    headers = {"content-type": "application/json"}
    if endpoint.key:
        headers["authorization"] = f"Bearer {endpoint.key}"
    status, text = await transport(endpoint.url, headers, json.dumps(payload).encode(), timeout)
    if not 200 <= status < 300:
        raise VaultError(f"{request} request failed: {status} {redact(text, endpoint.key)}")
    try:
        return json.loads(text)
    except ValueError:
        raise VaultError(
            f"{who} returned a non-JSON response: {redact(text, endpoint.key)}"
        ) from None

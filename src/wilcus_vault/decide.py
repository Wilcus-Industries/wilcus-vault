"""An OpenAI-compatible chat decider, so `vault discards restore` can re-run the gate
from the CLI. Library callers keep injecting their own; this is never a default."""

import contextlib
import os

from .decision import Decider, DeciderInput, Decision, gate_prompt, parse_decision
from .http import Transport, default_transport, post_json, redact, resolve_endpoint
from .term import VaultError

LOCAL_CHAT = "http://localhost:11434/v1/chat/completions"


def fetch_decider(
    *,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    transport: Transport = default_transport,
) -> Decider:
    """Build a Decider over `POST /v1/chat/completions`. Configuration errors raise
    here, before a database is opened. The model is always required: the local
    embedding default can be pulled once and forgotten, but a chat model that
    decides what happens to your notes is a choice nobody else makes for you."""
    ep = resolve_endpoint("FetchDecider", "VAULT_DECIDE", LOCAL_CHAT, endpoint, api_key)
    model = model if model is not None else os.environ.get("VAULT_DECIDE_MODEL")
    if not model:
        raise VaultError("FetchDecider: no model — pass model or set VAULT_DECIDE_MODEL")

    async def decide(input: DeciderInput) -> Decision:
        payload = {"model": model, "messages": [{"role": "user", "content": gate_prompt(input)}]}
        reply = await post_json(
            transport, ep, payload, timeout, request="decide", who=f"decider {model}"
        )
        content = None
        with contextlib.suppress(KeyError, IndexError, TypeError):
            content = reply["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            text = redact(str(reply), ep.key)
            raise VaultError(f"decider {model} returned no message content: {text}")
        return parse_decision(content.strip())

    return decide

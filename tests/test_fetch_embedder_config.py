"""FetchEmbedder configuration: local default, env, keys, endpoints, dims."""

import time

import pytest
from fakes import Reply, stub_transport

from wilcus_vault.fetch_embedder import FetchEmbedder
from wilcus_vault.term import VaultError

KEY = "sk-test-do-not-log-me"
REFUSED = OSError("Unable to connect. Is the computer able to access the url?")


def raising(error: BaseException) -> Reply:
    def reply(_payload: dict[str, object]) -> tuple[int, str]:
        raise error

    return reply


def test_refuses_nonsense_dims_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(VaultError, match="dims"):
        FetchEmbedder(api_key=KEY, dims=0)
    with pytest.raises(VaultError, match="dims"):
        FetchEmbedder(api_key=KEY, dims=1.5)  # type: ignore[arg-type]
    monkeypatch.setenv("VAULT_EMBED_DIMS", "wide")
    with pytest.raises(VaultError, match="dims"):
        FetchEmbedder(api_key=KEY)


async def test_unconfigured_is_a_small_local_model_never_a_cloud_provider() -> None:
    t = stub_transport(384)
    e = FetchEmbedder(transport=t)
    assert e.model == "all-minilm"
    assert e.dims == 384

    await e.embed(["a"])
    assert t.calls[0].url == "http://localhost:11434/v1/embeddings"
    # no key exists to send, and a local daemon does not want one
    assert "authorization" not in t.calls[0].headers


async def test_fails_fast_and_actionably_when_the_local_endpoint_is_unreachable() -> None:
    t = stub_transport(384, raising(REFUSED))
    e = FetchEmbedder(transport=t)
    started = time.perf_counter()
    with pytest.raises(VaultError) as info:
        await e.embed(["a"])
    assert str(info.value) == (
        "no embedder configured: start Ollama (`ollama pull all-minilm`) "
        "or configure a remote provider"
    )
    # one attempt, at the local endpoint: no retry storm, and above all no
    # silent second try against somebody's cloud API
    assert [c.url for c in t.calls] == ["http://localhost:11434/v1/embeddings"]
    assert time.perf_counter() - started < 1.0


async def test_zero_config_never_sends_an_ambient_cloud_key_to_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A key in the environment was put there for somebody's cloud provider. The
    # default endpoint is whatever process holds :11434; it does not get to
    # harvest that key just by being the default.
    t = stub_transport(384)
    monkeypatch.setenv("VAULT_EMBED_API_KEY", KEY)
    keyless = FetchEmbedder(transport=t)
    await keyless.embed(["a"])
    assert t.calls[0].url == "http://localhost:11434/v1/embeddings"
    assert KEY not in str(t.calls[0].headers)

    # An explicit key is a deliberate choice and still travels: a local gateway
    # (LiteLLM and friends) legitimately wants one.
    monkeypatch.delenv("VAULT_EMBED_API_KEY")
    gateway = FetchEmbedder(api_key=KEY, transport=t)
    await gateway.embed(["a"])
    assert t.calls[1].headers["authorization"] == f"Bearer {KEY}"

    # ...and so is an endpoint: configure one and the environment's key is yours.
    monkeypatch.setenv("VAULT_EMBED_API_KEY", KEY)
    configured = FetchEmbedder(endpoint="http://localhost:11434/v1/embeddings", transport=t)
    await configured.embed(["a"])
    assert t.calls[2].headers["authorization"] == f"Bearer {KEY}"


def test_makes_a_remote_endpoint_name_its_model_and_dims(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = "https://api.openai.com/v1/embeddings"
    # silently posting note bodies as all-minilm/384 is a wrong answer, not a default
    with pytest.raises(VaultError, match="model and dims"):
        FetchEmbedder(api_key=KEY, endpoint=remote)
    with pytest.raises(VaultError, match="dims"):
        FetchEmbedder(api_key=KEY, endpoint=remote, model="text-embedding-3-small")
    with pytest.raises(VaultError, match="model"):
        FetchEmbedder(api_key=KEY, endpoint=remote, dims=1536)
    FetchEmbedder(api_key=KEY, endpoint=remote, model="text-embedding-3-small", dims=1536)
    monkeypatch.setenv("VAULT_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VAULT_EMBED_DIMS", "1536")
    FetchEmbedder(api_key=KEY, endpoint=remote)


def test_says_which_setting_holds_a_malformed_endpoint_never_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an endpoint may carry `user:password@`, and this message is printed, logged
    # and pasted into bug reports: it names where to look, not what is there
    secret = "localhost:11434/x?token=sk-not-a-real-key"
    with pytest.raises(VaultError, match="invalid endpoint") as info:
        FetchEmbedder(endpoint=secret)
    assert "sk-not-a-real-key" not in str(info.value)
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", secret)
    with pytest.raises(VaultError, match="invalid VAULT_EMBED_ENDPOINT") as info:
        FetchEmbedder()
    assert "sk-not-a-real-key" not in str(info.value)


async def test_keeps_the_underlying_failure_as_the_cause_of_start_ollama() -> None:
    e = FetchEmbedder(transport=stub_transport(384, raising(REFUSED)))
    with pytest.raises(VaultError) as info:
        await e.embed(["a"])
    assert info.value.__cause__ is REFUSED


async def test_only_says_start_ollama_about_the_endpoint_it_chose_itself() -> None:
    # a local vLLM on :8000 is somebody's explicit choice; `ollama pull` is not
    # the fix for it, so its own error survives
    refusing = stub_transport(384, raising(REFUSED))
    e = FetchEmbedder(endpoint="http://localhost:8000/v1/embeddings", transport=refusing)
    with pytest.raises(OSError) as info:
        await e.embed(["a"])
    assert info.value is REFUSED


async def test_reports_a_hung_local_endpoint_as_a_timeout_not_as_start_ollama() -> None:
    hung = stub_transport(384, raising(TimeoutError("The operation timed out.")))
    e = FetchEmbedder(transport=hung, timeout=0.005)
    with pytest.raises(TimeoutError) as info:
        await e.embed(["a"])
    assert "no embedder configured" not in str(info.value)


def test_demands_a_key_for_a_remote_endpoint_but_not_for_a_local_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = "https://api.openai.com/v1/embeddings"
    assert isinstance(FetchEmbedder(dims=4), FetchEmbedder)
    with pytest.raises(VaultError, match="VAULT_EMBED_API_KEY"):
        FetchEmbedder(dims=4, endpoint=remote)
    for local in ("http://127.0.0.1:11434/v1/embeddings", "http://[::1]:11434/v1/embeddings"):
        assert isinstance(FetchEmbedder(dims=4, endpoint=local), FetchEmbedder)
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", remote)
    with pytest.raises(VaultError, match="VAULT_EMBED_API_KEY"):
        FetchEmbedder(dims=4)


async def test_env_config_selects_a_remote_provider_over_the_local_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = stub_transport(1536)
    monkeypatch.setenv("VAULT_EMBED_API_KEY", KEY)
    monkeypatch.setenv("VAULT_EMBED_ENDPOINT", "https://api.openai.com/v1/embeddings")
    monkeypatch.setenv("VAULT_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VAULT_EMBED_DIMS", "1536")
    e = FetchEmbedder(transport=t)
    assert [e.model, e.dims] == ["text-embedding-3-small", 1536]
    await e.embed(["a"])
    assert t.calls[0].url == "https://api.openai.com/v1/embeddings"
    assert t.calls[0].headers["authorization"] == f"Bearer {KEY}"

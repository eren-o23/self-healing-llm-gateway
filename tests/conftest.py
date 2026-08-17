from __future__ import annotations

import pytest
import yaml
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient

from gateway.config import get_config, get_redis, load_config
from gateway.providers import Outcome, ProviderResult

TEST_CONFIG = {
    "providers": {
        "anthropic": {
            "model": "claude-sonnet-5",
            "api_key_env": "ANTHROPIC_API_KEY",
            "latency_budget_ms": 15000,
        },
        "openai": {
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "latency_budget_ms": 15000,
        },
        "groq": {
            "model": "groq/llama-3.3-70b-versatile",
            "api_key_env": "GROQ_API_KEY",
            "latency_budget_ms": 5000,
        },
        "ollama": {
            "model": "ollama/llama3.2",
            "api_base_env": "OLLAMA_API_BASE",
            "api_base_default": "http://localhost:11434",
            "latency_budget_ms": 60000,
        },
    },
    "classes": {
        "interactive.chat": {"ladder": ["anthropic", "openai", "groq", "ollama"]},
        "interactive.classify": {
            "ladder": ["groq", "anthropic", "ollama"],
            # Shorter than the shipped 400ms: test_hedge.py waits the budget out
            # for real rather than faking a clock.
            "hedge_after_ms": 100,
        },
        "batch.generate": {
            "ladder": ["anthropic", "openai", "groq", "ollama"],
            "deferrable": True,
        },
    },
    "default_class": "interactive.chat",
    # Spelled out rather than inherited from the shipped defaults, so tuning
    # config/gateway.yaml for the demo cannot silently rewrite what these assert.
    "breaker": {
        "error_threshold": 0.5,
        "min_samples": 4,
        "cooldown_s": 10,
        "cooldown_max_s": 40,
        "probe_ratio": 0.1,
        "probe_successes_required": 2,
    },
    # Delays small enough that a test can advance past them with an injected
    # clock rather than sleeping, and max_attempts low enough that reaching the
    # DLQ is three lines rather than five.
    "queue": {
        "max_attempts": 3,
        "base_delay_s": 0.01,
        "max_delay_s": 0.04,
        "poll_interval_s": 0.01,
        "job_ttl_s": 60,
        "idempotency_ttl_s": 60,
    },
}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump(TEST_CONFIG))
    return path


@pytest.fixture
def config(config_path):
    return load_config(config_path)


@pytest.fixture
def all_keys_set(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(var, "test-key")


@pytest.fixture
def fake_redis():
    return FakeRedis(decode_responses=True)


@pytest.fixture
def use_test_config(config_path, monkeypatch):
    """Point get_config() at TEST_CONFIG, clearing the cache either side.

    get_config is lru_cached, and unlike a monkeypatched key - which
    ProviderConfig.available reads live - changing GATEWAY_CONFIG only takes
    effect once the cache is dropped. Anything calling get_config() outside a
    request needs this; breaker.py reads thresholds and latency budgets on every
    call, so its tests do.
    """
    monkeypatch.setenv("GATEWAY_CONFIG", str(config_path))
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def admin_secret(monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET", "test-admin-secret")


@pytest.fixture
def client(monkeypatch, use_test_config, all_keys_set, admin_secret, fake_redis):
    """App wired to the test config and a fake Redis.

    The Redis override is not a nicety: every request now records health, so
    without it each of these tests would sit through a TCP connect to a Redis
    that is not running before the failure is swallowed.
    """
    from gateway.main import app

    app.dependency_overrides[get_redis] = lambda: fake_redis
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


HEADERS = {
    "X-Tenant-Id": "acme",
    "X-Feature": "support-bot",
    "X-Request-Id": "req-1",
}

ADMIN_HEADERS = {"X-Admin-Secret": "test-admin-secret"}

BODY = {"messages": [{"role": "user", "content": "hello"}]}


class _Message:
    role = "assistant"
    content = "hi there"


class _Choice:
    index = 0
    message = _Message()
    finish_reason = "stop"


class _Usage:
    prompt_tokens = 11
    completion_tokens = 3


class _ModelResponse:
    id = "chatcmpl-abc"
    created = 1_700_000_000
    model = "gpt-4o-mini"
    choices = [_Choice()]
    usage = _Usage()


def ok_result(provider: str = "anthropic") -> ProviderResult:
    return ProviderResult(
        provider=provider,
        model="claude-sonnet-5",
        outcome=Outcome.OK,
        latency_ms=123.4,
        prompt_tokens=11,
        completion_tokens=3,
        cost_usd=0.000123,
        response=_ModelResponse(),
    )


@pytest.fixture
def stub_provider(monkeypatch):
    """Replace the provider call; these tests are about the gateway, not the network.

    Patched on gateway.router, which is where the dispatch lives now - main.py
    hands the ladder to the router and never calls a provider itself.
    """
    calls: list[str] = []

    def _install(result: ProviderResult):
        async def fake_call(name, provider, payload, chaos=None):
            calls.append(name)
            return result

        monkeypatch.setattr("gateway.router.call_provider", fake_call)
        return calls

    return _install


@pytest.fixture
def stub_ladder(monkeypatch):
    """Give each provider its own scripted result, for tests about the walk itself.

    A provider missing from the map succeeds, so a test only has to say which
    rungs are broken.
    """
    calls: list[str] = []

    def _install(results: dict[str, ProviderResult]):
        async def fake_call(name, provider, payload, chaos=None):
            calls.append(name)
            return results.get(name) or ok_result(name)

        monkeypatch.setattr("gateway.router.call_provider", fake_call)
        return calls

    return _install


def failed_result(provider: str, outcome: Outcome, latency_ms: float = 50.0):
    return ProviderResult(
        provider=provider,
        model="test-model",
        outcome=outcome,
        latency_ms=latency_ms,
        error=f"{outcome} from {provider}",
    )

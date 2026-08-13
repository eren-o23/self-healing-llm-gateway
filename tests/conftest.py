from __future__ import annotations

import pytest
import yaml
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient

from gateway.config import get_config, get_redis, load_config

TEST_CONFIG = {
    "providers": {
        "anthropic": {"model": "claude-sonnet-5", "api_key_env": "ANTHROPIC_API_KEY"},
        "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
        "groq": {"model": "groq/llama-3.3-70b-versatile", "api_key_env": "GROQ_API_KEY"},
        "ollama": {
            "model": "ollama/llama3.2",
            "api_base_env": "OLLAMA_API_BASE",
            "api_base_default": "http://localhost:11434",
        },
    },
    "classes": {
        "interactive.chat": {"ladder": ["anthropic", "openai", "groq", "ollama"]},
        "interactive.classify": {
            "ladder": ["groq", "ollama", "openai"],
            "hedge_after_ms": 400,
        },
        "batch.generate": {
            "ladder": ["anthropic", "openai", "groq", "ollama"],
            "deferrable": True,
        },
    },
    "default_class": "interactive.chat",
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
def client(config_path, monkeypatch, all_keys_set, fake_redis):
    """App wired to the test config and a fake Redis, caches cleared either side.

    The Redis override is not a nicety: every request now records health, so
    without it each of these tests would sit through a TCP connect to a Redis
    that is not running before the failure is swallowed.
    """
    from gateway.main import app

    monkeypatch.setenv("GATEWAY_CONFIG", str(config_path))
    get_config.cache_clear()
    app.dependency_overrides[get_redis] = lambda: fake_redis
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_config.cache_clear()


HEADERS = {
    "X-Tenant-Id": "acme",
    "X-Feature": "support-bot",
    "X-Request-Id": "req-1",
}

BODY = {"messages": [{"role": "user", "content": "hello"}]}

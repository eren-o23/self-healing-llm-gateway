"""API boundary: OpenAI shape in and out, and no unattributable traffic."""

from __future__ import annotations

import pytest
from litellm import exceptions as llm_exc

from gateway.providers import Outcome, ProviderResult
from tests.conftest import BODY, HEADERS, ok_result


def test_healthz_reports_provider_and_ladder_state(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["providers"]["ollama"] == "available"
    assert body["classes"]["interactive.classify"] == ["groq", "ollama", "openai"]


def test_models_lists_available_providers(client):
    data = client.get("/v1/models").json()["data"]
    assert {m["id"] for m in data} == {"anthropic", "openai", "groq", "ollama"}


def test_returns_openai_shaped_response(client, stub_provider):
    stub_provider(ok_result())

    body = client.post("/v1/chat/completions", json=BODY, headers=HEADERS).json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }


def test_response_carries_attribution(client, stub_provider):
    stub_provider(ok_result())

    meta = client.post("/v1/chat/completions", json=BODY, headers=HEADERS).json()["x_gateway"]

    assert meta["tenant"] == "acme"
    assert meta["feature"] == "support-bot"
    assert meta["request_id"] == "req-1"
    assert meta["provider"] == "anthropic"
    assert meta["cost_usd"] == pytest.approx(0.000123)


@pytest.mark.parametrize("missing", ["X-Tenant-Id", "X-Feature", "X-Request-Id"])
def test_missing_metadata_is_rejected(client, missing):
    headers = {k: v for k, v in HEADERS.items() if k != missing}

    response = client.post("/v1/chat/completions", json=BODY, headers=headers)

    assert response.status_code == 422


def test_empty_metadata_is_rejected(client):
    response = client.post(
        "/v1/chat/completions", json=BODY, headers={**HEADERS, "X-Tenant-Id": ""}
    )

    assert response.status_code == 422


def test_class_selects_the_ladder(client, stub_provider):
    calls = stub_provider(ok_result("groq"))

    client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={**HEADERS, "X-Request-Class": "interactive.classify"},
    )

    assert calls == ["groq"], "classify is cheap-first, so groq leads its ladder"


def test_unknown_class_is_rejected(client):
    response = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={**HEADERS, "X-Request-Class": "nope"},
    )

    assert response.status_code == 400


def test_provider_override_pins_the_backend(client, stub_provider):
    calls = stub_provider(ok_result("ollama"))

    client.post("/v1/chat/completions?provider=ollama", json=BODY, headers=HEADERS)

    assert calls == ["ollama"]


def test_unknown_provider_override_is_rejected(client):
    response = client.post("/v1/chat/completions?provider=nope", json=BODY, headers=HEADERS)

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (Outcome.RATE_LIMIT, 429),
        (Outcome.TIMEOUT, 504),
        (Outcome.SERVER_ERROR, 502),
        (Outcome.AUTH, 502),
        (Outcome.CONTENT_FILTER, 400),
        (Outcome.BAD_REQUEST, 400),
    ],
    ids=lambda v: str(v),
)
def test_provider_failures_map_to_http_status(client, stub_provider, outcome, status):
    stub_provider(
        ProviderResult(
            provider="anthropic",
            model="claude-sonnet-5",
            outcome=outcome,
            latency_ms=5.0,
            error="upstream said no",
        )
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert response.status_code == status
    assert response.json()["error"]["type"] == str(outcome)


def test_provider_exception_never_escapes_as_a_500(client, monkeypatch):
    """call_provider classifies rather than raises, so a sick provider is a 429, not a crash."""

    async def raising_call(name, provider, payload):
        from gateway.providers import call_provider as real

        async def boom(**_):
            raise llm_exc.RateLimitError(
                message="slow down", llm_provider="anthropic", model="claude-sonnet-5"
            )

        monkeypatch.setattr("litellm.acompletion", boom)
        return await real(name, provider, payload)

    monkeypatch.setattr("gateway.main.call_provider", raising_call)

    response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert response.status_code == 429


async def test_served_requests_land_in_the_health_window(client, stub_provider, fake_redis):
    stub_provider(ok_result())

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    from gateway.health import snapshot

    snap = await snapshot(fake_redis, "anthropic")
    assert snap.samples == 1
    assert snap.successes == 1


async def test_failed_calls_are_recorded_too(client, stub_provider, fake_redis):
    """A breaker fed only successes would never trip."""
    stub_provider(
        ProviderResult(
            provider="anthropic",
            model="claude-sonnet-5",
            outcome=Outcome.SERVER_ERROR,
            latency_ms=5.0,
            error="upstream said no",
        )
    )

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    from gateway.health import snapshot

    snap = await snapshot(fake_redis, "anthropic")
    assert snap.trippable_errors == 1


def test_unreachable_redis_does_not_fail_the_request(client, stub_provider, monkeypatch):
    """Observability is allowed to degrade; a paid-for completion is not."""
    stub_provider(ok_result())

    async def boom(*_args, **_kwargs):
        raise ConnectionError("redis is gone")

    monkeypatch.setattr("gateway.main.health.record", boom)

    response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi there"


def test_streaming_is_refused_rather_than_half_supported(client):
    response = client.post(
        "/v1/chat/completions", json={**BODY, "stream": True}, headers=HEADERS
    )

    assert response.status_code == 400


def test_missing_key_drops_provider_from_ladder(client, monkeypatch):
    """A provider without a key must degrade the ladder, never crash the service."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    health = client.get("/healthz").json()

    assert health["status"] == "ok"
    assert health["providers"]["groq"] == "disabled"
    assert "groq" not in health["classes"]["interactive.classify"]
    assert health["classes"]["interactive.classify"] == ["ollama", "openai"]


def test_pinning_a_disabled_provider_is_503(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.post("/v1/chat/completions?provider=groq", json=BODY, headers=HEADERS)

    assert response.status_code == 503

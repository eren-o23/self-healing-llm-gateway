"""Prometheus wiring: the collectors exist, and a served request moves them.

Every assertion is on a delta. prometheus_client's registry is a module-level
global, so counters carry over between tests in the same process and an absolute
value would depend on test ordering.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from gateway.providers import Outcome, ProviderResult
from tests.conftest import BODY, HEADERS, failed_result, ok_result


def _value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_the_collectors(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    body = response.text
    for name in (
        "gateway_requests_total",
        "gateway_request_duration_seconds",
        "gateway_provider_errors_total",
        "gateway_tokens_total",
        "gateway_cost_usd_total",
    ):
        assert name in body


def test_a_served_request_moves_the_counters(client, stub_provider):
    stub_provider(ok_result())
    labels = {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "class": "interactive.chat",
        "tenant": "acme",
        "feature": "support-bot",
        "outcome": "ok",
    }
    before = _value("gateway_requests_total", **labels)

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert _value("gateway_requests_total", **labels) == before + 1


def test_cost_is_attributed_to_the_tenant_that_caused_it(client, stub_provider):
    """The whole reason the metadata headers are mandatory."""
    stub_provider(ok_result())
    labels = {
        "tenant": "acme",
        "feature": "support-bot",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
    }
    before = _value("gateway_cost_usd_total", **labels)

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert _value("gateway_cost_usd_total", **labels) == pytest.approx(before + 0.000123)


def test_failures_are_counted_by_taxonomy_type(client, stub_provider):
    stub_provider(
        ProviderResult(
            provider="groq",
            model="groq/llama-3.3-70b-versatile",
            outcome=Outcome.RATE_LIMIT,
            latency_ms=12.0,
            error="slow down",
        )
    )
    labels = {"provider": "groq", "error_type": "rate_limit"}
    before = _value("gateway_provider_errors_total", **labels)

    # Pinned: unpinned, a rate limit now fails over down the whole ladder and the
    # counter moves once per rung. The walk is test_router.py's subject, not this
    # file's.
    client.post(
        "/v1/chat/completions?provider=groq",
        json=BODY,
        headers={**HEADERS, "X-Request-Class": "interactive.classify"},
    )

    assert _value("gateway_provider_errors_total", **labels) == before + 1


def test_failed_calls_do_not_book_cost_or_tokens(client, stub_provider):
    """A rate-limited call returned no tokens; billing for them would be a lie."""
    stub_provider(
        ProviderResult(
            provider="anthropic",
            model="claude-sonnet-5",
            outcome=Outcome.SERVER_ERROR,
            latency_ms=8.0,
            error="upstream said no",
        )
    )
    labels = {
        "tenant": "acme",
        "feature": "support-bot",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
    }
    before = _value("gateway_cost_usd_total", **labels)

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert _value("gateway_cost_usd_total", **labels) == before


# --- client-facing responses --------------------------------------------------


def test_one_response_is_counted_once_however_many_rungs_it_took(
    client, stub_ladder
):
    """The distinction the availability figure depends on.

    Two providers fail, the third serves: three provider calls, one answer to the
    caller. Counting availability off gateway_requests_total would put those
    failures in the denominator - so an outage the gateway completely absorbed
    would drag the number down, and one it absorbed by hedging would push it up.
    """
    labels = {"route": "/v1/chat/completions", "status": "200"}
    before = _value("gateway_responses_total", **labels)
    calls = stub_ladder(
        {
            "anthropic": failed_result("anthropic", Outcome.SERVER_ERROR),
            "openai": failed_result("openai", Outcome.TIMEOUT),
        }
    )

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert len(calls) == 3
    assert _value("gateway_responses_total", **labels) == before + 1


def test_a_rejection_raised_before_the_ladder_is_still_counted(client):
    """Streaming is refused with a 400 raised straight out of the handler.

    Nothing routes, so a counter incremented next to the ladder would miss it
    entirely - and a 5xx raised the same way is exactly what the availability
    figure must not be allowed to overlook. Hence one middleware rather than an
    increment per return path.
    """
    labels = {"route": "/v1/chat/completions", "status": "400"}
    before = _value("gateway_responses_total", **labels)

    response = client.post(
        "/v1/chat/completions", json={**BODY, "stream": True}, headers=HEADERS
    )

    assert response.status_code == 400
    assert _value("gateway_responses_total", **labels) == before + 1


def test_the_job_route_is_one_series_not_one_per_job(client):
    """A raw path label here would mint a new time series per job id."""
    client.get("/v1/jobs/some-job-id", headers=HEADERS)
    client.get("/v1/jobs/another-job-id", headers=HEADERS)

    assert _value("gateway_responses_total", route="/v1/jobs/{job_id}", status="404") >= 2


# --- circuit gauge ------------------------------------------------------------


async def test_the_circuit_gauge_has_series_before_any_traffic(
    fake_redis, use_test_config
):
    """The demo's money panel must not be blank on a cold `docker compose up`.

    The gauge is only ever set as a side effect of reading a breaker, so with no
    seeding a freshly started stack exports the HELP and TYPE lines and nothing
    underneath them.
    """
    from gateway import breaker
    from gateway.config import get_config

    providers = list(get_config().providers)
    await breaker.seed_gauge(fake_redis, providers)

    for provider in providers:
        # `is not None` rather than `== 0`: a missing series and a closed circuit
        # both read as zero through get_sample_value, and it is the missing one
        # this exists to catch.
        assert REGISTRY.get_sample_value(
            "gateway_circuit_state", {"provider": provider}
        ) is not None


async def test_seeding_survives_redis_being_down(use_test_config):
    """Boot order is not guaranteed, and a blank panel beats a service that won't start."""
    from gateway import breaker

    class _Dead:
        async def hgetall(self, *_args, **_kwargs):
            raise ConnectionError("redis is not up yet")

    await breaker.seed_gauge(_Dead(), ["anthropic"])


def test_latency_buckets_reach_past_ollama(client, stub_provider):
    """Ollama takes ~14s on CPU; the default buckets stop at 10s and hide it.

    Without this the local provider's p95 is +Inf, which would silently break
    phase 3's per-provider latency budget.
    """
    result = ok_result("ollama")
    result.latency_ms = 14_000.0
    stub_provider(result)
    labels = {"provider": "ollama", "class": "interactive.chat", "le": "20.0"}
    before = _value("gateway_request_duration_seconds_bucket", **labels)

    client.post("/v1/chat/completions?provider=ollama", json=BODY, headers=HEADERS)

    assert _value("gateway_request_duration_seconds_bucket", **labels) == before + 1

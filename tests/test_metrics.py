"""Prometheus wiring: the collectors exist, and a served request moves them.

Every assertion is on a delta. prometheus_client's registry is a module-level
global, so counters carry over between tests in the same process and an absolute
value would depend on test ordering.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from gateway.providers import Outcome, ProviderResult
from tests.conftest import BODY, HEADERS, ok_result


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

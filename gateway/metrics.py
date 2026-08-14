"""Prometheus collectors: the monitoring path, as opposed to health.py's control path.

These drive Grafana and the demo video. Nothing routes off them - see the module
docstring in health.py for why that separation is deliberate.

Collectors are defined at module scope because prometheus_client keeps a global
registry, and building one inside a function raises Duplicated timeseries the
second time it runs.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from gateway.providers import ProviderResult
from gateway.schemas import RequestMetadata

# ponytail: tenant x feature x provider x model is fine at demo cardinality; drop
# tenant from the histogram first if a real deployment ever strains it
# ponytail: single-process registry - needs prometheus_client's multiprocess mode
# if uvicorn ever runs with --workers > 1

REQUESTS = Counter(
    "gateway_requests_total",
    "Provider calls by attribution and outcome",
    ["provider", "model", "class", "tenant", "feature", "outcome"],
)

# The default buckets stop at 10s. Ollama genuinely takes ~14s on CPU and carries
# a deliberate 120s timeout, so on defaults every local call lands in +Inf and its
# p95 is unreadable - which would quietly break phase 3's per-provider latency
# budget before it was even written.
DURATION = Histogram(
    "gateway_request_duration_seconds",
    "Provider call latency",
    ["provider", "class"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)

ERRORS = Counter(
    "gateway_provider_errors_total",
    "Provider failures by taxonomy type",
    ["provider", "error_type"],
)

TOKENS = Counter(
    "gateway_tokens_total",
    "Tokens by direction",
    ["provider", "model", "direction"],
)

COST = Counter(
    "gateway_cost_usd_total",
    "Spend attributed to the tenant and feature that caused it",
    ["tenant", "feature", "provider", "model"],
)

# The demo's most important panel: a timeline of this gauge is the trip, the
# reroute and the recovery in one line. Set on every breaker read rather than
# only on transitions, so it stays live for as long as traffic flows.
CIRCUIT_STATE = Gauge(
    "gateway_circuit_state",
    "Breaker state per provider: 0 closed, 1 open, 2 half-open",
    ["provider"],
)

# "from" is a Python keyword, so these labels are passed as **{"from": ...} -
# the same shape observe() already needs for "class".
FAILOVERS = Counter(
    "gateway_failovers_total",
    "Ladder hops taken because a provider failed",
    ["from", "to", "class", "reason"],
)

# Hedging roughly doubles spend on the calls it fires. Measuring fired vs won vs
# wasted turns that from a caveat in the README into a number on a dashboard.
HEDGE = Counter(
    "gateway_hedge_total",
    "Hedge outcomes: fired, won_by_hedge, wasted",
    ["class", "outcome"],
)


def observe(result: ProviderResult, meta: RequestMetadata) -> None:
    """Fold one provider call into every collector. In-process, so it cannot fail."""
    REQUESTS.labels(
        provider=result.provider,
        model=result.model,
        **{"class": meta.request_class},
        tenant=meta.tenant,
        feature=meta.feature,
        outcome=str(result.outcome),
    ).inc()

    DURATION.labels(provider=result.provider, **{"class": meta.request_class}).observe(
        result.latency_ms / 1000
    )

    if not result.outcome.ok:
        ERRORS.labels(provider=result.provider, error_type=str(result.outcome)).inc()
        return

    TOKENS.labels(
        provider=result.provider, model=result.model, direction="prompt"
    ).inc(result.prompt_tokens)
    TOKENS.labels(
        provider=result.provider, model=result.model, direction="completion"
    ).inc(result.completion_tokens)

    COST.labels(
        tenant=meta.tenant,
        feature=meta.feature,
        provider=result.provider,
        model=result.model,
    ).inc(result.cost_usd)

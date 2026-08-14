"""Per-provider sliding health window, held in Redis.

Redis and Prometheus deliberately both carry health data. Prometheus is for
humans - dashboards, history, the demo video. This module is the control path:
the state phase 3's routing decision reads on every request. A gateway that
queried Prometheus to decide where to send traffic would be wiring a control
path through a monitoring system.

The window survives a restart by construction. The data lives in Redis and
eviction is time-based, so a gateway that just came up reads a correct view of
the last minute immediately rather than starting blind.
"""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gateway.config import get_config
from gateway.providers import Outcome

if TYPE_CHECKING:
    from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """One provider's recent behaviour, as the breaker will come to read it.

    `samples` counts everything in the window. `health_samples` counts only the
    calls that say something about the provider: successes plus trippable
    failures. Rates and percentiles are computed over the latter, because
    CONTENT_FILTER and BAD_REQUEST are the caller's fault and carry no
    information about whether the provider is well.

    Excluding them from the numerator alone is not enough. A flood of bad input
    would inflate the denominator and drive the error rate down, which would let
    one user's malformed traffic mask a genuinely sick provider - the same bug as
    letting bad input trip a breaker, just pointing the other way.

    An empty window reads as healthy. Fail-open is the right default for a
    gateway, and phase 3's min_samples is the real guard against acting on thin
    evidence - it gates on health_samples, not samples, so a hundred bad requests
    and three real failures is not enough evidence to trip.
    """

    provider: str
    samples: int = 0
    health_samples: int = 0
    successes: int = 0
    success_rate: float = 1.0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    trippable_errors: int = 0
    trippable_error_rate: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


def key(provider: str) -> str:
    """Public: breaker.py clears this window when a circuit starts probing."""
    return f"health:{provider}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _window_ms(window_s: int | None) -> int:
    return (window_s if window_s is not None else get_config().health.window_s) * 1000


async def record(
    r: Redis,
    provider: str,
    outcome: Outcome,
    latency_ms: float,
    *,
    window_s: int | None = None,
    now_ms: int | None = None,
) -> None:
    """Append one call to the provider's window and evict what fell out of it.

    The member carries a random token because sorted-set members must be unique.
    A process-local counter would do until phase 4 adds worker.py as a second
    writer against the same keys, at which point two sequences would collide and
    silently undercount.
    """
    now_ms = _now_ms() if now_ms is None else now_ms
    window_ms = _window_ms(window_s)
    member = f"{now_ms}:{uuid.uuid4().hex[:8]}:{outcome}:{latency_ms:.1f}"

    pipe = r.pipeline()
    pipe.zadd(key(provider), {member: now_ms})
    pipe.zremrangebyscore(key(provider), "-inf", now_ms - window_ms)
    pipe.expire(key(provider), (window_ms // 1000) * 2)
    await pipe.execute()


async def snapshot(
    r: Redis,
    provider: str,
    *,
    window_s: int | None = None,
    now_ms: int | None = None,
) -> HealthSnapshot:
    """Summarise the provider's current window.

    Filtered on read as well as evicted on write. Eviction only runs when
    something is written, so a provider that went quiet still holds its last
    samples - and a provider goes quiet precisely because it got sick and traffic
    moved away. Reading without the filter would hand the breaker that stale,
    rosy pre-failure window and it would never trip. The write-side eviction is
    only there to bound memory.
    """
    now_ms = _now_ms() if now_ms is None else now_ms
    cutoff = now_ms - _window_ms(window_s)
    members = await r.zrangebyscore(key(provider), f"({cutoff}", "+inf")
    return _summarise(provider, members)


def _summarise(provider: str, members: list[str]) -> HealthSnapshot:
    if not members:
        return HealthSnapshot(provider=provider)

    successes = 0
    trippable = 0
    errors: dict[str, int] = {}
    latencies: list[float] = []

    for member in members:
        _, _, name, latency = member.split(":", 3)
        outcome = Outcome(name)

        if outcome.ok:
            successes += 1
        else:
            errors[name] = errors.get(name, 0) + 1
            if outcome.trippable:
                trippable += 1

        if outcome.ok or outcome.trippable:
            latencies.append(float(latency))

    health_samples = successes + trippable
    p50, p95, p99 = _percentiles(latencies)
    return HealthSnapshot(
        provider=provider,
        samples=len(members),
        health_samples=health_samples,
        successes=successes,
        success_rate=successes / health_samples if health_samples else 1.0,
        errors_by_type=errors,
        trippable_errors=trippable,
        trippable_error_rate=trippable / health_samples if health_samples else 0.0,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
    )


def _percentiles(latencies: list[float]) -> tuple[float, float, float]:
    """p50/p95/p99, guarded at the sizes statistics.quantiles refuses to handle.

    quantiles raises below two data points, and one sample is a perfectly normal
    state for a window that just opened.
    """
    # ponytail: percentiles computed in-process over the raw window - fine to a few
    # thousand samples/window; move to Redis TDIGEST if the window ever gets big
    # enough to matter
    if not latencies:
        return 0.0, 0.0, 0.0
    if len(latencies) == 1:
        return latencies[0], latencies[0], latencies[0]
    cuts = statistics.quantiles(latencies, n=100, method="inclusive")
    return cuts[49], cuts[94], cuts[98]

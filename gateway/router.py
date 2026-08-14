"""The ladder walk: which provider serves a request, and what happens when it can't.

This is the module that makes the gateway self-healing rather than merely
instrumented. Phase 1 called ladder[0] and returned whatever came back; from here
a failed rung is a routing decision, not an error to hand the caller.

call_provider() never raises for provider failures - it returns a ProviderResult
carrying an Outcome - so this is a flat loop over results rather than a
try/except around every rung.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gateway import breaker, chaos, health, metrics
from gateway.config import GatewayConfig
from gateway.providers import ProviderCallLog, ProviderResult, call_provider
from gateway.schemas import RequestMetadata

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = logging.getLogger(__name__)

SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class RouteOutcome:
    """What the walk produced, plus the trail of everything it tried.

    `result` is None only when the ladder ran out. The attempts trail goes onto
    every response, successful or not: on a 503 it is the difference between
    "everything failed" and a list naming each provider and why it was passed
    over, which costs nothing to build and is the first thing anybody asks for.
    """

    result: ProviderResult | None
    attempts: list[ProviderCallLog] = field(default_factory=list)

    @property
    def last_failure(self) -> ProviderCallLog | None:
        """The last rung that actually ran, as opposed to one that was skipped.

        What the caller is told on an exhausted ladder depends on this. A ladder
        where every circuit was open genuinely has no capacity - that is what 503
        means. A ladder where every provider was rate limited is a 429, and
        flattening it to 503 throws away the one signal that tells a client to
        back off rather than retry immediately.
        """
        return next((a for a in reversed(self.attempts) if a.outcome != SKIPPED), None)


async def route(
    r: Redis,
    config: GatewayConfig,
    meta: RequestMetadata,
    payload: dict,
    ladder: list[str],
) -> RouteOutcome:
    """Walk the ladder until something answers."""
    attempts: list[ProviderCallLog] = []
    hedge_after_ms = config.classes[meta.request_class].hedge_after_ms
    start = 0

    if hedge_after_ms:
        routed, start = await _hedged_start(
            r, config, meta, payload, ladder, hedge_after_ms
        )
        attempts.extend(routed.attempts)
        if routed.result is not None:
            return RouteOutcome(routed.result, attempts)

    for i in range(start, len(ladder)):
        provider = ladder[i]
        admit, reason = await breaker.allows(r, provider)
        if not admit:
            attempts.append(_skipped(provider, reason))
            continue

        result = await attempt(r, config, meta, payload, provider)
        attempts.append(_logged(result))

        if _is_final(result):
            return RouteOutcome(result, attempts)

        _count_failover(provider, ladder, i, meta, result)

    return RouteOutcome(None, attempts)


async def _hedged_start(
    r: Redis,
    config: GatewayConfig,
    meta: RequestMetadata,
    payload: dict,
    ladder: list[str],
    hedge_after_ms: int,
) -> tuple[RouteOutcome, int]:
    """Race the first two admissible rungs, and report where the plain walk resumes.

    A latency-sensitive class would rather pay for a second call than wait out a
    slow tail. The budget is what keeps that honest: the hedge only fires for
    calls that have already missed it, so a provider behaving normally costs
    nothing extra.

    Returns (outcome, resume_index). A result means the race settled it; None
    means both rungs failed in a way worth walking on from, and resume_index is
    the rung after the hedge.

    # ponytail: hedges exactly one extra rung, never a fan-out across the ladder
    """
    admitted = await _admissible(r, ladder, limit=2)
    if len(admitted) < 2:
        return RouteOutcome(None, []), 0  # nothing to race; the plain walk handles it

    (first_i, first), (second_i, second) = admitted
    request_class = meta.request_class
    attempts: list[ProviderCallLog] = []

    primary = asyncio.create_task(attempt(r, config, meta, payload, first))
    done, _ = await asyncio.wait({primary}, timeout=hedge_after_ms / 1000)
    if done:
        result = primary.result()
        attempts.append(_logged(result))
        if _is_final(result):
            return RouteOutcome(result, attempts), 0
        _count_failover(first, ladder, first_i, meta, result)
        return RouteOutcome(None, attempts), first_i + 1

    metrics.HEDGE.labels(outcome="fired", **{"class": request_class}).inc()
    log.info(
        "hedge fired request_id=%s %s did not answer in %dms; adding %s",
        meta.request_id,
        first,
        hedge_after_ms,
        second,
    )
    hedge = asyncio.create_task(attempt(r, config, meta, payload, second))

    winner, loser = await _first_usable(primary, hedge)
    result = winner.result()
    served = _is_final(result)

    # "Won" means the extra call actually produced the answer. A hedge that came
    # back first with a failure did not win anything, so it counts as wasted -
    # which is the honest way to price hedging.
    metrics.HEDGE.labels(
        outcome="won_by_hedge" if (winner is hedge and served) else "wasted",
        **{"class": request_class},
    ).inc()

    # Both calls are real and billed, and attempt() has already recorded each to
    # health, Prometheus and the breaker. Only the trail order is decided here:
    # the winner first, because that is the one that served.
    attempts.append(_logged(result))
    if loser.done() and not loser.cancelled():
        attempts.append(_logged(loser.result()))

    if served:
        return RouteOutcome(result, attempts), 0
    _count_failover(result.provider, ladder, second_i, meta, result)
    return RouteOutcome(None, attempts), second_i + 1


async def _first_usable(
    primary: asyncio.Task, hedge: asyncio.Task
) -> tuple[asyncio.Task, asyncio.Task]:
    """The first task to return something worth having, and the one that lost.

    Whoever finishes first usually wins - but not if it finished by failing. In
    that case the other task is still this request's best chance, so it is
    awaited rather than thrown away.

    The loser is cancelled *and awaited*. An un-awaited cancelled task keeps
    running, and a provider mid-generation goes on billing for tokens it has
    already produced - the exact cost the hedge budget exists to bound.
    """
    done, pending = await asyncio.wait(
        {primary, hedge}, return_when=asyncio.FIRST_COMPLETED
    )

    if not pending:  # both landed in the same tick
        return (
            (primary, hedge) if _is_final(primary.result()) else (hedge, primary)
        )

    first, other = done.pop(), pending.pop()
    if _is_final(first.result()):
        other.cancel()
        await asyncio.gather(other, return_exceptions=True)
        return first, other

    await asyncio.wait({other})
    return (other, first) if _is_final(other.result()) else (first, other)


async def _admissible(
    r: Redis, ladder: list[str], *, limit: int
) -> list[tuple[int, str]]:
    """The first `limit` rungs whose circuits will take traffic, with their positions.

    Checked before racing so a hedge never spends its second call on a provider
    the breaker has already taken out.
    """
    admitted: list[tuple[int, str]] = []
    for i, provider in enumerate(ladder):
        if (await breaker.allows(r, provider))[0]:
            admitted.append((i, provider))
            if len(admitted) == limit:
                break
    return admitted


async def attempt(
    r: Redis,
    config: GatewayConfig,
    meta: RequestMetadata,
    payload: dict,
    provider: str,
) -> ProviderResult:
    """One rung: dispatch it, then record it everywhere that needs to know."""
    result = await call_provider(
        provider, config.providers[provider], payload, chaos=await chaos.get(r, provider)
    )
    await _observe(r, result, meta)
    return result


def _is_final(result: ProviderResult) -> bool:
    """Whether this result ends the walk.

    Success obviously ends it. So does a non-trippable failure: BAD_REQUEST and
    CONTENT_FILTER mean the caller sent something the provider was right to
    reject, and every remaining rung will reject it identically. Failing over
    would bill three more calls to reproduce the same 400 - and it is the same
    mistake as letting caller faults open a breaker, just pointed at the ladder
    instead of the circuit.
    """
    return result.outcome.ok or not result.outcome.trippable


def _count_failover(
    provider: str,
    ladder: list[str],
    i: int,
    meta: RequestMetadata,
    result: ProviderResult,
) -> None:
    next_provider = ladder[i + 1] if i + 1 < len(ladder) else "none"
    log.info(
        "failover request_id=%s %s -> %s reason=%s",
        meta.request_id,
        provider,
        next_provider,
        result.outcome,
    )
    metrics.FAILOVERS.labels(
        to=next_provider,
        reason=str(result.outcome),
        **{"from": provider, "class": meta.request_class},
    ).inc()


def _skipped(provider: str, reason: str) -> ProviderCallLog:
    return ProviderCallLog(
        provider=provider, outcome=SKIPPED, latency_ms=0.0, error=reason
    )


def _logged(result: ProviderResult) -> ProviderCallLog:
    return ProviderCallLog(
        provider=result.provider,
        outcome=str(result.outcome),
        latency_ms=round(result.latency_ms, 1),
        error=result.error,
    )


async def _observe(r: Redis, result: ProviderResult, meta: RequestMetadata) -> None:
    """Record the call to every store, without letting any of them reach the caller.

    Prometheus first and outside the guard: it is in-process and cannot really
    fail, so it keeps its data even when Redis is unreachable.

    The Redis writes are guarded because a monitoring write must not fail a
    completion the caller has already been billed for - Redis being down should
    cost observability, not availability.

    The consequence is worth naming because it is this phase's failure mode: no
    Redis means an empty window and a circuit that reads CLOSED, so nothing trips
    and every request goes to rung one. That is fail-open, and for a gateway it
    is the right direction to fail - a broken breaker must not take every
    provider offline at once.
    """
    metrics.observe(result, meta)
    try:
        await health.record(r, result.provider, result.outcome, result.latency_ms)
        await breaker.record(r, result.provider, result.outcome)
    except Exception:  # noqa: BLE001 - degraded observability beats a failed request
        log.warning(
            "health/breaker record failed for %s; it will read healthier than it is",
            result.provider,
        )

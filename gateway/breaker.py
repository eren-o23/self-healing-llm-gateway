"""Per-provider circuit breaker: the thing that actually takes a sick backend out.

State lives in a Redis hash rather than in the process, for the same reason the
health window does - it has to survive a restart, and in phase 4 a second process
(worker.py) will read the same circuits.

Transitions are computed lazily on read. A background timer flipping OPEN to
HALF_OPEN after a cooldown would need a scheduler, a task to supervise, and a
story for what happens when two of them run; comparing two integers when
somebody asks needs none of that and is exactly as correct.

Half-open is what makes this self-healing rather than merely defensive. Without
it a breaker opens once and stays open, which is a worse outage than the one it
was protecting against.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from gateway import health, metrics
from gateway.config import get_config
from gateway.providers import Outcome

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = logging.getLogger(__name__)


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_GAUGE_VALUE = {State.CLOSED: 0, State.OPEN: 1, State.HALF_OPEN: 2}


@dataclass(frozen=True, slots=True)
class CircuitState:
    """One provider's circuit, as the router and /admin/circuits both read it."""

    provider: str
    state: State = State.CLOSED
    opened_at_ms: int | None = None
    cooldown_s: float = 0.0
    probe_successes: int = 0
    retry_in_s: float = 0.0


async def seed_gauge(r: Redis, providers: list[str]) -> None:
    """Publish every circuit to Prometheus before any traffic arrives.

    gateway_circuit_state is set as a side effect of reading a breaker, so on a
    cold stack the gauge has no series at all: /metrics carries its HELP and TYPE
    lines and nothing else, and the circuit timeline - the panel the whole demo
    is built around - stays blank until traffic happens to flow. "docker compose
    up gives a working dashboard" is an acceptance criterion, so both processes
    call this at startup.

    Guarded, because Redis may not be up yet and observability must never be the
    thing that stops the gateway booting. Same fail-open rule as router._observe.

    Lives here rather than in main.py so worker.py can call it without importing
    the FastAPI app, for the same reason to_openai_response sits in schemas.
    """
    try:
        for provider in providers:
            await state_of(r, provider)
    except Exception:  # noqa: BLE001 - a blank panel beats a service that will not start
        log.warning("could not seed circuit state; panels fill in on the first request")


def _key(provider: str) -> str:
    return f"breaker:{provider}"


def _now_ms() -> int:
    return int(time.time() * 1000)


async def state_of(r: Redis, provider: str, *, now_ms: int | None = None) -> CircuitState:
    """Current state, applying the OPEN -> HALF_OPEN transition if the cooldown ran out.

    Reading is what advances the clock here, so a provider nobody asks about
    simply stays as it was - which is correct, because a circuit only matters at
    the moment something wants to route through it.
    """
    now_ms = _now_ms() if now_ms is None else now_ms
    cfg = get_config().breaker

    raw = await r.hgetall(_key(provider))
    state = State(raw.get("state") or State.CLOSED)
    opened_at_ms = int(raw["opened_at_ms"]) if raw.get("opened_at_ms") else None
    cooldown_s = float(raw.get("cooldown_s") or cfg.cooldown_s)
    probe_successes = int(raw.get("probe_successes") or 0)
    retry_in_s = 0.0

    if state is State.OPEN and opened_at_ms is not None:
        elapsed_s = (now_ms - opened_at_ms) / 1000
        if elapsed_s >= cooldown_s:
            await _enter_half_open(r, provider, cooldown_s)
            state, probe_successes = State.HALF_OPEN, 0
        else:
            retry_in_s = cooldown_s - elapsed_s

    metrics.CIRCUIT_STATE.labels(provider=provider).set(_GAUGE_VALUE[state])
    return CircuitState(
        provider=provider,
        state=state,
        opened_at_ms=opened_at_ms,
        cooldown_s=cooldown_s,
        probe_successes=probe_successes,
        retry_in_s=retry_in_s,
    )


async def allows(
    r: Redis, provider: str, *, now_ms: int | None = None
) -> tuple[bool, str]:
    """Whether to route to this provider, and a human-readable reason if not.

    The reason is not decoration: it goes into the attempts trail on every
    response and into the 503 body when the ladder runs out, which is the
    difference between "everything failed" and "here is what happened".
    """
    circuit = await state_of(r, provider, now_ms=now_ms)

    if circuit.state is State.CLOSED:
        return True, ""
    if circuit.state is State.OPEN:
        return False, f"circuit open, retry in {circuit.retry_in_s:.0f}s"

    # Half-open: a trickle of real traffic is the probe. Sending everything would
    # hand a still-sick provider the full load the instant the cooldown expires.
    if random.random() < get_config().breaker.probe_ratio:
        return True, ""
    return False, "circuit half-open, not selected as probe"


async def record(
    r: Redis, provider: str, outcome: Outcome, *, now_ms: int | None = None
) -> None:
    """Fold one call's outcome into the circuit. Called after every attempt.

    Reads the health window itself so the router does not have to know that the
    trip condition is a windowed statistic rather than a per-call one.
    """
    now_ms = _now_ms() if now_ms is None else now_ms
    circuit = await state_of(r, provider, now_ms=now_ms)

    if circuit.state is State.HALF_OPEN:
        await _record_probe(r, provider, outcome, circuit, now_ms)
    elif circuit.state is State.CLOSED:
        await _maybe_trip(r, provider, now_ms)
    # OPEN: a call that raced the breaker shut. Its outcome decides nothing.


async def _maybe_trip(r: Redis, provider: str, now_ms: int) -> None:
    """CLOSED -> OPEN on a bad window, either too many real failures or too slow.

    min_samples gates on health_samples - successes plus trippable failures -
    rather than the raw sample count. Gating on raw samples would let a hundred
    caller faults and three genuine failures clear the threshold at a 100% error
    rate, and trip a healthy provider on three data points.

    Caller faults cannot reach either condition: health.py already keeps
    CONTENT_FILTER and BAD_REQUEST out of the rate and out of the percentiles.
    That is why a provider returning nothing but content filters never trips,
    without a special case here.
    """
    cfg = get_config().breaker
    budget_ms = get_config().providers[provider].latency_budget_ms
    snap = await health.snapshot(r, provider, now_ms=now_ms)

    if snap.health_samples < cfg.min_samples:
        return

    if snap.trippable_error_rate > cfg.error_threshold:
        reason = (
            f"error rate {snap.trippable_error_rate:.0%} over "
            f"{snap.health_samples} samples"
        )
    elif snap.p95_ms > budget_ms:
        reason = f"p95 {snap.p95_ms:.0f}ms over budget {budget_ms:.0f}ms"
    else:
        return

    await _open(r, provider, now_ms, cfg.cooldown_s, reason)


async def _record_probe(
    r: Redis, provider: str, outcome: Outcome, circuit: CircuitState, now_ms: int
) -> None:
    """HALF_OPEN resolves on individual probe results, not on the window.

    The window is empty at this point by design (see _enter_half_open), so there
    is nothing to average over - and waiting to accumulate min_samples before
    letting a recovered provider back in would make recovery slower than the
    outage.
    """
    cfg = get_config().breaker

    if outcome.ok:
        successes = circuit.probe_successes + 1
        if successes < cfg.probe_successes_required:
            await r.hset(_key(provider), "probe_successes", str(successes))
            return
        log.info("circuit CLOSED for %s after %d successful probes", provider, successes)
        await r.hset(
            _key(provider),
            mapping={
                "state": State.CLOSED,
                "opened_at_ms": "",
                "cooldown_s": str(cfg.cooldown_s),
                "probe_successes": "0",
            },
        )
        metrics.CIRCUIT_STATE.labels(provider=provider).set(_GAUGE_VALUE[State.CLOSED])
        return

    if not outcome.trippable:
        return  # the caller's fault; it says nothing about whether the provider is well

    # One failed probe is enough. The provider has already proven it can fail, and
    # backing off further each time stops a flapping provider being probed to death.
    cooldown_s = min(circuit.cooldown_s * 2, cfg.cooldown_max_s)
    await _open(r, provider, now_ms, cooldown_s, f"probe failed with {outcome}")


async def _open(
    r: Redis, provider: str, now_ms: int, cooldown_s: float, reason: str
) -> None:
    log.warning(
        "circuit OPEN for %s: %s; cooling down %.0fs", provider, reason, cooldown_s
    )
    # ponytail: read-modify-write with no WATCH. Concurrent requests can admit an
    # extra probe or apply the cooldown doubling twice; both are harmless at any
    # load this will see. Upgrade path is a Lua script if it ever matters.
    await r.hset(
        _key(provider),
        mapping={
            "state": State.OPEN,
            "opened_at_ms": str(now_ms),
            "cooldown_s": str(cooldown_s),
            "probe_successes": "0",
        },
    )
    metrics.CIRCUIT_STATE.labels(provider=provider).set(_GAUGE_VALUE[State.OPEN])


async def _enter_half_open(r: Redis, provider: str, cooldown_s: float) -> None:
    """Start probing - and throw away the window that caused the trip.

    The window is still full of the failures that opened the circuit, and traffic
    moved away the moment it opened, so nothing has aged out. Left in place, the
    first successful probe would be judged against it and _maybe_trip would slam
    the circuit shut again on evidence from the previous outage. Probes need a
    clean window, and the history that matters for humans is in Prometheus, not
    here.
    """
    log.info("circuit HALF_OPEN for %s: cooldown elapsed, admitting probes", provider)
    await r.hset(
        _key(provider),
        mapping={
            "state": State.HALF_OPEN,
            "opened_at_ms": "",
            "cooldown_s": str(cooldown_s),
            "probe_successes": "0",
        },
    )
    await r.delete(health.key(provider))

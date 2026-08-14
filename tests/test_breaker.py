"""The state machine, driven by an injected clock.

Nothing here sleeps. A breaker whose tests wait out real cooldowns is a breaker
nobody runs, and the interesting behaviour is entirely about which side of a
boundary a timestamp falls on.

Probe admission is the one deliberately random thing in the module, so the tests
that care about it pin random.random() rather than hoping.
"""

from __future__ import annotations

import pytest
from fakeredis.aioredis import FakeRedis

from gateway import breaker, health
from gateway.breaker import State
from gateway.providers import Outcome

NOW = 1_700_000_000_000
COOLDOWN_MS = 10_000  # matches TEST_CONFIG's breaker.cooldown_s

pytestmark = pytest.mark.usefixtures("use_test_config")


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


async def _fill(r, provider="anthropic", *, outcomes, latency_ms=100.0, at=NOW):
    """Record calls into both stores the way the router does: health, then breaker."""
    for outcome in outcomes:
        await health.record(r, provider, outcome, latency_ms, now_ms=at)
        await breaker.record(r, provider, outcome, now_ms=at)


async def _state(r, provider="anthropic", at=NOW) -> State:
    return (await breaker.state_of(r, provider, now_ms=at)).state


# --- closed -------------------------------------------------------------------


async def test_a_fresh_provider_is_closed(redis):
    assert await _state(redis) == State.CLOSED
    assert await breaker.allows(redis, "anthropic", now_ms=NOW) == (True, "")


async def test_does_not_trip_below_min_samples(redis):
    """Three failures is not evidence. min_samples exists to say so."""
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 3)

    assert await _state(redis) == State.CLOSED


async def test_trips_on_error_rate(redis):
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)

    assert await _state(redis) == State.OPEN


async def test_does_not_trip_at_exactly_the_threshold(redis):
    """The condition is a strict >: half failing is the line, not past it."""
    await _fill(redis, outcomes=[Outcome.OK, Outcome.OK, Outcome.TIMEOUT, Outcome.TIMEOUT])

    assert await _state(redis) == State.CLOSED


async def test_trips_on_p95_over_that_providers_budget(redis):
    """Slow enough is broken, even with every call returning 200."""
    await _fill(redis, "groq", outcomes=[Outcome.OK] * 10, latency_ms=9000.0)

    assert await _state(redis, "groq") == State.OPEN


async def test_latency_budgets_are_per_provider(redis):
    """The same 20s that would open groq is a normal day for ollama.

    A single global budget would hold ollama's breaker open permanently and take
    the floor out of every ladder - the one failure this design cannot route
    around.
    """
    await _fill(redis, "ollama", outcomes=[Outcome.OK] * 10, latency_ms=20_000.0)
    await _fill(redis, "groq", outcomes=[Outcome.OK] * 10, latency_ms=20_000.0)

    assert await _state(redis, "ollama") == State.CLOSED
    assert await _state(redis, "groq") == State.OPEN


async def test_content_filter_only_never_trips(redis):
    """The acceptance criterion, and the most commonly botched part of a gateway.

    One user sending prompts that trip a safety filter must not take a healthy
    provider offline for everybody else.
    """
    await _fill(redis, outcomes=[Outcome.CONTENT_FILTER] * 50)

    assert await _state(redis) == State.CLOSED


async def test_bad_requests_cannot_dilute_a_real_failure_rate(redis):
    """The inverse trap: junk traffic must not hide a provider that is genuinely down."""
    await _fill(
        redis,
        outcomes=[Outcome.OK] + [Outcome.SERVER_ERROR] * 4 + [Outcome.BAD_REQUEST] * 50,
    )

    assert await _state(redis) == State.OPEN


async def test_providers_have_independent_circuits(redis):
    await _fill(redis, "anthropic", outcomes=[Outcome.SERVER_ERROR] * 4)

    assert await _state(redis, "anthropic") == State.OPEN
    assert await _state(redis, "groq") == State.CLOSED


# --- open ---------------------------------------------------------------------


async def test_open_refuses_traffic_and_says_when_to_come_back(redis):
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)

    admit, reason = await breaker.allows(redis, "anthropic", now_ms=NOW + 4000)

    assert admit is False
    assert "open" in reason and "6s" in reason


async def test_stays_open_right_up_to_the_cooldown(redis):
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)

    assert await _state(redis, at=NOW + COOLDOWN_MS - 1) == State.OPEN


async def test_becomes_half_open_once_the_cooldown_elapses(redis):
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)

    assert await _state(redis, at=NOW + COOLDOWN_MS) == State.HALF_OPEN


async def test_the_transition_is_persisted_not_just_returned(redis):
    """Lazy evaluation still has to be a write, or every reader recomputes it."""
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)
    await breaker.state_of(redis, "anthropic", now_ms=NOW + COOLDOWN_MS)

    assert await redis.hget("breaker:anthropic", "state") == "half_open"


async def test_probing_starts_on_a_clean_health_window(redis):
    """The window that opened the circuit must not be there to re-open it.

    Traffic moved away the moment the breaker tripped, so nothing aged out of it.
    Left in place, the first successful probe gets judged against the previous
    outage and the circuit slams shut on evidence that is no longer true.
    """
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)
    assert await redis.zcard(health.key("anthropic")) == 4

    await breaker.state_of(redis, "anthropic", now_ms=NOW + COOLDOWN_MS)

    assert await redis.zcard(health.key("anthropic")) == 0


# --- half-open ----------------------------------------------------------------


async def _half_open(r, provider="anthropic"):
    await _fill(r, provider, outcomes=[Outcome.SERVER_ERROR] * 4)
    await breaker.state_of(r, provider, now_ms=NOW + COOLDOWN_MS)


async def test_half_open_admits_only_the_probe_ratio(redis, monkeypatch):
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    monkeypatch.setattr(breaker.random, "random", lambda: 0.05)  # under 0.1
    assert (await breaker.allows(redis, "anthropic", now_ms=at))[0] is True

    monkeypatch.setattr(breaker.random, "random", lambda: 0.5)
    admit, reason = await breaker.allows(redis, "anthropic", now_ms=at)
    assert admit is False
    assert "probe" in reason


async def test_closes_after_the_required_consecutive_probes(redis):
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    await breaker.record(redis, "anthropic", Outcome.OK, now_ms=at)
    assert await _state(redis, at=at) == State.HALF_OPEN, "one probe is not enough"

    await breaker.record(redis, "anthropic", Outcome.OK, now_ms=at)
    assert await _state(redis, at=at) == State.CLOSED


async def test_one_failed_probe_reopens_with_a_doubled_cooldown(redis):
    """Half-open is a trial, not a reprieve. One failure ends it."""
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    await breaker.record(redis, "anthropic", Outcome.TIMEOUT, now_ms=at)

    circuit = await breaker.state_of(redis, "anthropic", now_ms=at)
    assert circuit.state == State.OPEN
    assert circuit.cooldown_s == 20.0


async def test_a_probe_success_then_a_failure_does_not_count_as_progress(redis):
    """Consecutive is load-bearing: a flapping provider must not creep to closed."""
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    await breaker.record(redis, "anthropic", Outcome.OK, now_ms=at)
    await breaker.record(redis, "anthropic", Outcome.TIMEOUT, now_ms=at)

    assert await _state(redis, at=at) == State.OPEN
    assert (await breaker.state_of(redis, "anthropic", now_ms=at)).probe_successes == 0


async def test_a_caller_fault_probe_neither_closes_nor_reopens(redis):
    """Bad input tells us nothing about recovery, so it must not resolve the trial."""
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    await breaker.record(redis, "anthropic", Outcome.BAD_REQUEST, now_ms=at)

    assert await _state(redis, at=at) == State.HALF_OPEN


async def test_cooldown_doubling_stops_at_the_cap(redis):
    """A provider down for an hour should not end up with an eight-hour cooldown."""
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    for _ in range(6):
        await breaker.record(redis, "anthropic", Outcome.TIMEOUT, now_ms=at)
        at += int((await breaker.state_of(redis, "anthropic", now_ms=at)).cooldown_s * 1000)

    assert (await breaker.state_of(redis, "anthropic", now_ms=at)).cooldown_s == 40.0


async def test_closing_resets_the_cooldown_to_base(redis):
    """Otherwise a provider that recovered still carries its worst outage forever."""
    await _half_open(redis)
    at = NOW + COOLDOWN_MS

    await breaker.record(redis, "anthropic", Outcome.TIMEOUT, now_ms=at)  # cooldown -> 20s
    at += 20_000
    await breaker.state_of(redis, "anthropic", now_ms=at)  # half-open again
    await breaker.record(redis, "anthropic", Outcome.OK, now_ms=at)
    await breaker.record(redis, "anthropic", Outcome.OK, now_ms=at)

    circuit = await breaker.state_of(redis, "anthropic", now_ms=at)
    assert circuit.state == State.CLOSED
    assert circuit.cooldown_s == 10.0


# --- restart ------------------------------------------------------------------


async def test_circuit_state_survives_a_restart(redis):
    """A restarted gateway must not forget which providers it had taken out.

    Forgetting means every restart sends full traffic straight back at whatever
    was broken - and a gateway that is falling over is exactly when restarts happen.
    """
    await _fill(redis, outcomes=[Outcome.SERVER_ERROR] * 4)

    restarted = FakeRedis(decode_responses=True, connection_pool=redis.connection_pool)

    assert await _state(restarted) == State.OPEN

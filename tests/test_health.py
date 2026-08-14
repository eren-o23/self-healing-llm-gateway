"""The health window is what phase 3's breaker will trip on, so its edges get tested.

Every test drives an explicit now_ms rather than the wall clock: boundary
behaviour is the point, and a real clock would make these flaky.
"""

from __future__ import annotations

import pytest
from fakeredis.aioredis import FakeRedis

from gateway.health import record, snapshot
from gateway.providers import Outcome

WINDOW_S = 60
WINDOW_MS = WINDOW_S * 1000
NOW = 1_700_000_000_000


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


async def _fill(r, provider="anthropic", *, outcomes, latency_ms=100.0, at=NOW):
    for outcome in outcomes:
        await record(r, provider, outcome, latency_ms, window_s=WINDOW_S, now_ms=at)


async def _snap(r, provider="anthropic", at=NOW):
    return await snapshot(r, provider, window_s=WINDOW_S, now_ms=at)


async def test_empty_window_reads_as_healthy(redis):
    """Fail-open: no evidence is not evidence of sickness."""
    snap = await _snap(redis)

    assert snap.samples == 0
    assert snap.health_samples == 0
    assert snap.success_rate == 1.0
    assert snap.trippable_error_rate == 0.0
    assert (snap.p50_ms, snap.p95_ms, snap.p99_ms) == (0.0, 0.0, 0.0)


async def test_single_sample_does_not_raise(redis):
    """statistics.quantiles refuses fewer than two points; a fresh window has one."""
    await _fill(redis, outcomes=[Outcome.OK], latency_ms=42.0)

    snap = await _snap(redis)

    assert snap.samples == 1
    assert snap.p50_ms == snap.p95_ms == snap.p99_ms == 42.0


async def test_sample_inside_the_window_is_counted(redis):
    await _fill(redis, outcomes=[Outcome.OK], at=NOW - WINDOW_MS + 1)

    assert (await _snap(redis)).samples == 1


async def test_sample_outside_the_window_is_gone(redis):
    await _fill(redis, outcomes=[Outcome.OK], at=NOW - WINDOW_MS - 1)

    assert (await _snap(redis)).samples == 0


async def test_sample_exactly_on_the_boundary_is_excluded(redis):
    """Read filter and write eviction must agree, or a sample flips depending on traffic."""
    await _fill(redis, outcomes=[Outcome.OK], at=NOW - WINDOW_MS)

    assert (await _snap(redis)).samples == 0


async def test_writing_evicts_what_fell_out_of_the_window(redis):
    """The read filter hides stale samples; the write is what stops them accumulating."""
    await _fill(redis, outcomes=[Outcome.OK], at=NOW - WINDOW_MS * 5)
    assert await redis.zcard("health:anthropic") == 1

    await _fill(redis, outcomes=[Outcome.OK], at=NOW)

    assert await redis.zcard("health:anthropic") == 1, "the old sample was evicted"


async def test_percentiles_against_a_known_sample(redis):
    for latency in range(1, 101):
        await record(
            redis, "anthropic", Outcome.OK, float(latency), window_s=WINDOW_S, now_ms=NOW
        )

    snap = await _snap(redis)

    assert snap.samples == 100
    assert snap.p50_ms == pytest.approx(50.5, abs=1.0)
    assert snap.p95_ms == pytest.approx(95.0, abs=1.0)
    assert snap.p99_ms == pytest.approx(99.0, abs=1.0)


async def test_errors_are_counted_by_taxonomy_type(redis):
    await _fill(
        redis,
        outcomes=[
            Outcome.OK,
            Outcome.RATE_LIMIT,
            Outcome.RATE_LIMIT,
            Outcome.TIMEOUT,
            Outcome.CONTENT_FILTER,
        ],
    )

    snap = await _snap(redis)

    assert snap.errors_by_type == {"rate_limit": 2, "timeout": 1, "content_filter": 1}
    assert snap.samples == 5


async def test_caller_faults_are_excluded_from_the_error_rate(redis):
    """Invariant: bad input must not make a healthy provider look sick.

    Nine caller-fault failures and one success is a provider behaving perfectly.
    If these counted, phase 3 would open a breaker because somebody sent junk.
    """
    await _fill(redis, outcomes=[Outcome.OK] + [Outcome.BAD_REQUEST] * 9)

    snap = await _snap(redis)

    assert snap.samples == 10
    assert snap.health_samples == 1
    assert snap.trippable_errors == 0
    assert snap.trippable_error_rate == 0.0
    assert snap.success_rate == 1.0


async def test_caller_faults_cannot_mask_a_sick_provider(redis):
    """The inverse trap: bad input must not dilute real failures out of the rate.

    Four real failures against one success is an 80% error rate. Counting the
    fifty bad requests in the denominator would report 7% and the breaker would
    sit closed over a provider that is plainly broken.
    """
    await _fill(
        redis,
        outcomes=[Outcome.OK] + [Outcome.SERVER_ERROR] * 4 + [Outcome.BAD_REQUEST] * 50,
    )

    snap = await _snap(redis)

    assert snap.samples == 55
    assert snap.health_samples == 5
    assert snap.trippable_error_rate == pytest.approx(0.8)


async def test_caller_faults_do_not_drag_p95_down(redis):
    """A fast 400 is not evidence the provider got quicker.

    Phase 3 trips on p95 against a per-provider latency budget. Letting single
    digit millisecond caller faults into the percentiles would hide a provider
    that had slowed to a crawl.
    """
    for _ in range(10):
        await record(redis, "anthropic", Outcome.OK, 900.0, window_s=WINDOW_S, now_ms=NOW)
    for _ in range(90):
        await record(
            redis, "anthropic", Outcome.BAD_REQUEST, 3.0, window_s=WINDOW_S, now_ms=NOW
        )

    snap = await _snap(redis)

    assert snap.p95_ms == 900.0


async def test_window_survives_a_restart(redis):
    """The acceptance criterion, honestly stated: the state was never in-process.

    A second client against the same Redis is what a restarted gateway is. It
    reads a correct view of the last minute immediately rather than starting blind.
    """
    await _fill(redis, outcomes=[Outcome.OK, Outcome.OK, Outcome.TIMEOUT])

    restarted = FakeRedis(decode_responses=True, connection_pool=redis.connection_pool)
    snap = await snapshot(restarted, "anthropic", window_s=WINDOW_S, now_ms=NOW)

    assert snap.samples == 3
    assert snap.successes == 2
    assert snap.errors_by_type == {"timeout": 1}


def test_admin_health_reports_every_configured_provider(client, stub_provider):
    from tests.conftest import ADMIN_HEADERS, BODY, HEADERS, ok_result

    stub_provider(ok_result())
    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    body = client.get("/admin/health", headers=ADMIN_HEADERS).json()

    assert set(body) == {"anthropic", "openai", "groq", "ollama"}
    assert body["anthropic"]["samples"] == 1
    assert body["anthropic"]["p95_ms"] == 123.4
    assert body["groq"]["samples"] == 0, "a provider with no traffic reads as healthy"
    assert body["groq"]["success_rate"] == 1.0


async def test_providers_do_not_share_a_window(redis):
    await _fill(redis, "anthropic", outcomes=[Outcome.SERVER_ERROR] * 3)
    await _fill(redis, "groq", outcomes=[Outcome.OK])

    assert (await _snap(redis, "anthropic")).trippable_error_rate == 1.0
    assert (await _snap(redis, "groq")).trippable_error_rate == 0.0

"""The delay queue: what is due, what gets rescheduled, and where it stops.

Nothing here sleeps. Every time-dependent call takes an injected now_ms, so a
backoff that would take a minute in production is two integers apart in a test.
"""

from __future__ import annotations

import pytest
from fakeredis.aioredis import FakeRedis

from gateway import queue
from gateway.queue import Status
from gateway.schemas import RequestMetadata

pytestmark = pytest.mark.usefixtures("use_test_config")

NOW = 1_700_000_000_000
PAYLOAD = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


def meta(request_id: str = "req-1", tenant: str = "acme") -> RequestMetadata:
    return RequestMetadata(
        tenant=tenant,
        feature="nightly-summary",
        request_id=request_id,
        request_class="batch.generate",
    )


# --- backoff ------------------------------------------------------------------


def test_the_backoff_ceiling_doubles_and_then_caps():
    """Full jitter means the delay is random, so the ceiling is what to assert on."""
    ceilings = [
        max(queue.retry_delay_s(n, base_s=1.0, cap_s=8.0) for _ in range(200))
        for n in range(6)
    ]

    # Doubling while under the cap, flat once it is reached.
    assert ceilings[0] < ceilings[1] < ceilings[2] < ceilings[3]
    assert ceilings[3] <= 8.0 and ceilings[4] <= 8.0 and ceilings[5] <= 8.0


def test_jitter_stays_inside_its_window_and_actually_varies():
    draws = [queue.retry_delay_s(2, base_s=1.0, cap_s=60.0) for _ in range(200)]

    assert all(0.0 <= d <= 4.0 for d in draws)  # min(1 * 2**2, 60)
    # The variance is the point: without it every queued job returns at once.
    assert len(set(draws)) > 100


def test_the_cap_binds_at_high_attempt_counts():
    assert all(
        queue.retry_delay_s(20, base_s=1.0, cap_s=60.0) <= 60.0 for _ in range(200)
    )


# --- enqueue and claim --------------------------------------------------------


async def test_a_queued_job_keeps_its_payload_and_attribution(redis):
    job_id = await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)

    job = await queue.get(redis, job_id)
    assert job is not None
    assert job.status is Status.QUEUED
    assert job.payload == PAYLOAD
    assert job.attempts == 0
    # Without this the drained call bills a tenant nobody can name.
    assert job.meta.tenant == "acme"
    assert job.meta.request_class == "batch.generate"
    assert await queue.depth(redis) == 1


async def test_popping_a_due_job_claims_it(redis):
    job_id = await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)

    job = await queue.pop_due(redis, now_ms=NOW)
    assert job is not None and job.id == job_id
    assert (await queue.get(redis, job_id)).status is Status.RUNNING
    assert await queue.depth(redis) == 0


async def test_a_job_that_is_not_due_yet_is_left_alone(redis):
    job_id = await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)
    job = await queue.pop_due(redis, now_ms=NOW)
    await queue.retry(redis, job, delay_s=5.0, error="server_error", now_ms=NOW)

    assert await queue.pop_due(redis, now_ms=NOW + 4_000) is None
    # Put back, not dropped.
    assert await queue.depth(redis) == 1
    assert (await queue.pop_due(redis, now_ms=NOW + 6_000)).id == job_id


async def test_an_empty_queue_pops_nothing(redis):
    assert await queue.pop_due(redis, now_ms=NOW) is None


async def test_the_earliest_job_comes_out_first(redis):
    late = await queue.enqueue(redis, meta("late"), PAYLOAD, now_ms=NOW + 1_000)
    early = await queue.enqueue(redis, meta("early"), PAYLOAD, now_ms=NOW)

    assert (await queue.pop_due(redis, now_ms=NOW + 2_000)).id == early
    assert (await queue.pop_due(redis, now_ms=NOW + 2_000)).id == late


# --- outcomes -----------------------------------------------------------------


async def test_retry_reschedules_later_and_counts_the_attempt(redis):
    job_id = await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)
    job = await queue.pop_due(redis, now_ms=NOW)

    ready_at = await queue.retry(redis, job, delay_s=2.5, error="timeout", now_ms=NOW)

    assert ready_at == NOW + 2_500
    reloaded = await queue.get(redis, job_id)
    assert reloaded.status is Status.QUEUED
    assert reloaded.attempts == 1
    assert reloaded.error == "timeout"


async def test_completion_stores_the_response(redis):
    job_id = await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)
    job = await queue.pop_due(redis, now_ms=NOW)

    await queue.complete(redis, job, {"id": "chatcmpl-1", "choices": []}, now_ms=NOW)

    done = await queue.get(redis, job_id)
    assert done.status is Status.DONE
    assert done.result == {"id": "chatcmpl-1", "choices": []}
    assert done.error is None
    assert await queue.depth(redis) == 0


async def test_a_dead_lettered_job_is_inspectable(redis):
    job_id = await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)
    job = await queue.pop_due(redis, now_ms=NOW)

    await queue.dead_letter(redis, job, "out of attempts", now_ms=NOW)

    assert (await queue.get(redis, job_id)).status is Status.DEAD
    assert [j.id for j in await queue.dead(redis)] == [job_id]


async def test_a_terminally_failed_job_stays_out_of_the_dlq(redis):
    """FAILED is bad input, DEAD is a survived outage. Mixing them blinds the DLQ."""
    await queue.enqueue(redis, meta(), PAYLOAD, now_ms=NOW)
    job = await queue.pop_due(redis, now_ms=NOW)

    await queue.fail(redis, job, "bad_request", now_ms=NOW)

    assert (await queue.get(redis, job.id)).status is Status.FAILED
    assert await queue.dead(redis) == []


async def test_an_unknown_job_id_is_none(redis):
    assert await queue.get(redis, "nope") is None

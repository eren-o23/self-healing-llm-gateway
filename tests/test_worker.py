"""One tick of the drain: what completes, what comes back, and what stops.

run_once() exists precisely so these tests do not have to start and stop a
`while True` to watch a single job go through.
"""

from __future__ import annotations

import time

import pytest
from fakeredis.aioredis import FakeRedis
from prometheus_client import REGISTRY

from gateway import breaker, queue, worker
from gateway.providers import Outcome
from gateway.queue import Status
from gateway.schemas import RequestMetadata
from tests.conftest import failed_result, ok_result

pytestmark = pytest.mark.usefixtures("use_test_config", "all_keys_set")

PAYLOAD = {"messages": [{"role": "user", "content": "hello"}]}


def _value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


def meta(request_class: str = "batch.generate") -> RequestMetadata:
    return RequestMetadata(
        tenant="acme",
        feature="nightly-summary",
        request_id="req-1",
        request_class=request_class,
    )


def all_broken(outcome: Outcome = Outcome.SERVER_ERROR) -> dict:
    return {
        name: failed_result(name, outcome)
        for name in ("anthropic", "openai", "groq", "ollama")
    }


async def test_an_empty_queue_is_not_work(redis, config):
    assert await worker.run_once(redis, config) is False


async def test_a_successful_job_stores_the_openai_response(redis, config, stub_ladder):
    stub_ladder({})
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)

    assert await worker.run_once(redis, config) is True

    job = await queue.get(redis, job_id)
    assert job.status is Status.DONE
    assert job.result["choices"][0]["message"]["content"] == "hi there"
    # Attribution survived the round trip through Redis.
    assert job.result["x_gateway"]["tenant"] == "acme"
    assert await queue.depth(redis) == 0


async def test_an_exhausted_ladder_comes_back_for_another_go(redis, config, stub_ladder):
    calls = stub_ladder(all_broken())
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)

    await worker.run_once(redis, config)

    job = await queue.get(redis, job_id)
    assert job.status is Status.QUEUED
    assert job.attempts == 1
    assert "server_error" in job.error
    assert await queue.depth(redis) == 1  # requeued, not lost
    assert len(calls) == 4  # it did walk the whole ladder first


async def test_retries_are_exhausted_into_the_dlq(redis, config, stub_ladder):
    """TEST_CONFIG allows three attempts, so the third is the last."""
    stub_ladder(all_broken())
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)

    for _ in range(3):
        # pop_due only yields a job whose delay has elapsed, and the delays are
        # jittered - pull the score forward rather than sleeping on a coin flip.
        await redis.zadd(queue.READY, {job_id: 0})
        await worker.run_once(redis, config)

    job = await queue.get(redis, job_id)
    assert job.status is Status.DEAD
    assert job.attempts == 2  # two retries, then the third attempt buried it
    assert [j.id for j in await queue.dead(redis)] == [job_id]
    assert await queue.depth(redis) == 0


async def test_bad_input_fails_terminally_and_is_never_retried(redis, config, stub_ladder):
    """Every remaining rung rejects it identically; retrying just bills for it again."""
    calls = stub_ladder(all_broken(Outcome.BAD_REQUEST))
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)

    await worker.run_once(redis, config)

    job = await queue.get(redis, job_id)
    assert job.status is Status.FAILED
    assert await queue.depth(redis) == 0
    assert await queue.dead(redis) == []  # not an outage casualty
    assert calls == ["anthropic"]  # the walk stopped at the first rung too


async def test_a_job_whose_class_no_longer_exists_is_not_retried_forever(
    redis, config, stub_ladder
):
    stub_ladder({})
    job_id = await queue.enqueue(redis, meta("class.that.was.removed"), PAYLOAD)

    await worker.run_once(redis, config)

    job = await queue.get(redis, job_id)
    assert job.status is Status.FAILED
    assert "no available provider" in job.error
    assert await queue.depth(redis) == 0


async def test_a_content_filter_is_the_callers_fault_not_a_retry(
    redis, config, stub_ladder
):
    stub_ladder(all_broken(Outcome.CONTENT_FILTER))
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)

    await worker.run_once(redis, config)

    assert (await queue.get(redis, job_id)).status is Status.FAILED


async def test_a_job_that_recovers_completes_on_a_later_attempt(
    redis, config, monkeypatch
):
    """The point of the whole phase: the provider comes back and the work lands."""
    broken = all_broken()
    state = {"healthy": False}

    async def fake_call(name, provider, payload, chaos=None):
        return ok_result(name) if state["healthy"] else broken[name]

    monkeypatch.setattr("gateway.router.call_provider", fake_call)
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)

    await worker.run_once(redis, config)
    assert (await queue.get(redis, job_id)).status is Status.QUEUED

    state["healthy"] = True
    await redis.zadd(queue.READY, {job_id: 0})
    await worker.run_once(redis, config)

    assert (await queue.get(redis, job_id)).status is Status.DONE


async def test_the_queue_collectors_move(redis, config, stub_ladder):
    """Deltas, not absolutes: prometheus_client's registry is a process global."""
    stub_ladder(all_broken())
    before_retries = _value("gateway_queue_retries_total", attempt="1")
    before_dlq = _value("gateway_dlq_total")
    before_dead = _value("gateway_job_age_seconds_count", status="dead")

    job_id = await queue.enqueue(redis, meta(), PAYLOAD)
    for _ in range(3):
        await redis.zadd(queue.READY, {job_id: 0})
        await worker.run_once(redis, config)

    assert _value("gateway_queue_retries_total", attempt="1") == before_retries + 1
    assert _value("gateway_dlq_total") == before_dlq + 1
    assert _value("gateway_job_age_seconds_count", status="dead") == before_dead + 1
    assert _value("gateway_queue_depth") == 0.0


async def test_an_open_circuit_does_not_spend_an_attempt(redis, config, stub_ladder):
    """Nothing was tried, so nothing was learned - and nothing should be spent.

    A breaker cooldown outlives a whole retry budget. Counting these dead-letters
    every job without a single provider call, during exactly the outage the queue
    exists to survive.
    """
    calls = stub_ladder({})
    job_id = await queue.enqueue(redis, meta(), PAYLOAD)
    # Opened against the real clock, because run_once() reads it - a fixed NOW
    # would sit a cooldown in the past and admit a probe on the first tick.
    now_ms = int(time.time() * 1000)
    for name in ("anthropic", "openai", "groq", "ollama"):
        await breaker._open(redis, name, now_ms, 3600.0, "test")

    for _ in range(10):
        await redis.zadd(queue.READY, {job_id: 0})
        await worker.run_once(redis, config)

    job = await queue.get(redis, job_id)
    assert job.status is Status.QUEUED
    assert job.attempts == 0
    assert calls == []
    assert await queue.dead(redis) == []


async def test_the_capacity_re_check_ignores_the_attempt_count(
    redis, config, stub_ladder, monkeypatch
):
    """Attempts count past provider failures, not how soon a circuit reopens.

    Scaling the re-check by them left a job that had failed six times waiting up
    to a minute after capacity was already back.
    """
    stub_ladder({})
    delays: list[int] = []
    monkeypatch.setattr(
        queue, "retry_delay_s", lambda attempt, base, cap: delays.append(attempt) or 0.0
    )

    job_id = await queue.enqueue(redis, meta(), PAYLOAD)
    await redis.hset(f"job:{job_id}", "attempts", "6")
    now_ms = int(time.time() * 1000)
    for name in ("anthropic", "openai", "groq", "ollama"):
        await breaker._open(redis, name, now_ms, 3600.0, "test")

    await worker.run_once(redis, config)

    assert delays == [0]

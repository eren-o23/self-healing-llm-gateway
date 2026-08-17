"""The process that drains the queue. A second service, not a background task.

Running this inside the API process would tie the retry path's lifetime to the
API's: a deploy or a crash would take the drain down with it, and a backlog of
long retries would compete with live requests for the same event loop. As its
own compose service the two scale and fail independently, which is the claim
the queue is there to make.

It calls get_redis() and get_config() directly - no FastAPI anywhere. Importing
gateway.config is also what runs load_dotenv(), which is the only reason this
process can see provider keys at all.

Routing goes through the same router.route() the API uses, so a queued job gets
the same ladder, the same circuit breakers and the same chaos injection. A
retry path with its own private notion of how to call a provider is a retry
path that behaves differently from the thing it is retrying.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from prometheus_client import start_http_server
from redis.asyncio import Redis

from gateway import metrics, queue, router
from gateway.config import GatewayConfig, get_config, get_redis
from gateway.providers import ProviderCallLog
from gateway.queue import Job, Status
from gateway.schemas import to_openai_response

log = logging.getLogger("gateway.worker")

METRICS_PORT = int(os.getenv("WORKER_METRICS_PORT", "8001"))


async def run_once(r: Redis, config: GatewayConfig) -> bool:
    """Handle at most one due job. True if there was one.

    Factored out of the loop so tests can drive a single tick; a test that has to
    start and stop a `while True` to observe one job is a test nobody writes.
    """
    metrics.QUEUE_DEPTH.set(await queue.depth(r))

    job = await queue.pop_due(r)
    if job is None:
        return False

    try:
        await _run(r, config, job)
    finally:
        metrics.QUEUE_DEPTH.set(await queue.depth(r))
    return True


async def _run(r: Redis, config: GatewayConfig, job: Job) -> None:
    ladder = config.ladder_for(job.meta.request_class)
    if not ladder:
        # Not retryable: no amount of waiting adds an API key.
        await _finish(r, job, Status.FAILED, f"no available provider for {job.meta.request_class}")
        return

    routed = await router.route(r, config, job.meta, job.payload, ladder)
    result = routed.result

    if result is not None and result.outcome.ok:
        body = to_openai_response(result, job.meta, routed.attempts).model_dump(mode="json")
        await queue.complete(r, job, body)
        _observe_age(job, Status.DONE)
        log.info(
            "job %s done provider=%s attempts=%d cost_usd=%.6f",
            job.id,
            result.provider,
            job.attempts + 1,
            result.cost_usd,
        )
        return

    if result is not None and not result.outcome.trippable:
        # The caller sent something every provider will reject identically.
        # Retrying it max_attempts times bills for the same 400 over and over -
        # the same rule that keeps bad input from opening a breaker and from
        # walking the ladder, now pointed at the queue.
        await _finish(r, job, Status.FAILED, f"{result.outcome}: {result.error}")
        return

    last = routed.last_failure
    await _retry_or_bury(r, config, job, _reason(last))


async def _retry_or_bury(
    r: Redis, config: GatewayConfig, job: Job, reason: str
) -> None:
    cfg = config.queue
    tries_used = job.attempts + 1

    if tries_used >= cfg.max_attempts:
        await queue.dead_letter(r, job, reason)
        metrics.DLQ.inc()
        _observe_age(job, Status.DEAD)
        log.error("job %s dead-lettered after %d attempts: %s", job.id, tries_used, reason)
        return

    delay_s = queue.retry_delay_s(job.attempts, cfg.base_delay_s, cfg.max_delay_s)
    await queue.retry(r, job, delay_s, reason)
    metrics.QUEUE_RETRIES.labels(attempt=str(tries_used)).inc()
    log.info(
        "job %s attempt %d/%d failed (%s); retrying in %.2fs",
        job.id,
        tries_used,
        cfg.max_attempts,
        reason,
        delay_s,
    )


async def _finish(r: Redis, job: Job, status: Status, reason: str) -> None:
    await queue.fail(r, job, reason)
    _observe_age(job, status)
    log.warning("job %s failed permanently: %s", job.id, reason)


def _reason(last: ProviderCallLog | None) -> str:
    """Why this attempt failed. None means every circuit was open."""
    return f"{last.outcome}: {last.error}" if last else "no provider available"


def _observe_age(job: Job, status: Status) -> None:
    age_s = max(0.0, time.time() - job.created_at_ms / 1000)
    metrics.JOB_AGE.labels(status=str(status)).observe(age_s)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    config = get_config()
    config.log_startup_state()

    # Own registry, own port: this is a different process from the API, so its
    # queue counters cannot appear on the gateway's /metrics. Prometheus scrapes
    # it as a second target.
    start_http_server(METRICS_PORT)
    log.info(
        "worker up; metrics on :%d, polling every %.1fs",
        METRICS_PORT,
        config.queue.poll_interval_s,
    )

    r = get_redis()
    while True:
        try:
            worked = await run_once(r, config)
        except Exception:  # noqa: BLE001 - one bad job must not stop the drain
            log.exception("worker tick failed")
            worked = False
        if not worked:
            await asyncio.sleep(config.queue.poll_interval_s)


if __name__ == "__main__":
    asyncio.run(main())

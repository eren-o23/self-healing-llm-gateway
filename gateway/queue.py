"""A Redis ZSET as a delay queue: one structure for the first enqueue and every retry.

Scoring by `ready_at` epoch ms is what collapses those two into one thing. A
plain list would need a second timer structure for backoff, and then a story for
keeping the two in step; here a retry is the same ZADD with a later score.

`ZPOPMIN` is atomic, so more than one worker can drain this safely. What it does
not give is a claim that survives the worker dying mid-job.

# ponytail: crash between ZPOPMIN and completion loses the job - acceptable for a
# demo; upgrade path is Redis Streams with consumer groups and XACK
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from gateway.config import get_config
from gateway.schemas import RequestMetadata

if TYPE_CHECKING:
    from redis.asyncio import Redis

READY = "queue:ready"
DLQ = "queue:dlq"


class Status(StrEnum):
    """Where a job is.

    FAILED and DEAD are deliberately different. DEAD is work that ran out of
    retries and belongs in the dead-letter queue somebody looks at. FAILED is
    input the provider was right to reject, which was never going to work and
    was never retried - filing it alongside the outage casualties would make the
    DLQ useless as a signal.
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


@dataclass(frozen=True, slots=True)
class Job:
    """One deferred request, as both the worker and /v1/jobs/{id} read it."""

    id: str
    status: Status
    meta: RequestMetadata
    payload: dict[str, Any]
    attempts: int = 0
    created_at_ms: int = 0
    updated_at_ms: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        """What a caller polling for this job gets back."""
        return {
            "job_id": self.id,
            "status": str(self.status),
            "request_class": self.meta.request_class,
            "request_id": self.meta.request_id,
            "attempts": self.attempts,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "response": self.result,
            "error": self.error,
        }


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def retry_delay_s(attempt: int, base_s: float, cap_s: float) -> float:
    """Full jitter: uniform over [0, min(base * 2**attempt, cap)].

    Full rather than equal jitter on purpose. The failure this guards against is
    a provider coming back and being knocked over by the entire backlog arriving
    in the same instant, and only the variance actually spreads that out - the
    exponential part alone just moves the stampede later.
    """
    return random.uniform(0, min(base_s * 2**attempt, cap_s))


async def enqueue(
    r: Redis, meta: RequestMetadata, payload: dict[str, Any], *, now_ms: int | None = None
) -> str:
    """Accept work for later, ready immediately. Returns the job id."""
    now_ms = _now_ms() if now_ms is None else now_ms
    job_id = uuid.uuid4().hex

    pipe = r.pipeline()
    pipe.hset(
        _key(job_id),
        mapping={
            "id": job_id,
            "status": Status.QUEUED,
            # The full attribution travels with the job. The worker rebuilds it
            # to route, and without it the drained call lands in the cost
            # counters with no tenant - which is the one thing the API boundary
            # refuses to let a synchronous request do.
            "meta": meta.model_dump_json(),
            "payload": json.dumps(payload),
            "attempts": "0",
            "created_at_ms": str(now_ms),
            "updated_at_ms": str(now_ms),
            "result": "",
            "error": "",
        },
    )
    pipe.expire(_key(job_id), get_config().queue.job_ttl_s)
    pipe.zadd(READY, {job_id: now_ms})
    await pipe.execute()
    return job_id


async def get(r: Redis, job_id: str) -> Job | None:
    raw = await r.hgetall(_key(job_id))
    if not raw:
        return None
    return Job(
        id=raw["id"],
        status=Status(raw["status"]),
        meta=RequestMetadata.model_validate_json(raw["meta"]),
        payload=json.loads(raw["payload"]),
        attempts=int(raw["attempts"]),
        created_at_ms=int(raw["created_at_ms"]),
        updated_at_ms=int(raw["updated_at_ms"]),
        result=json.loads(raw["result"]) if raw["result"] else None,
        error=raw["error"] or None,
    )


async def pop_due(r: Redis, *, now_ms: int | None = None) -> Job | None:
    """Claim the earliest job if it is due, else leave the queue as it was.

    ZPOPMIN pops the minimum score, so a queue whose earliest job is not yet due
    has nothing due at all - the not-yet-due case puts it straight back and the
    worker goes round again. Popping to look is the price of doing the check
    atomically.
    """
    now_ms = _now_ms() if now_ms is None else now_ms

    popped = await r.zpopmin(READY)
    if not popped:
        return None

    job_id, ready_at_ms = popped[0]
    if ready_at_ms > now_ms:
        await r.zadd(READY, {job_id: ready_at_ms})
        return None

    job = await get(r, job_id)
    if job is None:
        return None  # the hash aged out from under the index; nothing to run

    await _update(r, job_id, now_ms=now_ms, status=Status.RUNNING)
    return job


async def retry(
    r: Redis,
    job: Job,
    delay_s: float,
    error: str,
    *,
    count: bool = True,
    now_ms: int | None = None,
) -> int:
    """Put the job back with a later score. Returns when it becomes due.

    `count=False` reschedules without spending an attempt, for the case where
    nothing was actually tried. See worker._defer for why that distinction has
    to exist.
    """
    now_ms = _now_ms() if now_ms is None else now_ms
    ready_at_ms = now_ms + int(delay_s * 1000)
    await _update(
        r,
        job.id,
        now_ms=now_ms,
        status=Status.QUEUED,
        attempts=job.attempts + 1 if count else job.attempts,
        error=error,
    )
    await r.zadd(READY, {job.id: ready_at_ms})
    return ready_at_ms


async def complete(
    r: Redis, job: Job, result: dict[str, Any], *, now_ms: int | None = None
) -> None:
    await _update(
        r, job.id, now_ms=now_ms, status=Status.DONE, result=json.dumps(result), error=""
    )


async def fail(r: Redis, job: Job, error: str, *, now_ms: int | None = None) -> None:
    """Terminal, and not retried: the request itself is what the provider rejected."""
    await _update(r, job.id, now_ms=now_ms, status=Status.FAILED, error=error)


async def dead_letter(
    r: Redis, job: Job, error: str, *, now_ms: int | None = None
) -> None:
    now_ms = _now_ms() if now_ms is None else now_ms
    await _update(r, job.id, now_ms=now_ms, status=Status.DEAD, error=error)
    await r.zadd(DLQ, {job.id: now_ms})


async def depth(r: Redis) -> int:
    return await r.zcard(READY)


async def dead(r: Redis, *, limit: int = 100) -> list[Job]:
    """The dead-letter queue, newest last. Jobs whose hash has aged out are skipped."""
    ids = await r.zrange(DLQ, 0, limit - 1)
    jobs = [await get(r, job_id) for job_id in ids]
    return [job for job in jobs if job is not None]


async def _update(
    r: Redis, job_id: str, *, now_ms: int | None = None, **fields: Any
) -> None:
    fields["updated_at_ms"] = _now_ms() if now_ms is None else now_ms
    await r.hset(_key(job_id), mapping={k: str(v) for k, v in fields.items()})

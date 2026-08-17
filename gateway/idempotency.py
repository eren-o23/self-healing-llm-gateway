"""Idempotency-Key handling: replay a stored response instead of billing a second call.

This ships before the retry queue exists, and the ordering is not cosmetic. A
retry is a duplicate call by construction, so building retries first is exactly
how a network blip turns into duplicated side effects. The guard has to be in
place before anything is allowed to try twice.

The key is namespaced by tenant. A bare `Idempotency-Key: abc` is a string the
caller chose, and two tenants will pick the same one - without the namespace,
one tenant's replay hands back the other's response.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

IN_FLIGHT = "in_flight"


def _key(tenant: str, key: str) -> str:
    return f"idem:{tenant}:{key}"


async def claim(r: Redis, tenant: str, key: str, *, ttl_s: int) -> str | None:
    """Take ownership of the key, or report what is already stored under it.

    Returns None when the caller now owns the key and should do the work.
    Otherwise returns the stored value: IN_FLIGHT for a request still running, or
    the JSON envelope of a finished one.

    SET NX is what makes two concurrent requests with the same key safe - the
    check and the claim are one round trip, so there is no window for both to
    decide they are first.
    """
    if await r.set(_key(tenant, key), IN_FLIGHT, nx=True, ex=ttl_s):
        return None
    # A None here means the key expired between the failed SET and this GET.
    # Reporting it as in-flight is the conservative read: better a spurious 409
    # the caller retries than a duplicate call it cannot take back.
    return await r.get(_key(tenant, key)) or IN_FLIGHT


async def store(
    r: Redis, tenant: str, key: str, status: int, body: Any, *, ttl_s: int
) -> None:
    """Record the finished response so a replay never reaches a provider.

    The status travels with the body because they are replayed together: a
    deferrable request that was accepted must replay as 202 with its original
    job id, not as a 200.
    """
    await r.set(
        _key(tenant, key), json.dumps({"status": status, "body": body}), ex=ttl_s
    )


async def release(r: Redis, tenant: str, key: str) -> None:
    """Drop the claim so the caller can try again.

    Called on every failure path. Leaving a failed request's key in place would
    pin a transient 502 for the full TTL and answer every retry with a 409 - and
    intending to retry is the entire reason a client sends an idempotency key.
    """
    await r.delete(_key(tenant, key))

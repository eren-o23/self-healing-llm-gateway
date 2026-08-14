"""Fault injection, so failover can be demonstrated instead of described.

Pulled forward from phase 5: testing a breaker without a way to break something
on demand is needlessly painful, and the demo needs this anyway.

The injected fault raises a real LiteLLM exception at the real dispatch point,
so it travels the same path a genuine failure would - through
classify_exception(), into the health window, into the breaker. Synthesising a
ProviderResult instead would mean the demo exercised a code path that production
traffic never touches, which is the sort of test that passes right up until it
matters.

State is in Redis with a TTL, so an abandoned experiment expires on its own
rather than leaving a provider mysteriously broken.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from litellm import exceptions as llm_exc

from gateway.providers import Outcome

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = logging.getLogger(__name__)

# Which exception to raise for a requested outcome. Deliberately the real classes
# rather than a shortcut into the taxonomy: injecting an exception the taxonomy
# then has to classify is what proves the classification still works.
_EXCEPTION_FOR: dict[Outcome, type[Exception]] = {
    Outcome.RATE_LIMIT: llm_exc.RateLimitError,
    Outcome.TIMEOUT: llm_exc.Timeout,
    Outcome.SERVER_ERROR: llm_exc.InternalServerError,
    Outcome.AUTH: llm_exc.AuthenticationError,
    Outcome.CONTENT_FILTER: llm_exc.ContentPolicyViolationError,
    Outcome.BAD_REQUEST: llm_exc.BadRequestError,
}

INJECTABLE = frozenset(_EXCEPTION_FOR)


@dataclass(frozen=True, slots=True)
class ChaosSpec:
    provider: str
    error_rate: float = 0.0
    latency_ms: int = 0
    error_type: str = Outcome.SERVER_ERROR

    async def apply(self) -> None:
        """Add the requested latency, then fail with the requested probability.

        Latency first, and unconditionally: a provider that has got slow and a
        provider that has started failing are different symptoms, and the p95
        trip path needs the former without the latter.
        """
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self.error_rate and random.random() < self.error_rate:
            outcome = Outcome(self.error_type)
            raise _EXCEPTION_FOR[outcome](
                f"injected {outcome} for {self.provider}",
                llm_provider=self.provider,
                model="chaos",
            )


def _key(provider: str) -> str:
    return f"chaos:{provider}"


async def get(r: Redis, provider: str) -> ChaosSpec | None:
    raw = await r.get(_key(provider))
    return ChaosSpec(**json.loads(raw)) if raw else None


async def set_(r: Redis, spec: ChaosSpec, ttl_s: int) -> None:
    log.warning("chaos ENABLED for %s: %s (ttl %ds)", spec.provider, spec, ttl_s)
    await r.set(_key(spec.provider), json.dumps(asdict(spec)), ex=ttl_s)


async def clear(r: Redis, provider: str) -> bool:
    log.warning("chaos cleared for %s", provider)
    return bool(await r.delete(_key(provider)))


async def active(r: Redis, providers: list[str]) -> dict[str, ChaosSpec]:
    """Every provider currently under injected faults, for /admin/chaos."""
    specs = {p: await get(r, p) for p in providers}
    return {p: spec for p, spec in specs.items() if spec is not None}

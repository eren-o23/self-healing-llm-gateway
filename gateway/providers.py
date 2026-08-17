"""Provider dispatch via LiteLLM, plus the error taxonomy everything downstream depends on.

LiteLLM is used as an adapter and cost table only. Its Router - with built-in
fallbacks and retries - is deliberately unused, because the routing, breaking and
healing is what this project exists to build.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import litellm
from litellm import exceptions as llm_exc

from gateway.config import ProviderConfig

log = logging.getLogger(__name__)

# Chatty by default, and it re-prints provider errors we already classify.
litellm.suppress_debug_info = True


class Outcome(StrEnum):
    """Normalized result of a provider call.

    `trippable` decides whether an outcome counts against a provider's health.
    CONTENT_FILTER and BAD_REQUEST are the caller's fault: letting them open a
    breaker would take a healthy provider offline because somebody sent bad
    input. That distinction is the most commonly botched part of a gateway.
    """

    OK = "ok"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    CONTENT_FILTER = "content_filter"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"

    @property
    def trippable(self) -> bool:
        return self in _TRIPPABLE

    @property
    def ok(self) -> bool:
        return self is Outcome.OK


_TRIPPABLE = frozenset(
    {Outcome.RATE_LIMIT, Outcome.TIMEOUT, Outcome.SERVER_ERROR, Outcome.AUTH}
)

# Order matters: the first matching entry wins, so subclasses precede their bases.
# ContextWindowExceededError subclasses BadRequestError; both are the caller's
# fault and neither should trip a breaker.
_EXCEPTION_TAXONOMY: tuple[tuple[type[Exception], Outcome], ...] = (
    (llm_exc.RateLimitError, Outcome.RATE_LIMIT),
    (llm_exc.Timeout, Outcome.TIMEOUT),
    (llm_exc.APIConnectionError, Outcome.TIMEOUT),
    (llm_exc.AuthenticationError, Outcome.AUTH),
    (llm_exc.PermissionDeniedError, Outcome.AUTH),
    (llm_exc.ContentPolicyViolationError, Outcome.CONTENT_FILTER),
    (llm_exc.ContextWindowExceededError, Outcome.BAD_REQUEST),
    (llm_exc.NotFoundError, Outcome.BAD_REQUEST),
    (llm_exc.UnprocessableEntityError, Outcome.BAD_REQUEST),
    (llm_exc.BadRequestError, Outcome.BAD_REQUEST),
    (llm_exc.ServiceUnavailableError, Outcome.SERVER_ERROR),
    (llm_exc.InternalServerError, Outcome.SERVER_ERROR),
    (llm_exc.BadGatewayError, Outcome.SERVER_ERROR),
    (llm_exc.APIError, Outcome.SERVER_ERROR),
)


def classify_exception(exc: BaseException) -> Outcome:
    """Map any provider failure onto the taxonomy.

    Unknown exceptions become SERVER_ERROR: a failure we cannot explain is still
    a failure, and treating it as harmless would hide a genuinely sick provider.
    """
    if isinstance(exc, TimeoutError):
        return Outcome.TIMEOUT
    for exc_type, outcome in _EXCEPTION_TAXONOMY:
        if isinstance(exc, exc_type):
            return outcome
    log.warning("unclassified provider exception %s: %s", type(exc).__name__, exc)
    return Outcome.SERVER_ERROR


@dataclass(slots=True)
class ProviderResult:
    provider: str
    model: str
    outcome: Outcome
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    response: Any | None = None
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ProviderCallLog:
    """One rung of the ladder, for the x_gateway.attempts trail."""

    provider: str
    outcome: str
    latency_ms: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _cost_of(response: Any, model: str) -> float:
    """Cost in USD, or 0.0 when the model has no published price.

    The configured model name is passed explicitly rather than letting LiteLLM
    infer it from the response. Providers echo back their own bare model id -
    groq returns "llama-3.3-70b-versatile", but only "groq/llama-3.3-70b-versatile"
    is in the price table - so inference silently yields $0 for a model that is
    genuinely billed. Silent zero-cost is the failure mode that guts the whole
    point of a gateway, so the caller's configured name wins.

    The provider is resolved explicitly for the same reason the model name is.
    completion_cost() otherwise sniffs it from the string, and a model id that
    itself contains a slash defeats that: "groq/openai/gpt-oss-120b" reads as
    provider "openai", which is not where the price lives, so the lookup raises
    and the except below turns a billed call into $0. The price table has the
    key - only the routing to it was wrong.

    Self-hosted models (ollama) are genuinely free; unknown hosted models are a
    gap in LiteLLM's table. Both must return a number rather than raising - a
    pricing gap is not a reason to fail a request that already succeeded.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:  # noqa: BLE001 - unknown prefix; let completion_cost try
        provider = None

    try:
        return float(
            litellm.completion_cost(
                completion_response=response, model=model, custom_llm_provider=provider
            )
            or 0.0
        )
    except Exception:  # noqa: BLE001 - any pricing failure degrades to zero
        return 0.0


async def call_provider(
    name: str,
    provider: ProviderConfig,
    payload: dict[str, Any],
    chaos: Any | None = None,
) -> ProviderResult:
    """Dispatch one request to one provider. Never raises for provider failures.

    An injected fault (chaos.ChaosSpec) is applied inside the same try and inside
    the same timing span as the real call, so it is indistinguishable downstream
    from a genuine one: same classification, same latency accounting, same health
    window. The alternative - synthesising a failed ProviderResult somewhere up
    the stack - would demo a code path that real traffic never takes.

    Typed as Any to keep this module free of a Redis-aware import; the router
    fetches the spec and hands it down.
    """
    started = time.perf_counter()

    kwargs: dict[str, Any] = {
        **payload,
        "model": provider.model,
        "timeout": provider.timeout_s,
    }
    if provider.api_key:
        kwargs["api_key"] = provider.api_key
    if provider.api_base:
        kwargs["api_base"] = provider.api_base

    try:
        if chaos is not None:
            await chaos.apply()
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - classified, not swallowed
        outcome = classify_exception(exc)
        return ProviderResult(
            provider=name,
            model=provider.model,
            outcome=outcome,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )

    usage = getattr(response, "usage", None)
    return ProviderResult(
        provider=name,
        model=provider.model,
        outcome=Outcome.OK,
        latency_ms=(time.perf_counter() - started) * 1000,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=_cost_of(response, provider.model),
        response=response,
    )

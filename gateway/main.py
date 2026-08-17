"""FastAPI app: one service every model call routes through."""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from gateway import breaker, chaos, health, idempotency, router
from gateway.config import GatewayConfig, get_config, get_redis
from gateway.providers import Outcome, ProviderCallLog, ProviderResult
from gateway.schemas import (
    ChaosRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    GatewayMeta,
    RequestMetadata,
    Usage,
)

log = logging.getLogger(__name__)

# How a provider outcome surfaces to the caller. AUTH is a 502 on purpose: a
# missing or rejected provider key is the gateway's problem, not the caller's.
_STATUS_BY_OUTCOME: dict[Outcome, int] = {
    Outcome.RATE_LIMIT: 429,
    Outcome.TIMEOUT: 504,
    Outcome.SERVER_ERROR: 502,
    Outcome.AUTH: 502,
    Outcome.CONTENT_FILTER: 400,
    Outcome.BAD_REQUEST: 400,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_config().log_startup_state()
    yield


app = FastAPI(
    title="Self-Healing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


def require_metadata(
    config: Annotated[GatewayConfig, Depends(get_config)],
    x_tenant_id: Annotated[str, Header(min_length=1)],
    x_feature: Annotated[str, Header(min_length=1)],
    x_request_id: Annotated[str, Header(min_length=1)],
    x_request_class: Annotated[str | None, Header()] = None,
) -> RequestMetadata:
    """Reject unattributable traffic at the boundary.

    Without tenant and feature, cost can never be attributed - and retrofitting
    required fields onto a running gateway is the migration nobody wants. So the
    headers are mandatory from the first request the service ever serves.
    """
    request_class = x_request_class or config.default_class
    if request_class not in config.classes:
        raise HTTPException(
            status_code=400,
            detail=f"unknown request class {request_class!r}; "
            f"known: {sorted(config.classes)}",
        )
    return RequestMetadata(
        tenant=x_tenant_id,
        feature=x_feature,
        request_id=x_request_id,
        request_class=request_class,
    )


@app.get("/healthz")
async def healthz(config: Annotated[GatewayConfig, Depends(get_config)]) -> dict[str, Any]:
    return {
        "status": "ok",
        "providers": {
            name: ("available" if p.available else "disabled")
            for name, p in config.providers.items()
        },
        "classes": {name: config.ladder_for(name) for name in config.classes},
    }


def require_admin(x_admin_secret: Annotated[str | None, Header()] = None) -> None:
    """Shared secret on every /admin route, local compose or not.

    This router can break any provider on demand from anywhere that can reach the
    port. "It only runs in compose" is how an unauthenticated fault-injection API
    ends up somewhere it shouldn't be, so it is guarded from the first commit
    that introduces it.

    An unset ADMIN_SECRET disables the router rather than opening it. Failing
    closed on a missing secret is the whole point; a config gap must never be the
    thing that grants access.
    """
    expected = os.getenv("ADMIN_SECRET")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="admin API is disabled: ADMIN_SECRET is unset",
        )
    if not x_admin_secret or not secrets.compare_digest(x_admin_secret, expected):
        raise HTTPException(status_code=403, detail="bad or missing X-Admin-Secret")


admin = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@admin.get("/health")
async def admin_health(
    config: Annotated[GatewayConfig, Depends(get_config)],
    r: Annotated[Redis, Depends(get_redis)],
) -> dict[str, Any]:
    """Per-provider window, exactly as the breaker reads it."""
    return {name: asdict(await health.snapshot(r, name)) for name in config.providers}


@admin.get("/circuits")
async def admin_circuits(
    config: Annotated[GatewayConfig, Depends(get_config)],
    r: Annotated[Redis, Depends(get_redis)],
) -> dict[str, Any]:
    """Every circuit's state. The first thing to look at when routing surprises you."""
    return {
        name: asdict(await breaker.state_of(r, name)) for name in config.providers
    }


@admin.get("/chaos")
async def admin_chaos_list(
    config: Annotated[GatewayConfig, Depends(get_config)],
    r: Annotated[Redis, Depends(get_redis)],
) -> dict[str, Any]:
    return {
        name: asdict(spec)
        for name, spec in (await chaos.active(r, list(config.providers))).items()
    }


@admin.post("/chaos")
async def admin_chaos_set(
    request: ChaosRequest,
    config: Annotated[GatewayConfig, Depends(get_config)],
    r: Annotated[Redis, Depends(get_redis)],
) -> dict[str, Any]:
    if request.provider not in config.providers:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider {request.provider!r}; "
            f"known: {sorted(config.providers)}",
        )
    if request.error_type not in chaos.INJECTABLE:
        raise HTTPException(
            status_code=400,
            detail=f"error_type must be one of {sorted(chaos.INJECTABLE)}",
        )

    spec = chaos.ChaosSpec(
        provider=request.provider,
        error_rate=request.error_rate,
        latency_ms=request.latency_ms,
        error_type=request.error_type,
    )
    await chaos.set_(r, spec, request.ttl_s)
    return {"chaos": asdict(spec), "expires_in_s": request.ttl_s}


@admin.delete("/chaos/{provider}")
async def admin_chaos_clear(
    provider: str,
    r: Annotated[Redis, Depends(get_redis)],
) -> dict[str, Any]:
    return {"provider": provider, "cleared": await chaos.clear(r, provider)}


app.include_router(admin)


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models(config: Annotated[GatewayConfig, Depends(get_config)]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": name, "model": p.model}
            for name, p in config.providers.items()
            if p.available
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    meta: Annotated[RequestMetadata, Depends(require_metadata)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    r: Annotated[Redis, Depends(get_redis)],
    idempotency_key: Annotated[str | None, Header()] = None,
    provider: Annotated[
        str | None, Query(description="Pin a provider, bypassing the ladder")
    ] = None,
) -> Response:
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported")

    if idempotency_key:
        stored = await idempotency.claim(
            r, meta.tenant, idempotency_key, ttl_s=config.queue.idempotency_ttl_s
        )
        if stored is not None:
            return _replay(stored)

    try:
        status, body = await _serve(request, meta, config, r, provider)
    except BaseException:
        # Including HTTPException from _resolve_ladder. A claim left behind by a
        # request that never produced a response answers every retry with 409,
        # which is a worse outage than the one that caused it.
        if idempotency_key:
            await idempotency.release(r, meta.tenant, idempotency_key)
        raise

    if idempotency_key:
        if status < 400:
            await idempotency.store(
                r,
                meta.tenant,
                idempotency_key,
                status,
                body,
                ttl_s=config.queue.idempotency_ttl_s,
            )
        else:
            await idempotency.release(r, meta.tenant, idempotency_key)

    return JSONResponse(status_code=status, content=body)


def _replay(stored: str) -> JSONResponse:
    """Hand back what this key already produced, without touching a provider."""
    if stored == idempotency.IN_FLIGHT:
        raise HTTPException(
            status_code=409,
            detail="a request with this Idempotency-Key is already in flight",
        )
    envelope = json.loads(stored)
    return JSONResponse(
        status_code=envelope["status"],
        content=envelope["body"],
        # So a replay is visible in a trace rather than being inferred from a
        # provider counter that did not move.
        headers={"X-Idempotent-Replay": "true"},
    )


async def _serve(
    request: ChatCompletionRequest,
    meta: RequestMetadata,
    config: GatewayConfig,
    r: Redis,
    provider: str | None,
) -> tuple[int, dict[str, Any]]:
    """The request itself, as a status and a body.

    Split out from the handler so the idempotency guard has one place to store or
    release, rather than one at each of three returns - which is the shape that
    eventually grows a fourth return nobody remembers to wire up.
    """
    ladder = _resolve_ladder(config, meta.request_class, provider)
    payload = request.to_provider_kwargs()

    routed = await router.route(r, config, meta, payload, ladder)
    attempts = [asdict(a) for a in routed.attempts]

    if routed.result is None:
        return _exhausted_status(routed), _exhausted(meta, attempts, routed.last_failure)

    result = routed.result
    log.info(
        "request_id=%s tenant=%s feature=%s class=%s provider=%s outcome=%s "
        "latency_ms=%.1f cost_usd=%.6f attempts=%d",
        meta.request_id,
        meta.tenant,
        meta.feature,
        meta.request_class,
        result.provider,
        result.outcome,
        result.latency_ms,
        result.cost_usd,
        len(attempts),
    )

    if not result.outcome.ok:
        return _STATUS_BY_OUTCOME[result.outcome], {
            "error": {
                "message": result.error or str(result.outcome),
                "type": str(result.outcome),
                "provider": result.provider,
            },
            "x_gateway": {
                "request_id": meta.request_id,
                "request_class": meta.request_class,
                "attempts": attempts,
            },
        }

    return 200, _to_openai_response(result, meta, routed.attempts).model_dump(mode="json")


def _exhausted_status(routed: router.RouteOutcome) -> int:
    """What an exhausted ladder returns.

    503 is right only when nothing was attempted - every circuit open means there
    genuinely is no capacity. When rungs did run and failed, the caller gets what
    actually happened on the last one: a ladder of rate limits is a 429, and
    flattening that to 503 costs the client the signal telling it to back off
    instead of retrying straight away.
    """
    last = routed.last_failure
    return _STATUS_BY_OUTCOME[Outcome(last.outcome)] if last else 503


def _exhausted(
    meta: RequestMetadata,
    attempts: list[dict[str, Any]],
    last: ProviderCallLog | None,
) -> dict[str, Any]:
    """The body when the whole ladder is gone.

    Naming every provider and what it did - failed with what, or skipped because
    its circuit was open and for how much longer - costs nothing to assemble and
    is the difference between a debuggable outage and a shrug.
    """
    log.error(
        "ladder exhausted request_id=%s class=%s tried=%s",
        meta.request_id,
        meta.request_class,
        [a["provider"] for a in attempts],
    )
    return {
        "error": {
            "message": "every provider in the ladder failed or was unavailable",
            "type": str(last.outcome) if last else "no_provider_available",
            "ladder": [a["provider"] for a in attempts],
        },
        "x_gateway": {
            "request_id": meta.request_id,
            "request_class": meta.request_class,
            "attempts": attempts,
        },
    }


def _resolve_ladder(
    config: GatewayConfig, request_class: str, override: str | None
) -> list[str]:
    """The ordered providers to try, or a 4xx/503 explaining why there are none."""
    if override:
        if override not in config.providers:
            raise HTTPException(
                status_code=400,
                detail=f"unknown provider {override!r}; known: {sorted(config.providers)}",
            )
        if not config.providers[override].available:
            raise HTTPException(
                status_code=503,
                detail=f"provider {override!r} is disabled: "
                f"{config.providers[override].api_key_env} is unset",
            )
        return [override]

    ladder = config.ladder_for(request_class)
    if not ladder:
        raise HTTPException(
            status_code=503,
            detail=f"no available provider for class {request_class!r}; "
            "every provider in its ladder is missing an API key",
        )
    return ladder


def _to_openai_response(
    result: ProviderResult, meta: RequestMetadata, attempts: list[ProviderCallLog]
) -> ChatCompletionResponse:
    response = result.response
    choices = [
        Choice(
            index=getattr(c, "index", i),
            message=ChatMessage(
                role=getattr(c.message, "role", "assistant") or "assistant",
                content=getattr(c.message, "content", None),
            ),
            finish_reason=getattr(c, "finish_reason", None),
        )
        for i, c in enumerate(getattr(response, "choices", []))
    ]
    return ChatCompletionResponse(
        id=getattr(response, "id", None) or f"chatcmpl-{uuid.uuid4().hex}",
        created=getattr(response, "created", None) or int(time.time()),
        model=getattr(response, "model", None) or result.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
        x_gateway=GatewayMeta(
            provider=result.provider,
            model=result.model,
            request_class=meta.request_class,
            request_id=meta.request_id,
            tenant=meta.tenant,
            feature=meta.feature,
            latency_ms=round(result.latency_ms, 1),
            cost_usd=result.cost_usd,
            attempts=[asdict(a) for a in attempts],
        ),
    )

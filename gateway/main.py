"""FastAPI app: one service every model call routes through."""

from __future__ import annotations

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from gateway import breaker, chaos, health, idempotency, metrics, queue, router
from gateway.config import GatewayConfig, get_config, get_redis
from gateway.providers import Outcome, ProviderCallLog
from gateway.schemas import (
    ChaosRequest,
    ChatCompletionRequest,
    RequestMetadata,
    to_openai_response,
)

log = logging.getLogger(__name__)

# How a provider outcome surfaces to the caller. AUTH and MODEL_NOT_FOUND are
# 502s on purpose: a rejected key or a model that has been decommissioned out
# from under the config is the gateway's problem, not the caller's.
#
# Every non-OK Outcome needs an entry. _serve and _exhausted_status both index
# this directly, so a missing member is a KeyError and a 500 on the one request
# that would have explained the misconfiguration - hence the test below it.
_STATUS_BY_OUTCOME: dict[Outcome, int] = {
    Outcome.RATE_LIMIT: 429,
    Outcome.TIMEOUT: 504,
    Outcome.SERVER_ERROR: 502,
    Outcome.AUTH: 502,
    Outcome.MODEL_NOT_FOUND: 502,
    Outcome.CONTENT_FILTER: 400,
    Outcome.BAD_REQUEST: 400,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    config.log_startup_state()
    # Through dependency_overrides rather than calling get_redis() straight,
    # because startup is outside the request cycle and would otherwise ignore
    # the override every test sets - opening a real connection to whatever is
    # listening on localhost:6379, which is worse than slow: it silently passes
    # against leftover state from a compose stack somebody forgot to stop.
    resolve = app.dependency_overrides.get(get_redis, get_redis)
    await breaker.seed_gauge(resolve(), list(config.providers))
    yield


app = FastAPI(
    title="Self-Healing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def count_responses(request: Request, call_next):
    """Count what the caller was actually told.

    One middleware rather than an increment at each return, because
    chat_completions has four of those plus HTTPExceptions raised from
    _resolve_ladder, _replay and require_admin - the same reasoning that put the
    idempotency release in a single `except BaseException` instead of at every
    exit. A response the counter misses is a response the availability figure
    silently rounds in its own favour.

    Labelled with the templated route rather than the raw path, so /v1/jobs/{id}
    is one series and not one per job id.
    """
    response = await call_next(request)
    route = request.scope.get("route")
    metrics.RESPONSES.labels(
        route=getattr(route, "path", "unmatched"), status=str(response.status_code)
    ).inc()
    return response


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


@admin.get("/dlq")
async def admin_dlq(r: Annotated[Redis, Depends(get_redis)]) -> dict[str, Any]:
    """Work that survived an outage and still ran out of attempts.

    Deliberately not a list of everything that failed - a job rejected for bad
    input never got here, so anything in this queue is worth looking at.
    """
    jobs = await queue.dead(r)
    return {"count": len(jobs), "jobs": [job.public() for job in jobs]}


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
    # Classification at the boundary, exactly as phase 1 set it up: an
    # interactive class fails fast, a deferrable one is accepted and drained
    # later. An explicit ?provider= pin is an operator saying "call this now",
    # so it takes the synchronous path whatever the class says.
    if provider is None and config.classes[meta.request_class].deferrable:
        return await _accept(r, meta, request)

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

    return 200, to_openai_response(result, meta, routed.attempts).model_dump(mode="json")


async def _accept(
    r: Redis, meta: RequestMetadata, request: ChatCompletionRequest
) -> tuple[int, dict[str, Any]]:
    """Take the work now, do it later.

    This is what a total outage costs a deferrable caller: a job id instead of a
    completion. Nothing is attempted here - queueing only when the ladder has
    already failed would make the 202 arrive after every rung had timed out,
    which is the worst of both shapes.
    """
    job_id = await queue.enqueue(r, meta, request.to_provider_kwargs())
    log.info(
        "queued job=%s request_id=%s tenant=%s feature=%s class=%s",
        job_id,
        meta.request_id,
        meta.tenant,
        meta.feature,
        meta.request_class,
    )
    return 202, {
        "job_id": job_id,
        "status": str(queue.Status.QUEUED),
        "poll": f"/v1/jobs/{job_id}",
        "x_gateway": {
            "request_id": meta.request_id,
            "request_class": meta.request_class,
        },
    }


@app.get("/v1/jobs/{job_id}")
async def get_job(
    job_id: str,
    meta: Annotated[RequestMetadata, Depends(require_metadata)],
    r: Annotated[Redis, Depends(get_redis)],
) -> dict[str, Any]:
    """Collect a deferred result.

    Another tenant's job is a 404 rather than a 403: a job id is the only thing
    a caller needs to guess, and confirming that one exists is the half of the
    answer worth withholding.
    """
    job = await queue.get(r, job_id)
    if job is None or job.meta.tenant != meta.tenant:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    return job.public()


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



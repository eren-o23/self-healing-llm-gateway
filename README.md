# Self-Healing LLM Gateway

[![CI](https://github.com/eren-o23/self-healing-llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/eren-o23/self-healing-llm-gateway/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)

**A LLM gateway that keeps serving when a provider goes down.**

**Why it exists.** Anything built on a hosted model inherits that provider's outages, rate limits
and retired models. This is the layer that makes somebody else's bad day survivable: it notices the
failure, routes around it, and lets the provider back in once it recovers — instead of passing a
502 straight through to your users.

**Start here.** [What the demo produced](#what-the-demo-actually-produced) is ten lines of output
and shows the whole arc. If you read one design section, make it
[the error taxonomy](#the-error-taxonomy-is-the-spine) — it is the decision everything else falls
out of.

## The short version

- **Circuit breaking, per provider.** A failing backend is taken out of rotation; half-open probes
  let it back in on its own. State lives in Redis, so it survives a restart.
- **Automatic failover** down a ladder that differs by request class — a cheap classification call
  and a long generation call should not fail over the same way. One class hedges.
- **A retry queue for deferrable work.** When there is nowhere to send a request, the API answers
  `202` with a job id and a separate worker drains it later, with backoff, idempotency and a DLQ.
- **Cost attributed** to the tenant and feature that caused it, enforced at the boundary.
- **Measured, not asserted.** In one clean run, 60/60 requests returned 200 through a 30-second
  total failure of the provider that had been serving all of them. The figure comes out of
  Prometheus in three lines of arithmetic, [below](#what-the-demo-actually-produced).
- **169 tests**, no services needed, plus a demo script that breaks a provider on purpose so the
  claim can be watched rather than believed.

**Stack:** Python 3.12 · FastAPI · Redis · Prometheus · Grafana · LiteLLM (as an adapter only) ·
Docker Compose.

<!-- Screenshots of the board during a demo run go here: healthy, tripped, recovered. -->

## What the demo actually produced

One run of `./scripts/demo.sh` against a freshly started stack. Anthropic is failing 100% of calls
from T+20s to T+50s:

```
  8  200  anthropic   2212ms  [anthropic:ok]
T+20   breaking anthropic
 10  200  groq         250ms  [anthropic:server_error -> groq:ok]   failover
 19  200  groq         310ms  [anthropic:server_error -> groq:ok]
 20  200  groq         266ms  [anthropic:skipped -> groq:ok]        circuit open, no call made
 37  200  groq         400ms  [anthropic:skipped -> groq:ok]
T+50   clearing anthropic; watch the circuit half-open and close
 38  200  groq         630ms  [anthropic:skipped -> groq:ok]
 39  200  anthropic   2140ms  [anthropic:ok]                        probe succeeded, closed again
 60  200  anthropic   2062ms  [anthropic:ok]
```

Ten failovers, then the circuit opens and the remaining eighteen requests skip the dead provider
without dialling it at all — that transition from `server_error` to `skipped` at request 20 is the
breaker doing its job. Latency drops from ~2.2s to ~330ms on the way in, because the rung it fails
over to happens to be faster, and comes back up on the way out.

The circuit for the broken provider, sampled every 10 seconds across that window:

```
anthropic  ...XX~.....
groq       ...........
ollama     ...........

. closed   X open   ~ half-open
```

And the measured figure, from `scripts/availability.py`:

```
window          10m
responses       60
5xx to caller   0
availability    100.000%
```

**60 of 60 requests returned 200**, through a 30-second total failure of the provider serving all
of them at the start. Read it for what it is: one two-minute run at roughly one request per second,
not a production SLO. It is measured rather than asserted, which is the only property being claimed
— `availability.py` reads the counters out of Prometheus and divides them, and the arithmetic is
three lines long precisely so it can be checked.

## Run it

```bash
cp .env.example .env          # fill in whichever provider keys you have
docker compose up -d          # gateway, worker, redis, prometheus, grafana
./scripts/demo.sh             # watch http://localhost:3000
python3 scripts/availability.py --window 10m
```

Grafana opens straight onto the board — provisioned from `config/grafana/`, nothing to import.
A provider whose key is missing is dropped from every ladder at startup rather than being fatal,
so the stack comes up and serves with whatever you have. `ollama` needs no key and is the floor of
every ladder.

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: acme' -H 'X-Feature: support-bot' -H 'X-Request-Id: 1' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

Tenant, feature and request id are mandatory. Cost that cannot be attributed to whoever caused it
is cost nobody can act on, and retrofitting required fields onto a live gateway is the migration
nobody wants — so they are required from the first request the service ever serves.

## How it works

A request is classified at the boundary, and its class names an ordered ladder of providers:

```
                    ┌──────────────────────────────────────────────┐
  request ─────────►│  classify → ladder → breaker → provider       │
                    │                ▲         │                   │
                    │                └── failover on trippable ─────┤
                    └───────────────┬──────────────────────────────┘
                                    │ deferrable class, or nothing left
                                    ▼
                              Redis delay queue ──► worker ──► same ladder
```

- **`router.py`** walks the ladder. `call_provider()` never raises for a provider failure — it
  returns a result carrying an outcome — so the walk is a flat loop rather than a try/except per
  rung. One class hedges: if the first rung misses its budget, a second is raced alongside it.
- **`breaker.py`** holds one circuit per provider in Redis. Transitions are computed lazily on
  read, so there is no scheduler and no background task to supervise. Half-open probes are what
  make it self-healing rather than merely defensive.
- **`health.py`** keeps a sliding window per provider, also in Redis, and it is what the breaker
  reads to decide.
- **`queue.py` / `worker.py`** take deferrable work when there is nowhere to send it. The API
  answers 202 with a job id; a separate process drains a Redis ZSET delay queue through the *same*
  router, with full-jitter backoff and a dead-letter queue.
- **`idempotency.py`** shipped before any retry existed, which is the only order that works —
  building retries first is how a blip becomes a duplicated side effect.

### The error taxonomy is the spine

Every provider failure is normalised to one outcome, and one flag on it — `trippable` — answers
three separate questions the same way:

| | trippable | meaning |
|---|---|---|
| `rate_limit`, `timeout`, `server_error` | yes | the provider is sick |
| `auth`, `model_not_found` | yes | *our* configuration is stale; this provider is unusable for everyone |
| `content_filter`, `bad_request` | **no** | the caller sent something every provider will reject identically |

A non-trippable failure must not open a circuit (one user's bad input would take a healthy provider
offline for everybody), must not fail over (three more rungs bill for the same 400), and must not
be retried by the worker. Getting this wrong in either direction is the most commonly botched part
of a gateway, and both directions have tests.

`model_not_found` is on the trippable side because it was on the other side once and that was a
bug: Groq decommissioned a configured model mid-project, the ladder read it as the caller's fault
and stopped dead at that rung, the circuit never opened over a provider that could not serve a
single request, and queued work failed terminally.

## Two decisions worth defending

**Why not LiteLLM's Router.** LiteLLM is used here as a provider adapter and a price table, and
its Router — which ships fallbacks, retries and cooldowns — is deliberately unused. The routing,
the breaking and the healing are the thing being built; delegating them would leave a project that
configures a library. It also would not have produced the parts that turned out to matter: a
taxonomy where caller faults are structurally incapable of opening a circuit, per-provider latency
budgets, and a queue that knows the difference between "this failed" and "nothing was tried".

**Why Redis *and* Prometheus.** They hold overlapping data on purpose. Prometheus is for humans —
dashboards, percentiles over time, the demo. Redis is the state the *routing decision* reads, and
it survives a restart: after `docker compose restart gateway` the Prometheus counters start again
from zero while `/admin/health` comes back byte-identical. A gateway that queried Prometheus to
decide where to send a request would put a monitoring system on its control path, which fails in
the worst possible direction — the query gets slow or unavailable exactly when things are on fire.

Health recording is fail-open for the same reason: if Redis is unreachable the window reads empty,
everything looks healthy and nothing trips. That is the right way for a breaker to fail.

## Trade-offs and known ceilings

The deliberate shortcuts are marked `ponytail:` in the source, each naming its ceiling and the
upgrade path. The ones worth knowing before reading the code:

- **Streaming is refused with a 400 on purpose.** Half-supported streaming is worse than none.
- **A crash between `ZPOPMIN` and completion loses the job.** The upgrade is Redis Streams with
  consumer groups and `XACK`.
- **Hedging roughly doubles spend on the calls it fires.** It is measured rather than hand-waved:
  fired, won and wasted are three separate counters, and a hedge that came back first *with a
  failure* counts as wasted, not won.
- **One Prometheus registry per process.** Running uvicorn with `--workers > 1` needs
  `prometheus_client`'s multiprocess mode.
- **The breaker's state transition is a read-modify-write with no `WATCH`.** Concurrent requests
  can admit an extra probe. Harmless at this scale, wrong at a much larger one.
- **Percentiles are computed in-process over the raw window.** Fine to a few thousand samples.
- **`openai` has no key in this deployment**, so the live ladder is three deep —
  `anthropic → groq → ollama`, with a free local model as the floor. Enough to demonstrate
  failover; a fourth rung would only make the ladder longer, not the behaviour different.

## Tests

```bash
uv sync && uv run pytest      # 169 tests, no services needed - fakeredis
uv run ruff check
```

Every phase of this project has had at least one bug the tests did not catch, found by running the
real stack. Those are the interesting ones, and each is now a test: an attempt where every circuit
was open counting against the retry budget; a capacity re-check that scaled with the attempt count
and idled through the recovery it was waiting for; a cost lookup silently returning $0 for a
genuinely billed call.

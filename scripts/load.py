#!/usr/bin/env python3
"""Steady traffic through the gateway, one line per request.

Python rather than a shell loop on purpose. Driving this with curl needs
conditional headers and JSON extraction inline, and the quoting that requires
gets mangled silently - which looks exactly like a gateway bug and is not.

Standard library only, so it runs without the project's venv.

    scripts/load.py --count 30 --interval 0.5 --class interactive.chat
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

BODY = {
    "messages": [{"role": "user", "content": "Reply with one word: ok"}],
    # 64 rather than 8: groq now runs a reasoning model, which spends ~20 tokens
    # thinking before it emits anything. At 8 it returns finish_reason=length and
    # empty content - a successful call with nothing in it, which reads on camera
    # as a broken gateway.
    "max_tokens": 64,
}


def call(base: str, request_class: str, request_id: str, timeout: float):
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(BODY).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Tenant-Id": "acme",
            "X-Feature": "support-bot",
            "X-Request-Id": request_id,
            "X-Request-Class": request_class,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")
    except Exception as exc:  # noqa: BLE001 - a dead gateway is a result too
        return 0, {"error": {"type": type(exc).__name__}}


def describe(status: int, body: dict) -> str:
    """Served-by plus the full trail, which is what makes a failover legible."""
    if "job_id" in body:  # a deferrable class was accepted rather than attempted
        return f"{status}  {'queued':<10}  [{body['job_id'][:8]}]"

    gateway = body.get("x_gateway", {})
    trail = " -> ".join(
        f"{a['provider']}:{a['outcome']}" for a in gateway.get("attempts", [])
    )
    served = gateway.get("provider") or body.get("error", {}).get("type", "?")
    latency = gateway.get("latency_ms")
    suffix = f"  {latency:.0f}ms" if latency else ""
    return f"{status}  {served:<10}{suffix}  [{trail}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--class", dest="request_class", default="interactive.chat")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--base", default=os.getenv("GATEWAY_URL", "http://localhost:8000")
    )
    args = parser.parse_args()

    for i in range(1, args.count + 1):
        started = time.time()
        status, body = call(args.base, args.request_class, f"load-{i}", args.timeout)
        print(f"{i:>3}  {describe(status, body)}", flush=True)
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()

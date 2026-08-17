#!/usr/bin/env python3
"""Availability across a window, measured from Prometheus rather than asserted.

    1 - (5xx returned to the caller / responses returned to the caller)

This reads gateway_responses_total, not gateway_requests_total, and the
difference is the whole point. gateway_requests_total counts calls *out* to
providers: a request that failed over twice before succeeding is three of them
and one answer to the caller. Dividing that would put failures the gateway
completely absorbed into the denominator - the number would move whenever
failover worked, which is precisely backwards.

Only /v1/chat/completions counts. /metrics, /healthz and the admin router are
not the service anybody is measuring.

Standard library only, so it runs without the project's venv.

    scripts/availability.py --window 5m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

ROUTE = "/v1/chat/completions"


def delta(selector: str, window: str) -> str:
    """Counter growth over the window, as now-minus-then.

    Deliberately not increase(): a counter series only springs into existence
    the first time its labels are observed, so the 0 -> N jump happens at
    creation and there is no earlier sample for a range function to measure
    against. The result is 0.

    That is not a corner case here, it is the main case. status="502" does not
    exist until the first outage, and the first outage is exactly what the demo
    produces - so increase() reports a clean 100% across a window containing
    nothing but failures. Wrong, and wrong in the flattering direction, which is
    the one thing this script exists to prevent.

    `or vector(0)` supplies the missing "then", which is the honest reading:
    a series that did not exist had not counted anything yet.

    Known ceiling: this cannot see through a counter reset, so a gateway restart
    mid-window under-reports the failures. Prometheus keeps its own view of a
    restart while the in-process counters go back to zero - so if one happened,
    measure a window that starts after it.
    """
    return f"sum({selector}) - (sum({selector} offset {window}) or vector(0))"


def query(base: str, expr: str, timeout: float) -> float:
    """One instant query, summed. Absent series mean zero, not an error."""
    url = f"{base}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = json.load(response)

    if body.get("status") != "success":
        raise SystemExit(f"prometheus rejected the query: {body.get('error')}\n  {expr}")

    return sum(float(series["value"][1]) for series in body["data"]["result"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", default="5m", help="lookback, in Prometheus duration syntax"
    )
    parser.add_argument(
        "--base", default=os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    served = f'gateway_responses_total{{route="{ROUTE}"}}'
    errored = f'gateway_responses_total{{route="{ROUTE}",status=~"5.."}}'

    total = query(args.base, delta(served, args.window), args.timeout)
    failed = query(args.base, delta(errored, args.window), args.timeout)

    if total == 0:
        print(f"no requests to {ROUTE} in the last {args.window}.")
        print("Run scripts/demo.sh first, or widen --window.")
        sys.exit(1)

    availability = 1 - failed / total

    print(f"window          {args.window}")
    print(f"responses       {total:.0f}")
    print(f"5xx to caller   {failed:.0f}")
    print(f"availability    {availability * 100:.3f}%")


if __name__ == "__main__":
    main()

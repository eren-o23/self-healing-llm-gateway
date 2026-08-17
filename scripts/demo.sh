#!/usr/bin/env bash
# The demo arc, driven the same way every time: steady load, anthropic breaks at
# T+20s, the fault clears at T+50s, traffic runs on through the recovery.
#
# Watch it at http://localhost:3000 while this runs.
#
# The shell here is deliberately thin. Every loop, every conditional header and
# every piece of JSON handling lives in scripts/load.py, because inlining that
# sort of thing has silently mangled a test harness in three separate phases of
# this project - each time producing a wall of 400s that looked exactly like a
# gateway bug and was not. Two fixed curl calls is all the shell is trusted with.

set -euo pipefail

BASE="${GATEWAY_URL:-http://localhost:8000}"
PROVIDER="${DEMO_PROVIDER:-anthropic}"
# load.py paces itself at *at most* one request per RATE seconds - if a provider
# takes longer than that, provider latency sets the tempo instead. Anthropic runs
# ~2s here, so 60 requests is roughly two minutes, which leaves a good stretch of
# recovery after the fault clears at T+50s.
RATE="${DEMO_INTERVAL:-1}"
TOTAL="${DEMO_REQUESTS:-60}"

if [[ -z "${ADMIN_SECRET:-}" && -f .env ]]; then
  ADMIN_SECRET="$(grep -E '^ADMIN_SECRET=' .env | cut -d= -f2-)"
fi

# Failing closed on an unset secret is deliberate in the gateway - /admin answers 503
# rather than running unauthenticated. Say so here rather than letting the demo
# run on with an un-injected fault and a flat, uninteresting graph.
if [[ -z "${ADMIN_SECRET:-}" ]]; then
  echo "ADMIN_SECRET is unset; /admin would answer 503 and no fault would be injected." >&2
  echo "Set it in .env (see .env.example) or export it before running this." >&2
  exit 1
fi

chaos_on() {
  curl -sS -X POST "$BASE/admin/chaos" \
    -H "X-Admin-Secret: $ADMIN_SECRET" \
    -H 'Content-Type: application/json' \
    -d "{\"provider\":\"$PROVIDER\",\"error_rate\":1.0,\"error_type\":\"server_error\",\"ttl_s\":60}" \
    >/dev/null
}

chaos_off() {
  curl -sS -X DELETE "$BASE/admin/chaos/$PROVIDER" -H "X-Admin-Secret: $ADMIN_SECRET" >/dev/null
}

# However this exits - finished, failed or Ctrl-C - the provider gets un-broken.
# The TTL on the injected fault would expire on its own anyway; this just means
# nobody is left staring at a mysteriously sick provider in the meantime.
trap 'chaos_off || true' EXIT

echo "T+0    steady load: ${TOTAL} requests at ${RATE}s intervals"
python3 scripts/load.py --count "$TOTAL" --interval "$RATE" &
LOAD=$!

sleep 20
echo "T+20   breaking $PROVIDER"
chaos_on

sleep 30
echo "T+50   clearing $PROVIDER; watch the circuit half-open and close"
chaos_off

wait "$LOAD"
echo
echo "done. The measured availability across that window:"
echo "  python3 scripts/availability.py --window 5m"

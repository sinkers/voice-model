#!/usr/bin/env bash
# tests/smoke_test.sh — basic sanity checks for the voice-model pipeline
# Prints PASS/FAIL for each check. Exit 0 if all pass, 1 if any fail.

set -uo pipefail

PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

pass() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

# ── Required binaries ─────────────────────────────────────────────────────────
for bin in python3 ffmpeg ssh; do
  if command -v "$bin" &>/dev/null; then
    pass "binary '$bin' found"
  else
    fail "binary '$bin' not found — install it"
  fi
done

# ── RunPod API key ────────────────────────────────────────────────────────────
if [ -n "${RUNPOD_API_KEY:-}" ]; then
  pass "RUNPOD_API_KEY env var is set"
elif [ -f "$HOME/.runpod_api_key" ]; then
  pass "~/.runpod_api_key exists"
else
  fail "no API key: set RUNPOD_API_KEY env var or create ~/.runpod_api_key"
fi

# ── monitor_pod.sh executable + usage ────────────────────────────────────────
MONITOR="$PROJECT_DIR/scripts/monitor_pod.sh"
if [ -x "$MONITOR" ]; then
  pass "monitor_pod.sh is executable"
else
  fail "monitor_pod.sh is not executable (run: chmod +x scripts/monitor_pod.sh)"
fi

# Running with no args should print usage (exit 2) not crash unexpectedly
if [ -x "$MONITOR" ]; then
  output=$("$MONITOR" 2>&1 || true)
  if echo "$output" | grep -qi "usage"; then
    pass "monitor_pod.sh prints usage when called with no args"
  else
    fail "monitor_pod.sh did not print usage when called with no args (got: $output)"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

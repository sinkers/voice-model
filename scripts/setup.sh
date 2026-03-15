#!/usr/bin/env bash
# setup.sh — Apply all required patches for voice model project + OpenClaw audio
# Safe to re-run (idempotent).
#
# What this does:
#   1. Pins api.telegram.org to IPv4 in /etc/hosts (fixes Node.js fetch failures)
#   2. Patches OpenClaw config with IPv4-only Telegram network settings
#   3. Verifies Node.js fetch can reach Telegram
#   4. Creates required project directories

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
HOSTS_FILE="/etc/hosts"
TELEGRAM_IPV4="149.154.166.110"
TELEGRAM_HOST="api.telegram.org"

ok()   { echo "  ✅  $*"; }
warn() { echo "  ⚠️   $*"; }
info() { echo "  ℹ️   $*"; }
fail() { echo "  ❌  $*"; exit 1; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Voice Model Project — Setup & Patch Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Project directories ────────────────────────────────────────────────────
echo "[ 1/4 ] Creating project directories..."

mkdir -p "$PROJECT_DIR/data/raw" \
         "$PROJECT_DIR/data/wavs" \
         "$PROJECT_DIR/data/clips" \
         "$PROJECT_DIR/scripts" \
         "$PROJECT_DIR/output" \
         "$PROJECT_DIR/notes"

ok "Directories ready"

# ── 2. /etc/hosts — pin Telegram to IPv4 ─────────────────────────────────────
echo ""
echo "[ 2/4 ] Patching /etc/hosts for Telegram IPv4..."

if grep -q "$TELEGRAM_HOST" "$HOSTS_FILE" 2>/dev/null; then
    EXISTING=$(grep "$TELEGRAM_HOST" "$HOSTS_FILE")
    if echo "$EXISTING" | grep -q "$TELEGRAM_IPV4"; then
        ok "/etc/hosts already has correct IPv4 entry: $TELEGRAM_IPV4 $TELEGRAM_HOST"
    else
        warn "Found existing /etc/hosts entry for $TELEGRAM_HOST but IP differs:"
        warn "  $EXISTING"
        warn "Remove it manually and re-run if you want this script to apply the fix."
    fi
else
    echo "$TELEGRAM_IPV4 $TELEGRAM_HOST" | sudo tee -a "$HOSTS_FILE" > /dev/null
    ok "Added: $TELEGRAM_IPV4 $TELEGRAM_HOST → $HOSTS_FILE"
fi

# Verify
RESOLVED=$(getent ahosts "$TELEGRAM_HOST" 2>/dev/null | awk '{print $1}' | head -1)
if [ "$RESOLVED" = "$TELEGRAM_IPV4" ]; then
    ok "$TELEGRAM_HOST resolves to $RESOLVED (IPv4 ✓)"
else
    warn "$TELEGRAM_HOST resolved to: $RESOLVED (expected $TELEGRAM_IPV4)"
    warn "DNS fix may not have taken effect — check /etc/nsswitch.conf"
fi

# ── 3. OpenClaw config — IPv4-only Telegram network settings ─────────────────
echo ""
echo "[ 3/4 ] Patching OpenClaw config for IPv4-only Telegram..."

if [ ! -f "$OPENCLAW_CONFIG" ]; then
    warn "OpenClaw config not found at $OPENCLAW_CONFIG — skipping"
else
    # Check if network settings already present
    if python3 -c "
import json, sys
with open('$OPENCLAW_CONFIG') as f:
    cfg = json.load(f)
tg = cfg.get('channels', {}).get('telegram', {})
net = tg.get('network', {})
if net.get('autoSelectFamily') == False and net.get('dnsResultOrder') == 'ipv4first':
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        ok "OpenClaw config already has IPv4-only network settings"
    else
        python3 - <<'PYEOF'
import json, sys

config_path = "$OPENCLAW_CONFIG"

with open(config_path) as f:
    cfg = json.load(f)

# Ensure path exists
cfg.setdefault("channels", {}).setdefault("telegram", {})

cfg["channels"]["telegram"]["network"] = {
    "autoSelectFamily": False,
    "dnsResultOrder": "ipv4first"
}

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("  ✅  OpenClaw config updated with IPv4-only network settings")
PYEOF
    fi
fi

# ── 4. Verify Node.js fetch ───────────────────────────────────────────────────
echo ""
echo "[ 4/4 ] Verifying Node.js fetch → api.telegram.org..."

FETCH_RESULT=$(node --input-type=module <<'JSEOF' 2>&1
try {
  const r = await fetch("https://api.telegram.org", { signal: AbortSignal.timeout(8000) });
  process.stdout.write("OK " + r.status + "\n");
} catch(e) {
  process.stdout.write("FAIL " + (e.cause?.code ?? e.message) + "\n");
}
JSEOF
)

if echo "$FETCH_RESULT" | grep -q "^OK"; then
    ok "Node.js fetch → api.telegram.org: $FETCH_RESULT"
else
    fail "Node.js fetch still failing: $FETCH_RESULT\n  Check /etc/hosts and /etc/nsswitch.conf"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All done! ✅"
echo ""
echo "  Next steps:"
echo "    - Restart OpenClaw gateway if it's running:"
echo "      systemctl --user restart openclaw-gateway.service"
echo "    - Start recording:"
echo "      python3 scripts/record.py"
echo "    - Or send audio files via Telegram"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

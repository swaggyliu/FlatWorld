#!/usr/bin/env bash
# Ephemeral Cloudflare quick tunnel → local :8765 (bypasses Aliyun SG).
set -euo pipefail
BIN=/tmp/cloudflared
LOG=/tmp/pusht_tunnel.log
PIDF=/tmp/pusht_tunnel.pid

if [ ! -x "$BIN" ]; then
  echo "missing $BIN" >&2
  exit 1
fi

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "tunnel already running pid=$(cat "$PIDF")"
  grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOG" | tail -1 || true
  exit 0
fi

pkill -f 'cloudflared tunnel --url' 2>/dev/null || true
: > "$LOG"
nohup "$BIN" tunnel --url http://127.0.0.1:8765 --no-autoupdate >"$LOG" 2>&1 &
echo $! >"$PIDF"
echo "started pid=$(cat "$PIDF")"

for i in $(seq 1 40); do
  url=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOG" | tail -1 || true)
  if [ -n "$url" ]; then
    echo "TUNNEL_URL=$url"
    exit 0
  fi
  sleep 1
done
echo "tunnel url not found yet; last log:"
tail -40 "$LOG"
exit 1

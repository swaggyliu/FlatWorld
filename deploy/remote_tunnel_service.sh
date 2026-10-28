#!/usr/bin/env bash
# Install cloudflared quick-tunnel as a user systemd unit (survives logout).
set -euo pipefail
BIN="${HOME}/.local/bin/cloudflared"
mkdir -p "$(dirname "$BIN")"
if [ ! -x "$BIN" ]; then
  if [ -x /tmp/cloudflared ]; then
    cp /tmp/cloudflared "$BIN"
  else
    echo "missing cloudflared binary" >&2
    exit 1
  fi
fi

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/flatworld-pusht-tunnel.service" <<EOF
[Unit]
Description=Cloudflare quick tunnel for FlatWorld Push-T
After=network-online.target flatworld-pusht.service
Wants=flatworld-pusht.service

[Service]
Type=simple
ExecStart=${BIN} tunnel --url http://127.0.0.1:8765 --no-autoupdate
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

loginctl enable-linger "$(whoami)" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now flatworld-pusht-tunnel.service
# Stop ad-hoc nohup tunnel if present
if [ -f /tmp/pusht_tunnel.pid ]; then
  kill "$(cat /tmp/pusht_tunnel.pid)" 2>/dev/null || true
  rm -f /tmp/pusht_tunnel.pid
fi
pkill -f '/tmp/cloudflared tunnel' 2>/dev/null || true
sleep 4
journalctl --user -u flatworld-pusht-tunnel.service -n 30 --no-pager || true
echo "---"
journalctl --user -u flatworld-pusht-tunnel.service --no-pager 2>/dev/null \
  | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | tail -1 \
  || true

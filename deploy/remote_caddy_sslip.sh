#!/usr/bin/env bash
# Terminate TLS on Aliyun for GitHub Pages WSS (no Cloudflare relay).
# Hostname: 8.138.152.185.sslip.io  →  127.0.0.1:8765
# Requires Aliyun SG inbound TCP 80 + 443 (ACME + HTTPS).
set -euo pipefail

HOST="8.138.152.185.sslip.io"
BIN="$HOME/.local/bin/caddy"
mkdir -p "$(dirname "$BIN")" "$HOME/.config/caddy"

if [ ! -x "$BIN" ]; then
  echo "downloading caddy…"
  tmp=$(mktemp -d)
  url="https://github.com/caddyserver/caddy/releases/download/v2.10.2/caddy_2.10.2_linux_amd64.tar.gz"
  mirror="https://ghfast.top/${url}"
  if ! curl -fsL --connect-timeout 20 --max-time 180 -o "$tmp/caddy.tgz" "$mirror"; then
    curl -fsL --connect-timeout 30 --max-time 180 -o "$tmp/caddy.tgz" "$url"
  fi
  tar -xzf "$tmp/caddy.tgz" -C "$tmp" caddy
  mv "$tmp/caddy" "$BIN"
  chmod +x "$BIN"
  rm -rf "$tmp"
fi
"$BIN" version

cat > "$HOME/.config/caddy/Caddyfile" <<EOF
${HOST} {
        reverse_proxy 127.0.0.1:8765
}
EOF

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/flatworld-pusht-caddy.service" <<EOF
[Unit]
Description=Caddy TLS front for FlatWorld Push-T (sslip.io)
After=network-online.target flatworld-pusht.service
Wants=flatworld-pusht.service

[Service]
Type=simple
Environment=XDG_DATA_HOME=%h/.local/share
Environment=XDG_CONFIG_HOME=%h/.config
ExecStart=${BIN} run --config %h/.config/caddy/Caddyfile --adapter caddyfile
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

loginctl enable-linger "$(whoami)" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now flatworld-pusht-caddy.service
sleep 6
systemctl --user --no-pager status flatworld-pusht-caddy.service || true
journalctl --user -u flatworld-pusht-caddy.service -n 40 --no-pager || true
curl -sS --connect-timeout 15 "https://${HOST}/health" || true
echo
echo "CADDY_HOST=https://${HOST}"

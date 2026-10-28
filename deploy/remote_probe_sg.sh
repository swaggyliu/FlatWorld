#!/usr/bin/env bash
set -euo pipefail
echo "region=$(curl -s http://100.100.100.200/latest/meta-data/region-id)"
MAC=$(curl -s http://100.100.100.200/latest/meta-data/mac)
echo "mac=$MAC"
echo "sgs=$(curl -s http://100.100.100.200/latest/meta-data/network/interfaces/macs/${MAC}/security-group-ids)"
echo "vpc=$(curl -s http://100.100.100.200/latest/meta-data/network/interfaces/macs/${MAC}/vpc-id)"

# cloudflared binary (CN-friendly mirrors)
URLS=(
  "https://github.com/cloudflare/cloudflared/releases/download/2026.10.5/cloudflared-linux-amd64"
  "https://ghfast.top/https://github.com/cloudflare/cloudflared/releases/download/2026.10.5/cloudflared-linux-amd64"
  "https://mirror.ghproxy.com/https://github.com/cloudflare/cloudflared/releases/download/2026.10.5/cloudflared-linux-amd64"
)
for u in "${URLS[@]}"; do
  echo "try $u"
  if curl -fsL --connect-timeout 20 --max-time 120 -o /tmp/cloudflared "$u"; then
    sz=$(wc -c </tmp/cloudflared)
    echo "got size=$sz"
    if [ "$sz" -gt 1000000 ]; then
      chmod +x /tmp/cloudflared
      /tmp/cloudflared --version || true
      exit 0
    fi
  fi
done
echo CLOUDFLARED_FAIL
exit 1

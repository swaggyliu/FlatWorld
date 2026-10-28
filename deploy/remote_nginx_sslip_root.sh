#!/usr/bin/env bash
# ONE-TIME root/sudo helper: real TLS on :443 for Pages WSS (no Cloudflare).
# Run on aliyun-node2 as a user with passwordless or interactive sudo, e.g.:
#   sudo bash /tmp/remote_nginx_sslip_root.sh
#   # or from your laptop after scp:
#   ssh -t numericsv@8.138.152.185 'sudo bash -s' < deploy/remote_nginx_sslip_root.sh
#
# Prerequisites (Aliyun 安全组 inbound):
#   TCP 80  — ACME HTTP-01
#   TCP 443 — HTTPS / WSS
# (Port 8765 can remain open for direct health checks.)
set -euo pipefail

HOST="${PUSHT_TLS_HOST:-8.138.152.185.sslip.io}"
UPSTREAM="${PUSHT_UPSTREAM:-127.0.0.1:8765}"
APP_USER="${PUSHT_APP_USER:-numericsv}"
APP_HOME="$(getent passwd "$APP_USER" | cut -d: -f6)"

if [ "$(id -u)" -ne 0 ]; then
  echo "must run as root (sudo). Example:"
  echo "  sudo bash $0"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y nginx certbot python3-certbot-nginx

# Stop anything that might conflict on 80/443 (user caddy, etc.)
systemctl stop caddy 2>/dev/null || true
pkill -u "$APP_USER" -f 'caddy run' 2>/dev/null || true || true

cat > /etc/nginx/sites-available/flatworld-pusht <<EOF
server {
    listen 80;
    server_name ${HOST};
    location / {
        proxy_pass http://${UPSTREAM};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
EOF
ln -sfn /etc/nginx/sites-available/flatworld-pusht /etc/nginx/sites-enabled/flatworld-pusht
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx

certbot --nginx -d "$HOST" --non-interactive --agree-tos \
  --register-unsafely-without-email --redirect

# Also stage certs for optional user-space nginx (8443) without re-issue.
USER_CERT_DIR="${APP_HOME}/.local/nginx/certs"
mkdir -p "$USER_CERT_DIR"
cp -f "/etc/letsencrypt/live/${HOST}/fullchain.pem" "$USER_CERT_DIR/fullchain.pem"
cp -f "/etc/letsencrypt/live/${HOST}/privkey.pem" "$USER_CERT_DIR/privkey.pem"
chown -R "${APP_USER}:${APP_USER}" "${APP_HOME}/.local/nginx"
chmod 640 "$USER_CERT_DIR/privkey.pem"
chmod 644 "$USER_CERT_DIR/fullchain.pem"

# Prefer system nginx on 443; stop user 8443 front if running.
sudo -u "$APP_USER" XDG_RUNTIME_DIR="/run/user/$(id -u "$APP_USER")" \
  systemctl --user disable --now flatworld-pusht-nginx.service 2>/dev/null || true

curl -fsS --connect-timeout 15 "https://${HOST}/health"
echo
echo "OK public WSS: wss://${HOST}/ws"
echo "Pages config should prefer: wss://${HOST}/ws"

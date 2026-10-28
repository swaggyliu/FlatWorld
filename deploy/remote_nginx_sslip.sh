#!/usr/bin/env bash
# User-space nginx TLS front for Push-T (no root).
# Default listen: 8443 → 127.0.0.1:8765  (hostname 8.138.152.185.sslip.io)
#
# Real Let's Encrypt certs require binding :80/:443 (root/setcap) — see
# deploy/remote_nginx_sslip_root.sh. Without that, this script uses an
# existing cert under ~/.local/nginx/certs or generates a self-signed one
# (browsers will reject WSS from GitHub Pages until a public CA cert is used).
#
# Aliyun SG: open TCP 8443 (or 443 if using the root helper). Port 8765 can stay.
set -euo pipefail

HOST="${PUSHT_TLS_HOST:-8.138.152.185.sslip.io}"
LISTEN_PORT="${PUSHT_NGINX_PORT:-8443}"
UPSTREAM="${PUSHT_UPSTREAM:-127.0.0.1:8765}"
PREFIX="${HOME}/.local/nginx"
BIN="${PREFIX}/root/usr/sbin/nginx"
ROOTFS="${PREFIX}/root"

mkdir -p "$PREFIX"/{conf,logs,certs,tmp,run,html} \
  "$PREFIX/tmp"/{client_body,proxy,fastcgi,uwsgi,scgi} \
  "$HOME/tmp/nginx-deb"

if [ ! -x "$BIN" ]; then
  echo "extracting nginx from apt packages (no root install)…"
  cd "$HOME/tmp/nginx-deb"
  apt-get download nginx-core nginx-common \
    libnginx-mod-http-xslt-filter libnginx-mod-http-geoip \
    libnginx-mod-mail libnginx-mod-stream
  mkdir -p "$ROOTFS"
  for deb in ./*.deb; do
    dpkg-deb -x "$deb" "$ROOTFS"
  done
fi
"$BIN" -v

if [ ! -f "$PREFIX/conf/mime.types" ]; then
  cp "$ROOTFS/etc/nginx/mime.types" "$PREFIX/conf/mime.types"
fi

CERT="$PREFIX/certs/fullchain.pem"
KEY="$PREFIX/certs/privkey.pem"
if [ -f "$CERT" ] && [ -f "$KEY" ]; then
  echo "using existing cert: $CERT"
else
  echo "no LE cert found; generating self-signed for ${HOST} (Pages WSS needs a public CA — run remote_nginx_sslip_root.sh)"
  openssl req -x509 -nodes -newkey rsa:2048 -days 30 \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=${HOST}" \
    -addext "subjectAltName=DNS:${HOST}"
fi

cat > "$PREFIX/conf/nginx.conf" <<EOF
worker_processes 1;
error_log ${PREFIX}/logs/error.log warn;
pid ${PREFIX}/run/nginx.pid;
daemon off;

events {
    worker_connections 1024;
}

http {
    include       ${PREFIX}/conf/mime.types;
    default_type  application/octet-stream;
    access_log    ${PREFIX}/logs/access.log;
    sendfile      on;
    keepalive_timeout 65;
    client_max_body_size 8m;
    # Ubuntu nginx hardcodes /var/lib/nginx — override for user prefix.
    client_body_temp_path ${PREFIX}/tmp/client_body;
    proxy_temp_path       ${PREFIX}/tmp/proxy;
    fastcgi_temp_path     ${PREFIX}/tmp/fastcgi;
    uwsgi_temp_path       ${PREFIX}/tmp/uwsgi;
    scgi_temp_path        ${PREFIX}/tmp/scgi;

    map \$http_upgrade \$connection_upgrade {
        default upgrade;
        ''      close;
    }

    server {
        listen ${LISTEN_PORT} ssl;
        server_name ${HOST};

        ssl_certificate     ${CERT};
        ssl_certificate_key ${KEY};
        ssl_protocols       TLSv1.2 TLSv1.3;

        location / {
            proxy_pass http://${UPSTREAM};
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection \$connection_upgrade;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }
    }
}
EOF

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/flatworld-pusht-nginx.service" <<EOF
[Unit]
Description=User nginx TLS front for FlatWorld Push-T (sslip.io)
After=network-online.target flatworld-pusht.service
Wants=flatworld-pusht.service

[Service]
Type=simple
ExecStart=${BIN} -p ${PREFIX} -c ${PREFIX}/conf/nginx.conf
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# Stop failed caddy unit if present (cannot bind :443 without capabilities).
systemctl --user disable --now flatworld-pusht-caddy.service 2>/dev/null || true

loginctl enable-linger "$(whoami)" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now flatworld-pusht-nginx.service
sleep 2
systemctl --user --no-pager status flatworld-pusht-nginx.service || true
journalctl --user -u flatworld-pusht-nginx.service -n 30 --no-pager || true

echo "--- local health (may be self-signed) ---"
curl -skS --connect-timeout 5 "https://127.0.0.1:${LISTEN_PORT}/health" || true
echo
echo "NGINX_WSS=wss://${HOST}:${LISTEN_PORT}/ws"
echo "NGINX_HTTPS=https://${HOST}:${LISTEN_PORT}/health"
echo "Note: public reachability needs Aliyun SG TCP ${LISTEN_PORT}."
echo "Trusted browser WSS needs a public CA cert (see deploy/remote_nginx_sslip_root.sh)."

# Deploy Push-T web backend on aliyun-node2

Host: `numericsv@8.138.152.185` (CPU, `cn-guangzhou`). Key: `D:/power-center/keys/id_ed25519`.

## 1. 安全组

| 端口 | 用途 |
|------|------|
| **8765** | 游戏 HTTP / WS（已开即可 `http://8.138.152.185:8765/`） |
| **80 + 443** | Pages 真 TLS（系统 nginx + certbot）。80/443 入站需放行；进程 bind 需要 **root/`setcap`** |
| **8443** | 无 root 时的 user nginx TLS（`remote_nginx_sslip.sh`）。需单独放行；自签证书时浏览器仍会拒 WSS |

GitHub Pages 页面是 **https**。浏览器会拦截到 `ws://…` 的 WebSocket（混合内容），
跟机器能不能出网无关——所以 Pages 上的画布必须连 **wss://**，且证书须为公信 CA。

## 2. TLS 前端（nginx，推荐）

### 现状（无密码 sudo）

- 用户空间 nginx 已可跑：`flatworld-pusht-nginx` → `127.0.0.1:8765`，监听 **8443**（自签证书）。
- **不能** bind `:80`/`:443`（`ip_unprivileged_port_start=1024`，无 `setcap`）。
- Let's Encrypt HTTP-01 / TLS-ALPN 需要公网 **80 或 443**，因此 **当前拿不到公信证书**；Pages 仍依赖 Cloudflare 隧道回退。
- 探测：80/443 从外网是「连接被拒」（多半 SG 已放行、本机无监听）；8443 超时（SG 未放行）。

本地已验证（自签）：

```bash
curl -sk https://127.0.0.1:8443/health   # {"ok":true,...}
systemctl --user status flatworld-pusht-nginx
```

### 无 root：仅高端口（准备 / 自测）

```bash
scp -i D:/power-center/keys/id_ed25519 deploy/remote_nginx_sslip.sh numericsv@8.138.152.185:/tmp/
ssh -i D:/power-center/keys/id_ed25519 numericsv@8.138.152.185 'bash /tmp/remote_nginx_sslip.sh'
```

→ `wss://8.138.152.185.sslip.io:8443/ws`（需 SG 放行 8443；自签则 Pages 不可用）。

### 有 sudo：真 `:443` + Let's Encrypt（Pages 直连）

1. 阿里云安全组放行 **TCP 80、443**（若尚未放行）。
2. 在机器上执行：

```bash
scp -i D:/power-center/keys/id_ed25519 deploy/remote_nginx_sslip_root.sh numericsv@8.138.152.185:/tmp/
ssh -t -i D:/power-center/keys/id_ed25519 numericsv@8.138.152.185 'sudo bash /tmp/remote_nginx_sslip_root.sh'
```

脚本会：`apt install nginx certbot`、反代到 `127.0.0.1:8765`、签发 `8.138.152.185.sslip.io`、启用 HTTPS 跳转。

成功后公网：

- Health: `https://8.138.152.185.sslip.io/health`
- WSS: `wss://8.138.152.185.sslip.io/ws`

### 备选：一次性 setcap（仍要 sudo）

若希望继续用用户 systemd 的 nginx/caddy 听 443：

```bash
sudo setcap 'cap_net_bind_service=+ep' /home/numericsv/.local/nginx/root/usr/sbin/nginx
# 或 caddy: sudo setcap 'cap_net_bind_service=+ep' /home/numericsv/.local/bin/caddy
```

然后把 `PUSHT_NGINX_PORT=443` 跑 `remote_nginx_sslip.sh`，并用 root 助手签发的证书放到 `~/.local/nginx/certs/`。

### Cloudflare quick tunnel（当前 Pages 可用回退）

无 root、有公信 TLS：`bash /tmp/remote_tunnel_service.sh`  
服务：`flatworld-pusht-tunnel`。

Pages 的 `docs/pusht/config.js` 会先试 `wss://8.138.152.185.sslip.io/ws`，失败再回退隧道。

## 3. 后端服务

复用 CI venv + 阿里云 PyPI：

```bash
export VIRTUAL_ENV=/opt/flexphysics/compute-venv
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
uv pip install tensordict rsl-rl-lib hydra-core
systemctl --user status flatworld-pusht
```

## 4. GitHub Pages

- Settings → Pages → Source: **GitHub Actions**
- 推送 `docs/**` 到 `main` 后访问：`https://swaggyliu.github.io/FlatWorld/pusht/`

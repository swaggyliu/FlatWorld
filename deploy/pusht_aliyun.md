# Deploy Push-T web backend on aliyun-node2

Host: `numericsv@8.138.152.185` (CPU, `cn-guangzhou`). Key: `D:/power-center/keys/id_ed25519`.

## 1. 安全组

| 端口 | 用途 |
|------|------|
| **8765** | 游戏 HTTP / WS（已开即可 `http://8.138.152.185:8765/`） |
| **80 + 443** | 可选：给 Pages 用的真 TLS（Caddy + `sslip.io`）。需 **root/`setcap`** 才能监听 443 |

GitHub Pages 页面是 **https**。浏览器会拦截到 `ws://…` 的 WebSocket（混合内容），
跟机器能不能出网无关——所以 Pages 上的画布必须连 **wss://**。

可选两条路：

1. **Caddy + sslip.io**（直连阿里云 IP，无 Cloudflare）：放行 80/443，且能 bind 443  
   `bash /tmp/remote_caddy_sslip.sh` → `wss://8.138.152.185.sslip.io/ws`
2. **Cloudflare quick tunnel**（当前默认，无 root）：TLS 终止在隧道上，后端仍是本机 8765  
   `bash /tmp/remote_tunnel_service.sh`

Pages 的 `docs/pusht/config.js` 会先试 sslip.io，失败再回退隧道。

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

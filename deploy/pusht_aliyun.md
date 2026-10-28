# Deploy Push-T web backend on aliyun-node2

Host: `numericsv@8.138.152.185` (CPU, `cn-guangzhou`). Key: `D:/power-center/keys/id_ed25519`.

## 1. 阿里云安全组放行 TCP 8765

1. 打开 [ECS 控制台](https://ecs.console.aliyun.com/) → 地域选 **华南1（广州）**
2. 实例 `i-7xv2m1atdhnhh0i890xw`（公网 IP `8.138.152.185`）→ **安全组** → 点安全组 ID 进入
3. **入方向** → **手动添加**：
   - 协议：自定义 TCP
   - 端口：`8765/8765`
   - 授权对象：`0.0.0.0/0`（或你自己的 IP）
   - 策略：允许
4. 保存后测：`http://8.138.152.185:8765/health` 应返回 `{"ok":true,...}`

> **注意：** GitHub Pages 是 **https**，浏览器会拦截 `ws://IP:8765`（混合内容）。  
> Pages 前端必须连 **wss://**（下面 Cloudflare 隧道，或你自己给 8765 配 TLS）。  
> 放行 8765 方便本机 `http://` 直连调试；挂 Pages 仍用隧道 WSS。

## 2. Cloudflare 隧道（给 Pages 用的 WSS）

```bash
bash /tmp/remote_tunnel_service.sh   # 装成 user systemd
journalctl --user -u flatworld-pusht-tunnel -n 50 --no-pager \
  | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | tail -1
```

把打印的 `https://….trycloudflare.com` 写进 `docs/pusht/config.js` 的 `PUSHT_WS`（`wss://…/ws`），再 push 触发 Pages。

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

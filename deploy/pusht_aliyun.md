# Deploy Push-T web backend (private ops)

Keep real hosts / SSH users / key paths in **gitignored** `deploy/local.env`
(see `local.env.example`). Do not paste production IPs or key paths into
README, Pages, or public docs.

## Security notes

- **Private keys never belong in git.** Only a local path in `local.env`.
- Pages must use `wss://` (HTTPS page → plain `ws://` is blocked by browsers).
- A public WSS URL is required for the browser game; treat the game host as a
  public service (SG lockdown, fail2ban/ssh keys only, no password SSH).

## One-time security group

| Port | Purpose |
|------|---------|
| **8765** | Loopback only; remove public security-group/firewall rules |
| **80 + 443** | nginx + Let's Encrypt for Pages WSS |

The app binds to `127.0.0.1:8765`. Only nginx / the local tunnel connects to
it; use HTTPS `/health` or a loopback request over SSH for health checks.
Never expose 8765 publicly, even when using Cloudflare. Every public entry
must reach the same app admission and message limits.

## Browser and abuse limits

The browser origin allowlist defaults to `https://swaggyliu.github.io` and
local development on port 8765. It is enforced on the WebSocket handshake;
HTTP CORS settings alone do not protect WebSockets. For another frontend or
for playing on the backend's own HTTPS page, add its exact origin with
`--allowed-origin https://your-host` or `PUSHT_ALLOWED_ORIGINS` (comma-separated).
Do not use a wildcard. Origins are a browser defense, not authentication.

Defaults: 8 total sessions, 2 per client IP, 6 connection attempts per minute
(token bucket), 90 seconds idle, 60 seconds continuously paused, 30 seconds
after finishing, and 15 minutes maximum per connection. Expiry releases the
slot; press Rematch to reconnect. `--max-sessions-per-ip` can be raised for a
shared NAT. Run one app worker: these quotas are in-process.

Messages are limited to 1 KiB and 60/second with a burst of 90 (the joystick
sends about 40/second). Rematch has a 2-second cooldown; pause/difficulty
share a 250 ms cooldown. Game reset and snapshots run outside the event loop.
Malformed, binary, oversized or excessive messages close the connection.

Uvicorn trusts forwarding headers only from loopback proxies. The public
nginx must overwrite `X-Forwarded-For` with `$remote_addr`, rather than trust
an incoming value. Apply the updated nginx config as well as restarting the
app when upgrading an existing host. If another trusted proxy is added in
front of nginx, configure real client IP resolution for that proxy explicitly.

After upgrading, check `ss -ltnp` over SSH: the listener must be
`127.0.0.1:8765`, never `0.0.0.0:8765` or `[::]:8765`. Verify public HTTPS
`/health` still works and a public connection to 8765 fails. Close the 8765
cloud security-group rule as defense in depth.

## Configure locally

```powershell
copy deploy\local.env.example deploy\local.env
# edit deploy\local.env
```

## Sync / install (Windows)

```powershell
.\deploy\sync_pusht_aliyun.ps1
```

Reads `deploy/local.env`. Remote scripts expect env vars:

- `PUSHT_TLS_HOST` — TLS hostname (e.g. `$IP.sslip.io`)
- `PUSHT_APP_USER` — Linux account that owns the app
- `PUSHT_REMOTE_DIR` — checkout path (default `$HOME/flatworld`)

## Public entry: flexphysics.com (ICP / 小程序 TCP 备案)

Game process stays on node2 (`127.0.0.1:8765`). Gateway
`www.flexphysics.com` already WebSocket-proxies `/cal/task-data/ws/` to
node2 `:9527`. A small bridge in the compute Starlette app exposes Push-T:

| URL | Role |
|-----|------|
| `wss://www.flexphysics.com/cal/task-data/ws/_fw` | Game WebSocket |
| `https://www.flexphysics.com/cal/task-data/ws/_fw/health` | Health |

Install / refresh the bridge on the game host (app user):

```bash
bash deploy/mount_pusht_flexphysics.sh
```

Optional cleaner path `/flatworld/` needs a gateway nginx edit (root); see
`deploy/gateway_nginx_flatworld_pusht.snippet.conf`. Until then the `/_fw`
bridge is the public entry.

Pages `docs/pusht/config.js` points only at the flexphysics.com WSS URL.
Do not put raw ECS IPs, sslip hostnames, SSH users, or key paths in tracked
files — keep those in gitignored `deploy/local.env`.

## nginx TLS on node2 (sslip.io fallback)

User-space (high port, self-signed — not enough for Pages):

```bash
export PUSHT_TLS_HOST=...   # from local.env
bash deploy/remote_nginx_sslip.sh
```

System nginx + certbot (needs sudo + SG 80/443):

```bash
sudo -E bash deploy/remote_nginx_sslip_root.sh
```

## Service

```bash
systemctl --user status flatworld-pusht
journalctl --user -u flatworld-pusht -f
```

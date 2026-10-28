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
| **8765** | Optional direct app HTTP/WS |
| **80 + 443** | nginx + Let's Encrypt for Pages WSS |

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

## nginx TLS (preferred for Pages)

User-space (high port, self-signed — not enough for Pages):

```bash
export PUSHT_TLS_HOST=...   # from local.env
bash deploy/remote_nginx_sslip.sh
```

System nginx + certbot (needs sudo + SG 80/443):

```bash
sudo -E bash deploy/remote_nginx_sslip_root.sh
```

Then point Pages `docs/pusht/config.js` at `wss://$PUSHT_TLS_HOST/ws`
(without committing the literal production host in markdown docs).

## Service

```bash
systemctl --user status flatworld-pusht
journalctl --user -u flatworld-pusht -f
```

# Sync FlatWorld + Push-T weights to aliyun-node2 and install a CPU venv.
$ErrorActionPreference = "Stop"
$Key = "D:\power-center\keys\id_ed25519"
$HostUser = "numericsv@8.138.152.185"
$Remote = "/home/numericsv/flatworld"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$ssh = @("-i", $Key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new")

Write-Host "== mkdir remote =="
ssh @ssh $HostUser "mkdir -p $Remote"

Write-Host "== rsync / scp tree =="
# Prefer scp -r of a filtered tarball (rsync may be missing on Windows).
$tar = Join-Path $env:TEMP "flatworld-pusht.tgz"
if (Test-Path $tar) { Remove-Item $tar -Force }
Push-Location $Root
tar -czf $tar `
  --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude="*.pyc" `
  --exclude=paper --exclude=.cursor --exclude=learning/data `
  --exclude=learning/checkpoints/push_t_rsl `
  --exclude="learning/checkpoints/**/logs" `
  --exclude="learning/checkpoints/**/_*" `
  --exclude=supermimic2d --exclude=test2D --exclude=docs/gallery `
  flatworld learning/checkpoints/push_t learning/checkpoints/push_t_wm_rsl `
  learning/tasks learning/rl learning/models learning/env learning/world_models `
  learning/train_rsl_push_t.py learning/train_rsl_wm_push_t.py `
  learning/planner.py learning/solver_planner.py learning/__init__.py `
  game pyproject.toml requirements.txt README.md
Pop-Location

scp @ssh $tar "${HostUser}:/tmp/flatworld-pusht.tgz"
ssh @ssh $HostUser @"
set -e
mkdir -p $Remote
tar -xzf /tmp/flatworld-pusht.tgz -C $Remote
rm -f /tmp/flatworld-pusht.tgz
cd $Remote
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -U pip wheel
pip install 'warp-lang>=1.7' 'numpy>=1.24' 'scipy>=1.10' 'matplotlib>=3.7' \
  'torch' --index-url https://download.pytorch.org/whl/cpu
pip install 'tensordict' 'rsl-rl-lib' 'fastapi' 'uvicorn[standard]' 'hydra-core' 'omegaconf'
pip install -e .
# user systemd unit
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/flatworld-pusht.service <<'EOF'
[Unit]
Description=FlatWorld Push-T You vs AI WebSocket
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/numericsv/flatworld
Environment=PYTHONUNBUFFERED=1
Environment=FLATWORLD_DEVICE=cpu
ExecStart=/home/numericsv/flatworld/.venv/bin/python -m game.web_server --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now flatworld-pusht.service
systemctl --user status flatworld-pusht.service --no-pager || true
curl -sS http://127.0.0.1:8765/health || true
"@

Write-Host "== done. Open Aliyun SG TCP 8765 if health is only local. =="
Write-Host "Public health: http://8.138.152.185:8765/health"
Write-Host "Play: docs/pusht/  (ws://8.138.152.185:8765/ws)"

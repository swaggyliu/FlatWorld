#!/usr/bin/env bash
# Reuse CI compute-venv (torch+warp). Install only missing deps via Aliyun PyPI.
set -euo pipefail

APP_DIR="${PUSHT_REMOTE_DIR:-$HOME/flatworld}"
VENV="${PUSHT_VENV:-/opt/flexphysics/compute-venv}"
PY="$VENV/bin/python"
export VIRTUAL_ENV="$VENV"
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"

echo "Using $PY (app=$APP_DIR)"
if [ -f /tmp/probe_imports.py ]; then
  "$PY" /tmp/probe_imports.py || true
fi

uv_install() {
  local pkg="$1"
  echo "uv pip install $pkg (index=$UV_INDEX_URL)"
  uv pip install "$pkg"
}

"$PY" -c "import tensordict" 2>/dev/null || uv_install tensordict
"$PY" -c "import rsl_rl" 2>/dev/null || uv_install rsl-rl-lib
"$PY" -c "import hydra" 2>/dev/null || uv_install hydra-core
"$PY" -c "import omegaconf" 2>/dev/null || uv_install omegaconf

cd "$APP_DIR"
rm -rf "$APP_DIR/.venv"
uv pip install -e . --no-deps

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/flatworld-pusht.service" <<EOF
[Unit]
Description=FlatWorld Push-T You vs AI WebSocket
After=network.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=FLATWORLD_DEVICE=cpu
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=VIRTUAL_ENV=${VENV}
ExecStart=${PY} -m game.web_server --host 127.0.0.1 --port 8765 --hz 30 --allow-no-origin
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

loginctl enable-linger "$(whoami)" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable flatworld-pusht.service
systemctl --user restart flatworld-pusht.service
sleep 5
systemctl --user --no-pager status flatworld-pusht.service || true
journalctl --user -u flatworld-pusht.service -n 40 --no-pager || true
curl -sS http://127.0.0.1:8765/health || true
echo
echo REUSE_DONE

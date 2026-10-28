#!/usr/bin/env bash
set -euo pipefail
APP_DIR="${PUSHT_REMOTE_DIR:-$HOME/flatworld}"
cd "$APP_DIR"
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
rm -rf .venv
uv venv .venv --python python3
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install \
  "warp-lang>=1.7" "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7" \
  tensordict "rsl-rl-lib" fastapi "uvicorn[standard]" hydra-core omegaconf meshio
uv pip install -e .
python -c 'import flatworld, torch, fastapi; print("imports_ok", torch.__version__)'

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
ExecStart=${APP_DIR}/.venv/bin/python -m game.web_server --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload || true
systemctl --user enable flatworld-pusht.service || true
systemctl --user restart flatworld-pusht.service || true
sleep 2
systemctl --user --no-pager status flatworld-pusht.service || true
curl -sS http://127.0.0.1:8765/health || true
echo
echo INSTALL_DONE

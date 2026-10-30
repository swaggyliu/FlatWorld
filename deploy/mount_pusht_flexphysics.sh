#!/usr/bin/env bash
# Install Push-T bridge into node2 compute Starlette (flexphysics.com /cal WS).
# Run on the game host as the app user, from a checkout that has deploy/.
set -euo pipefail

RT="${FLEXPHYSICS_RT:-/var/www/flex-physics/webSite/Django-Server/realTimeServer}"
HERE="$(cd "$(dirname "$0")" && pwd)"

test -d "$RT"
test -f "$HERE/flexphysics_pusht_proxy.py"
test -f "$HERE/remote_solver_server_with_pusht.py"

ts="$(date +%Y%m%d%H%M%S)"
if [ -f "$RT/solverServer.py" ] && ! grep -q 'flatworld_ws' "$RT/solverServer.py" 2>/dev/null; then
  cp -a "$RT/solverServer.py" "$RT/solverServer.py.bak.pusht.${ts}"
fi

python3 -c "
from pathlib import Path
rt = Path('$RT')
here = Path('$HERE')
for src, dst in [
    (here / 'flexphysics_pusht_proxy.py', rt / 'flexphysics_pusht_proxy.py'),
    (here / 'remote_solver_server_with_pusht.py', rt / 'solverServer.py'),
]:
    text = src.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    dst.write_text(text, encoding='utf-8')
    print('wrote', dst)
"

sudo -n systemctl restart flexPhysics
sleep 2
systemctl is-active flexPhysics
curl -fsS --connect-timeout 3 http://127.0.0.1:8765/health
echo
curl -fsS --connect-timeout 3 http://127.0.0.1:9527/cal/task-data/ws/_fw/health
echo
echo "Public: https://www.flexphysics.com/cal/task-data/ws/_fw/health"
echo "WSS:    wss://www.flexphysics.com/cal/task-data/ws/_fw"

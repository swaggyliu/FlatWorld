(() => {
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");
  const scoreEl = document.getElementById("score");
  const backendEl = document.getElementById("backend");

  const params = new URLSearchParams(location.search);
  const wsUrl = params.get("ws") || window.PUSHT_WS;
  backendEl.textContent = wsUrl;

  const keys = new Set();
  let state = null;
  let ws = null;
  let reconnectAt = 0;

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function syncKeys() {
    send({ type: "keys", down: [...keys] });
  }

  function connect() {
    statusEl.textContent = "connecting…";
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      statusEl.textContent = "connected";
      send({ type: "hello" });
      syncKeys();
    };
    ws.onclose = () => {
      statusEl.textContent = "disconnected — retrying";
      reconnectAt = performance.now() + 1500;
    };
    ws.onerror = () => {};
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "error") {
          statusEl.textContent = msg.message || "error";
          return;
        }
        state = msg;
        if (msg.status) statusEl.textContent = msg.status;
        if (msg.score) {
          scoreEl.textContent =
            `YOU ${msg.score.you} — AI ${msg.score.ai}` +
            (msg.score.draw ? ` (draw ${msg.score.draw})` : "");
        }
      } catch (_) {}
    };
  }

  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (["w", "a", "s", "d"].includes(k)) {
      e.preventDefault();
      if (!keys.has(k)) {
        keys.add(k);
        syncKeys();
      }
    } else if (k === " " || k === "r") {
      e.preventDefault();
      send({ type: "rematch" });
    } else if (k === "p") {
      e.preventDefault();
      send({ type: "pause" });
    }
  });
  window.addEventListener("keyup", (e) => {
    const k = e.key.toLowerCase();
    if (keys.delete(k)) syncKeys();
  });
  canvas.addEventListener("pointerdown", () => canvas.focus());

  // ---- draw helpers (top-down, same mapping as raylib game) ----
  const SCALE = 280;
  const PAD_Y = 100;

  function worldToScreen(panelX, panelW, x, y) {
    const ox = panelX + panelW * 0.5 - 0.95 * SCALE;
    const oy = canvas.height - PAD_Y;
    return [ox + x * SCALE, oy - y * SCALE];
  }

  function drawBox(panelX, panelW, x, y, th, hw, hh, fill, stroke) {
    const [cx, cy] = worldToScreen(panelX, panelW, x, y);
    const w = 2 * hw * SCALE;
    const h = 2 * hh * SCALE;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-th);
    ctx.fillStyle = fill;
    ctx.fillRect(-w / 2, -h / 2, w, h);
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.strokeRect(-w / 2, -h / 2, w, h);
    }
    ctx.restore();
  }

  function drawSpirit(panelX, panelW, x, y, r, lookX, lookY, happy) {
    const [cx, cy] = worldToScreen(panelX, panelW, x, y);
    const pr = r * SCALE;
    ctx.beginPath();
    ctx.fillStyle = "#62cfbe";
    ctx.arc(cx, cy, pr, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.strokeStyle = "rgba(180,240,230,0.7)";
    ctx.lineWidth = 3;
    ctx.arc(cx, cy, pr * 1.15, 0, Math.PI * 2);
    ctx.stroke();
    const n = Math.hypot(lookX, lookY) || 1;
    const ex = (lookX / n) * pr * 0.35;
    const ey = -(lookY / n) * pr * 0.35;
    ctx.fillStyle = "#122";
    ctx.beginPath();
    ctx.arc(cx + ex - pr * 0.18, cy + ey, pr * 0.12, 0, Math.PI * 2);
    ctx.arc(cx + ex + pr * 0.18, cy + ey, pr * 0.12, 0, Math.PI * 2);
    ctx.fill();
    if (happy) {
      ctx.strokeStyle = "#122";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx + ex, cy + ey + pr * 0.22, pr * 0.22, 0.1, Math.PI - 0.1);
      ctx.stroke();
    }
  }

  function drawTrail(panelX, panelW, trail, color) {
    if (!trail || trail.length < 2) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    for (let i = 0; i < trail.length; i++) {
      const [sx, sy] = worldToScreen(panelX, panelW, trail[i][0], trail[i][1]);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
  }

  function drawPanel(panelX, panelW, side, goal, label, labelColor, bg) {
    ctx.fillStyle = bg;
    ctx.fillRect(panelX, 0, panelW, canvas.height);
    if (!side) return;
    const st = side.states;
    const geom = side.geom;
    if (goal) {
      // translucent ghost T
      drawBox(panelX, panelW, goal.stem[0], goal.stem[1], goal.theta,
        geom[1][0], geom[1][1], "rgba(230,90,100,0.30)", "rgba(230,70,85,0.55)");
      drawBox(panelX, panelW, goal.bar[0], goal.bar[1], goal.theta,
        geom[2][0], geom[2][1], "rgba(230,90,100,0.30)", "rgba(230,70,85,0.55)");
    }
    drawTrail(panelX, panelW, side.trail, "rgba(62,207,190,0.45)");
    for (let i = 1; i < st.length; i++) {
      const fill = i === 1 ? "#e9c46a" : "#c9b8a0";
      drawBox(panelX, panelW, st[i][0], st[i][1], st[i][2],
        geom[i][0], geom[i][1], fill, null);
    }
    if (st.length > 0) {
      const lookX = st.length > 1 ? st[1][0] - st[0][0] : 1;
      const lookY = st.length > 1 ? st[1][1] - st[0][1] : 0;
      drawSpirit(panelX, panelW, st[0][0], st[0][1], geom[0][0],
        lookX, lookY, !!side.ok);
    }
    ctx.fillStyle = labelColor;
    ctx.font = "16px Segoe UI, sans-serif";
    ctx.fillText(label, panelX + 16, canvas.height - 18);
  }

  function frame(t) {
    if (!ws || ws.readyState === WebSocket.CLOSED) {
      if (performance.now() >= reconnectAt) connect();
    }
    const mid = canvas.width / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state) {
      drawPanel(0, mid - 2, state.left, state.goal, "YOU · WASD", "#62cfbe", "#182430");
      drawPanel(mid + 2, mid - 2, state.right, state.goal, "AI · WM+PPO", "#ba8cd6", "#1c2030");
      ctx.fillStyle = "#3c5064";
      ctx.fillRect(mid - 2, 0, 4, canvas.height);

      const overlay = state.phase === "over" ? state.status
        : (state.phase === "countdown"
          ? (state.countdown > 0 ? String(Math.ceil(state.countdown)) : "GO!")
          : "");
      if (overlay) {
        let col = "#e8c46a";
        if (/YOU WIN/i.test(overlay)) col = "#62cfbe";
        else if (/AI WIN/i.test(overlay)) col = "#ba8cd6";
        ctx.fillStyle = col;
        ctx.font = "bold 64px Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(overlay, canvas.width / 2, canvas.height * 0.42);
        ctx.textAlign = "left";
      }
    } else {
      ctx.fillStyle = "#8a93a0";
      ctx.font = "20px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for server…", canvas.width / 2, canvas.height / 2);
      ctx.textAlign = "left";
    }
    requestAnimationFrame(frame);
  }

  connect();
  requestAnimationFrame(frame);
})();

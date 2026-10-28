(() => {
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");
  const scoreEl = document.getElementById("score");
  const diffEl = document.getElementById("difficulty");
  const rematchBtn = document.getElementById("rematch");
  const pauseBtn = document.getElementById("pause");
  const stickEl = document.getElementById("stick");
  const stickBase = stickEl?.querySelector(".stick-base");
  const stickKnob = document.getElementById("stick-knob");

  const params = new URLSearchParams(location.search);
  const endpoints = [
    params.get("ws"),
    window.PUSHT_WS,
    window.PUSHT_WS_FALLBACK,
  ].filter(Boolean);
  let epIndex = 0;
  let wsUrl = endpoints[0] || "";

  let stick = { x: 0, y: 0 };
  let state = null;
  let ws = null;
  let reconnectAt = 0;
  let everOpened = false;
  let retryDelay = 1500;
  let difficulty = "medium";
  let portrait = false;
  let scale = 280;
  let padY = 100;

  // Stick drag state (widget or YOU panel).
  let stickDragging = false;
  let stickSource = null; // "widget" | "panel"
  let panelAim = null;

  // Coalesce stick WS sends (~40Hz) while always keeping the latest vector.
  const STICK_SEND_MS = 25;
  let stickDirty = false;
  let stickLastSend = 0;

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function flushStick(force) {
    if (!stickDirty && !force) return;
    const now = performance.now();
    if (!force && now - stickLastSend < STICK_SEND_MS) return;
    stickDirty = false;
    stickLastSend = now;
    send({ type: "stick", x: stick.x, y: stick.y });
  }

  function syncStick(x, y, immediate) {
    stick.x = x;
    stick.y = y;
    stickDirty = true;
    updateKnob(x, y);
    if (immediate) flushStick(true);
  }

  function setStickFromVector(dx, dy, maxR) {
    const r = Math.hypot(dx, dy);
    let x = 0;
    let y = 0;
    if (r > 1e-6) {
      const clipped = Math.min(r, maxR) / maxR;
      x = (dx / r) * clipped;
      y = (dy / r) * clipped;
    }
    // Deadzone
    if (Math.hypot(x, y) < 0.06) {
      x = 0;
      y = 0;
    }
    syncStick(x, y, false);
  }

  function updateKnob(x, y) {
    if (!stickKnob || !stickBase) return;
    const maxPx = stickBase.clientWidth * 0.28;
    stickKnob.style.transform =
      `translate(${x * maxPx}px, ${-y * maxPx}px)`;
  }

  function setDifficulty(level, rematch) {
    difficulty = level;
    if (diffEl) {
      for (const btn of diffEl.querySelectorAll("button")) {
        btn.classList.toggle("active", btn.dataset.level === level);
      }
    }
    send({ type: "difficulty", level });
    if (rematch) send({ type: "rematch" });
  }

  function rematch() {
    if (!ws || ws.readyState === WebSocket.CLOSED) {
      epIndex = 0;
      wsUrl = endpoints[0] || "";
      everOpened = false;
      retryDelay = 1500;
      connect();
      return;
    }
    send({ type: "rematch" });
  }

  function connect() {
    if (!wsUrl) {
      statusEl.textContent = "offline";
      return;
    }
    statusEl.textContent = "connecting…";
    try {
      ws = new WebSocket(wsUrl);
    } catch (_) {
      statusEl.textContent = "offline";
      return;
    }
    ws.onopen = () => {
      everOpened = true;
      retryDelay = 1500;
      statusEl.textContent = "connected";
      send({ type: "hello" });
      send({ type: "difficulty", level: difficulty });
      syncStick(stick.x, stick.y, true);
    };
    ws.onclose = (event) => {
      state = null;
      if ([1000, 1008, 1009].includes(event.code)) {
        statusEl.textContent = event.reason || "Session ended — press Rematch to reconnect";
        reconnectAt = Infinity;
        return;
      }
      if (!everOpened && epIndex + 1 < endpoints.length) {
        epIndex += 1;
        wsUrl = endpoints[epIndex];
        statusEl.textContent = "reconnecting…";
        reconnectAt = performance.now() + 400;
        return;
      }
      statusEl.textContent = "disconnected — retrying";
      reconnectAt = performance.now() + retryDelay + Math.random() * 500;
      retryDelay = Math.min(30000, retryDelay * 2);
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
        if (msg.difficulty && msg.difficulty !== difficulty) {
          difficulty = msg.difficulty;
          if (diffEl) {
            for (const btn of diffEl.querySelectorAll("button")) {
              btn.classList.toggle(
                "active", btn.dataset.level === difficulty);
            }
          }
        }
        if (msg.score) {
          scoreEl.textContent =
            `YOU ${msg.score.you} — AI ${msg.score.ai}` +
            (msg.score.draw ? ` (draw ${msg.score.draw})` : "");
        }
      } catch (_) {}
    };
  }

  function layoutMode() {
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    const narrow = window.innerWidth < 720;
    return coarse || narrow;
  }

  function resizeCanvas() {
    portrait = layoutMode();
    const cssW = Math.min(canvas.clientWidth || window.innerWidth - 24, 1100);
    let w;
    let h;
    if (portrait) {
      w = Math.round(Math.min(640, cssW * (window.devicePixelRatio > 1.5 ? 1.25 : 1.5)));
      h = Math.round(w * 1.55);
    } else {
      w = 1100;
      h = 560;
    }
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const panelW = portrait ? w : w / 2;
    const panelH = portrait ? h / 2 : h;
    scale = Math.min(panelW * 0.42, panelH * 0.55);
    padY = Math.max(48, panelH * 0.16);
  }

  function youPanelRect() {
    if (portrait) {
      return { x: 0, y: 0, w: canvas.width, h: canvas.height / 2 - 2 };
    }
    return { x: 0, y: 0, w: canvas.width / 2 - 2, h: canvas.height };
  }

  function canvasPointFromEvent(e) {
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (canvas.width / Math.max(r.width, 1));
    const cy = (e.clientY - r.top) * (canvas.height / Math.max(r.height, 1));
    return [cx, cy];
  }

  function worldToScreen(panelX, panelY, panelW, panelH, x, y) {
    const ox = panelX + panelW * 0.5 - 0.95 * scale;
    const oy = panelY + panelH - padY;
    return [ox + x * scale, oy - y * scale];
  }

  function screenToWorld(panelX, panelY, panelW, panelH, sx, sy) {
    const ox = panelX + panelW * 0.5 - 0.95 * scale;
    const oy = panelY + panelH - padY;
    return [(sx - ox) / scale, (oy - sy) / scale];
  }

  // ---- virtual joystick widget ----
  function onStickPointerDown(e) {
    if (!stickBase) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    stickDragging = true;
    stickSource = "widget";
    stickBase.setPointerCapture?.(e.pointerId);
    moveStickFromEvent(e);
    flushStick(true);
    e.preventDefault();
  }

  function moveStickFromEvent(e) {
    const r = stickBase.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    const dx = e.clientX - cx;
    // Screen Y down → stick Y up
    const dy = cy - e.clientY;
    setStickFromVector(dx, dy, r.width * 0.42);
  }

  function onStickPointerMove(e) {
    if (!stickDragging || stickSource !== "widget") return;
    moveStickFromEvent(e);
    e.preventDefault();
  }

  function endStickDrag(source, e) {
    if (!stickDragging || (source && stickSource !== source)) return;
    stickDragging = false;
    stickSource = null;
    panelAim = null;
    syncStick(0, 0, true);
    if (e) {
      try {
        const el = source === "panel" ? canvas : stickBase;
        el?.releasePointerCapture?.(e.pointerId);
      } catch (_) {}
    }
  }

  function onStickPointerUp(e) {
    endStickDrag("widget", e);
  }

  if (stickBase) {
    stickBase.addEventListener("pointerdown", onStickPointerDown);
    stickBase.addEventListener("pointermove", onStickPointerMove);
    stickBase.addEventListener("pointerup", onStickPointerUp);
    stickBase.addEventListener("pointercancel", onStickPointerUp);
    stickBase.addEventListener("lostpointercapture", () => endStickDrag("widget"));
  }

  // ---- drag on YOU panel also drives the same stick ----
  function onCanvasPointerDown(e) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (stickDragging && stickSource === "widget") return;
    const [sx, sy] = canvasPointFromEvent(e);
    const you = youPanelRect();
    if (sx >= you.x && sx <= you.x + you.w && sy >= you.y && sy <= you.y + you.h) {
      stickDragging = true;
      stickSource = "panel";
      canvas.setPointerCapture?.(e.pointerId);
      updatePanelStick(sx, sy);
      flushStick(true);
      e.preventDefault();
    }
  }

  function updatePanelStick(sx, sy) {
    const you = youPanelRect();
    const world = screenToWorld(you.x, you.y, you.w, you.h, sx, sy);
    panelAim = world;
    let ox = 0.95;
    let oy = 0.55;
    if (state?.left?.states?.[0]) {
      ox = state.left.states[0][0];
      oy = state.left.states[0][1];
    }
    setStickFromVector(world[0] - ox, world[1] - oy, 0.55);
  }

  function onCanvasPointerMove(e) {
    if (!stickDragging || stickSource !== "panel") return;
    const [sx, sy] = canvasPointFromEvent(e);
    updatePanelStick(sx, sy);
    e.preventDefault();
  }

  function onCanvasPointerUp(e) {
    endStickDrag("panel", e);
  }

  canvas.addEventListener("pointerdown", onCanvasPointerDown);
  canvas.addEventListener("pointermove", onCanvasPointerMove);
  canvas.addEventListener("pointerup", onCanvasPointerUp);
  canvas.addEventListener("pointercancel", onCanvasPointerUp);
  canvas.addEventListener("lostpointercapture", () => endStickDrag("panel"));

  // Safety: blur / tab hide zeros the stick so force never sticks.
  function zeroStickIfNeeded() {
    if (stickDragging || stick.x || stick.y) {
      stickDragging = false;
      stickSource = null;
      panelAim = null;
      syncStick(0, 0, true);
    }
  }
  window.addEventListener("blur", zeroStickIfNeeded);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) zeroStickIfNeeded();
  });

  if (diffEl) {
    diffEl.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-level]");
      if (!btn) return;
      setDifficulty(btn.dataset.level, true);
    });
  }
  rematchBtn?.addEventListener("click", () => rematch());
  pauseBtn?.addEventListener("click", () => send({ type: "pause" }));

  // Shortcuts: rematch / pause / difficulty only (stick is the sole move input).
  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (k === " " || k === "r") {
      e.preventDefault();
      rematch();
    } else if (k === "p") {
      e.preventDefault();
      send({ type: "pause" });
    } else if (k === "1") {
      e.preventDefault();
      setDifficulty("low", true);
    } else if (k === "2") {
      e.preventDefault();
      setDifficulty("medium", true);
    } else if (k === "3") {
      e.preventDefault();
      setDifficulty("high", true);
    }
  });
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("orientationchange", () => setTimeout(resizeCanvas, 150));

  function drawBox(panelX, panelY, panelW, panelH, x, y, th, hw, hh, fill, stroke) {
    const [cx, cy] = worldToScreen(panelX, panelY, panelW, panelH, x, y);
    const w = 2 * hw * scale;
    const h = 2 * hh * scale;
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

  function drawSpirit(panelX, panelY, panelW, panelH, x, y, r, lookX, lookY, happy) {
    const [cx, cy] = worldToScreen(panelX, panelY, panelW, panelH, x, y);
    const pr = r * scale;
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

  function drawTrail(panelX, panelY, panelW, panelH, trail, color) {
    if (!trail || trail.length < 2) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    for (let i = 0; i < trail.length; i++) {
      const [sx, sy] = worldToScreen(
        panelX, panelY, panelW, panelH, trail[i][0], trail[i][1]);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
  }

  function drawPanel(panelX, panelY, panelW, panelH, side, goal, label, labelColor, bg) {
    ctx.fillStyle = bg;
    ctx.fillRect(panelX, panelY, panelW, panelH);
    if (!side) return;
    const st = side.states;
    const geom = side.geom;
    if (goal) {
      drawBox(panelX, panelY, panelW, panelH, goal.stem[0], goal.stem[1], goal.theta,
        geom[1][0], geom[1][1], "rgba(230,90,100,0.30)", "rgba(230,70,85,0.55)");
      drawBox(panelX, panelY, panelW, panelH, goal.bar[0], goal.bar[1], goal.theta,
        geom[2][0], geom[2][1], "rgba(230,90,100,0.30)", "rgba(230,70,85,0.55)");
    }
    drawTrail(panelX, panelY, panelW, panelH, side.trail, "rgba(62,207,190,0.45)");
    for (let i = 1; i < st.length; i++) {
      const fill = i === 1 ? "#e9c46a" : "#c9b8a0";
      drawBox(panelX, panelY, panelW, panelH, st[i][0], st[i][1], st[i][2],
        geom[i][0], geom[i][1], fill, null);
    }
    if (st.length > 0) {
      const lookX = st.length > 1 ? st[1][0] - st[0][0] : 1;
      const lookY = st.length > 1 ? st[1][1] - st[0][1] : 0;
      drawSpirit(panelX, panelY, panelW, panelH, st[0][0], st[0][1], geom[0][0],
        lookX, lookY, !!side.ok);
    }
    if (panelAim && label.startsWith("YOU") && stickDragging
        && stickSource === "panel" && st && st.length > 0) {
      const [ax, ay] = worldToScreen(
        panelX, panelY, panelW, panelH, panelAim[0], panelAim[1]);
      const [sx, sy] = worldToScreen(
        panelX, panelY, panelW, panelH, st[0][0], st[0][1]);
      ctx.beginPath();
      ctx.strokeStyle = "rgba(98,207,190,0.85)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.moveTo(sx, sy);
      ctx.lineTo(ax, ay);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.fillStyle = "rgba(98,207,190,0.9)";
      ctx.arc(ax, ay, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = labelColor;
    ctx.font = `${portrait ? 14 : 16}px Segoe UI, sans-serif`;
    ctx.fillText(label, panelX + 12, panelY + panelH - 12);
  }

  function frame() {
    if (!ws || ws.readyState === WebSocket.CLOSED) {
      if (performance.now() >= reconnectAt) connect();
    }
    flushStick(false);
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let youRect;
    let aiRect;
    if (portrait) {
      const mid = canvas.height / 2;
      youRect = { x: 0, y: 0, w: canvas.width, h: mid - 2 };
      aiRect = { x: 0, y: mid + 2, w: canvas.width, h: mid - 2 };
      ctx.fillStyle = "#3c5064";
      ctx.fillRect(0, mid - 2, canvas.width, 4);
    } else {
      const mid = canvas.width / 2;
      youRect = { x: 0, y: 0, w: mid - 2, h: canvas.height };
      aiRect = { x: mid + 2, y: 0, w: mid - 2, h: canvas.height };
      ctx.fillStyle = "#3c5064";
      ctx.fillRect(mid - 2, 0, 4, canvas.height);
    }

    if (state) {
      drawPanel(youRect.x, youRect.y, youRect.w, youRect.h,
        state.left, state.goal, "YOU · stick", "#62cfbe", "#182430");
      drawPanel(aiRect.x, aiRect.y, aiRect.w, aiRect.h,
        state.right, state.goal, "AI · WM+PPO", "#ba8cd6", "#1c2030");

      const overlay = state.phase === "over" ? state.status
        : (state.phase === "countdown"
          ? (state.countdown > 0 ? String(Math.ceil(state.countdown)) : "GO!")
          : "");
      if (overlay) {
        let col = "#e8c46a";
        if (/YOU WIN/i.test(overlay)) col = "#62cfbe";
        else if (/AI WIN/i.test(overlay)) col = "#ba8cd6";
        ctx.fillStyle = col;
        ctx.font = `bold ${portrait ? 42 : 64}px Segoe UI, sans-serif`;
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

  if (diffEl) setDifficulty(difficulty, false);
  resizeCanvas();
  connect();
  requestAnimationFrame(frame);
})();

(() => {
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");
  const scoreEl = document.getElementById("score");
  const scoreValues = {
    you: document.getElementById("score-you"),
    ai: document.getElementById("score-ai"),
    draw: document.getElementById("score-draw"),
  };
  const scorePanels = {
    you: document.getElementById("score-you-panel"),
    ai: document.getElementById("score-ai-panel"),
  };
  const scoreLabels = {
    you: document.getElementById("score-you-label"),
    ai: document.getElementById("score-ai-label"),
  };
  let previousScore = null;
  const diffEl = document.getElementById("difficulty");
  const rematchBtn = document.getElementById("rematch");
  const pauseBtn = document.getElementById("pause");
  const settingsBtn = document.getElementById("settings");
  const settingsDialog = document.getElementById("settings-dialog");
  const controlSelect = document.getElementById("control-mode");
  const durationInput = document.getElementById("round-duration");
  const musicToggle = document.getElementById("music-enabled");
  const sfxToggle = document.getElementById("sfx-enabled");
  const hintEl = document.getElementById("control-hint");
  const clockEl = document.getElementById("round-clock");
  const timeEl = document.getElementById("time-left");
  const progressEl = document.getElementById("time-progress");
  const validDuration = value => Number.isInteger(value) && value >= 10 && value <= 180;
  const preferences = { mode: "drag", music: true, sfx: true, duration: 30 };
  try {
    const saved = JSON.parse(localStorage.getItem("flatworld-pusht-settings") || "null");
    if (saved?.mode === "drag" || saved?.mode === "wasd") preferences.mode = saved.mode;
    if (typeof saved?.music === "boolean") preferences.music = saved.music;
    if (typeof saved?.sfx === "boolean") preferences.sfx = saved.sfx;
    if (validDuration(saved?.duration)) preferences.duration = saved.duration;
  } catch (_) {}
  const sound = window.PushTAudio ? new window.PushTAudio() : {
    unlock() {}, configure() {}, setPlaying() {}, push() {}, win() {},
  };
  sound.configure(preferences.music, preferences.sfx);

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
  let clock = null;
  const motion = new window.PushTMotion();
  let ws = null;
  let reconnectAt = 0;
  let everOpened = false;
  let retryDelay = 1500;
  let difficulty = "medium";
  let portrait = false;
  let scale = 280;
  let padY = 100;

  // Drag directly on the player's panel; both modes send the same bounded vector.
  let stickDragging = false;
  let pointerId = null;
  let panelAim = null;
  const heldKeys = new Set();

  // Coalesce stick WS sends (~60Hz) while always keeping the latest vector.
  const STICK_SEND_MS = 16;
  let stickDirty = false;
  let stickLastSend = 0;
  let onlineCount = 0;

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function updateScore(msg) {
    const score = msg.score;
    if (!score || !scoreEl) return;
    // Keep cached pages with the original single-line score working too.
    if (!scoreValues.you || !scoreValues.ai || !scoreValues.draw) {
      scoreEl.textContent = `YOU ${score.you} — AI ${score.ai}` +
        (score.draw ? ` (draw ${score.draw})` : "");
      return;
    }
    for (const key of ["you", "ai", "draw"]) {
      const value = score[key] ?? 0;
      const el = scoreValues[key];
      if (el.textContent !== String(value)) el.textContent = String(value);
      el.classList.toggle("is-long", String(value).length > 3);
      if (previousScore && value > (previousScore[key] ?? 0)
          && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        el.animate?.([
          { transform: "scale(1)" },
          { transform: "scale(1.2)", filter: "brightness(1.5)", offset: 0.4 },
          { transform: "scale(1)" },
        ], { duration: 420, easing: "ease-out" });
      }
    }
    for (const key of ["you", "ai"]) {
      const won = msg.phase === "over" && msg.winner === (key === "you" ? "player" : "ai");
      const leading = score[key] > score[key === "you" ? "ai" : "you"];
      const result = won ? "winner" : leading ? "leading" : "";
      if (scorePanels[key]) scorePanels[key].dataset.result = result;
      if (scoreLabels[key]) scoreLabels[key].textContent = won ? "ROUND WINNER" : leading ? "LEADING" : "WINS";
    }
    scoreEl.setAttribute("aria-label", `Match score: You ${score.you}, AI ${score.ai}, draws ${score.draw ?? 0}`);
    previousScore = { ...score };
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

  function setDifficulty(level, rematch) {
    difficulty = level;
    if (diffEl) {
      for (const btn of diffEl.querySelectorAll("button")) {
        btn.classList.toggle("active", btn.dataset.level === level);
      }
    }
    send({ type: "difficulty", level });
    if (rematch) sendRematch();
  }

  function sendRematch() {
    send({ type: "rematch", round_duration: preferences.duration });
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
    sendRematch();
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
      send({ type: "hello", round_duration: preferences.duration });
      send({ type: "difficulty", level: difficulty });
      syncStick(stick.x, stick.y, true);
    };
    ws.onclose = (event) => {
      previousScore = null;
      state = null;
      clock = null;
      motion.clear();
      zeroStickIfNeeded();
      updateSound();
      updateClock();
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
        if (msg.type !== "state") return;
        const previousPhase = state?.phase;
        state = msg;
        motion.push(msg, performance.now());
        receiveClock(msg);
        if (msg.paused || msg.phase === "over") zeroStickIfNeeded();
        updateClock();
        updateSound();
        if (msg.phase === "over" && previousPhase && previousPhase !== "over" && msg.winner === "player") {
          sound.win();
        }
        playPushSound(msg);
        if (pauseBtn) {
          pauseBtn.textContent = msg.paused ? "Resume" : "Pause";
          pauseBtn.setAttribute("aria-pressed", String(!!msg.paused));
        }
        if (msg.status && statusEl.textContent !== msg.status) statusEl.textContent = msg.status;
        if (msg.difficulty && msg.difficulty !== difficulty) {
          difficulty = msg.difficulty;
          if (diffEl) {
            for (const btn of diffEl.querySelectorAll("button")) {
              btn.classList.toggle(
                "active", btn.dataset.level === difficulty);
            }
          }
        }
        updateScore(msg);
        updateOnline(msg);
      } catch (_) {}
    };
  }

  function updateOnline(msg) {
    const n = Number(msg.online);
    if (!Number.isFinite(n) || n < 1) return;
    onlineCount = Math.max(1, Math.floor(n));
    const el = document.getElementById("online");
    if (!el) return;
    el.textContent = onlineCount === 1
      ? "Online · 1 player"
      : `Online · ${onlineCount} players`;
  }

  // Display-only EE lead toward the stick so drag doesn't wait on the next tick.
  function predictYouSide(side) {
    if (!side?.states?.length) return side;
    const sx = stick.x;
    const sy = stick.y;
    if (!sx && !sy) return side;
    const lead = 0.045;  // metres of visual anticipation
    const states = side.states.map((pose, i) => {
      if (i !== 0) return pose;
      const out = pose.slice();
      out[0] = pose[0] + sx * lead;
      out[1] = pose[1] + sy * lead;
      return out;
    });
    return { ...side, states };
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

  function canMove() {
    return !!state && !state.paused && state.phase !== "over" && !settingsDialog?.open;
  }

  // Dragging sets a world-space target, so holding a finger still also steers
  // toward that point as the player moves, rather than applying stale force.
  function onCanvasPointerDown(e) {
    sound.unlock();
    if (preferences.mode !== "drag" || !canMove() || stickDragging) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const [sx, sy] = canvasPointFromEvent(e);
    const you = youPanelRect();
    if (sx >= you.x && sx <= you.x + you.w && sy >= you.y && sy <= you.y + you.h) {
      stickDragging = true;
      pointerId = e.pointerId;
      canvas.focus({ preventScroll: true });
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
    steerToAim();
  }

  function steerToAim() {
    if (!panelAim) return;
    let ox = 0.95;
    let oy = 0.55;
    if (state?.left?.states?.[0]) {
      ox = state.left.states[0][0];
      oy = state.left.states[0][1];
    }
    setStickFromVector(panelAim[0] - ox, panelAim[1] - oy, 0.55);
  }

  function onCanvasPointerMove(e) {
    if (!stickDragging || e.pointerId !== pointerId) return;
    const [sx, sy] = canvasPointFromEvent(e);
    updatePanelStick(sx, sy);
    flushStick(true);  // drag must feel live; don't wait for coalesce
    e.preventDefault();
  }

  function onCanvasPointerUp(e) {
    if (e.pointerId === pointerId) zeroStickIfNeeded();
  }

  canvas.addEventListener("pointerdown", onCanvasPointerDown);
  canvas.addEventListener("pointermove", onCanvasPointerMove);
  canvas.addEventListener("pointerup", onCanvasPointerUp);
  canvas.addEventListener("pointercancel", onCanvasPointerUp);
  canvas.addEventListener("lostpointercapture", onCanvasPointerUp);

  // Safety: blur / tab hide zeros the stick so force never sticks.
  function zeroStickIfNeeded() {
    heldKeys.clear();
    const captured = pointerId;
    pointerId = null;
    if (stickDragging || stick.x || stick.y) {
      stickDragging = false;
      panelAim = null;
      syncStick(0, 0, true);
    }
    if (captured !== null) {
      try { canvas.releasePointerCapture?.(captured); } catch (_) {}
    }
  }
  window.addEventListener("blur", zeroStickIfNeeded);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) zeroStickIfNeeded();
    updateSound();
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

  function movementKey(e) {
    return { KeyW: "w", KeyA: "a", KeyS: "s", KeyD: "d" }[e.code]
      || ("wasd".includes(e.key.toLowerCase()) && e.key.length === 1 ? e.key.toLowerCase() : null);
  }

  function syncKeys() {
    let x = Number(heldKeys.has("d")) - Number(heldKeys.has("a"));
    let y = Number(heldKeys.has("w")) - Number(heldKeys.has("s"));
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    syncStick(x, y, true);
  }

  window.addEventListener("keydown", (e) => {
    sound.unlock();
    if (settingsDialog?.open || e.target?.closest?.("input, select, textarea, button, a, [contenteditable]")) return;
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    const move = movementKey(e);
    if (move && preferences.mode === "wasd" && canMove()) {
      e.preventDefault();
      if (!heldKeys.has(move)) { heldKeys.add(move); syncKeys(); }
      return;
    }
    if (e.repeat) return;
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
  window.addEventListener("keyup", (e) => {
    const move = movementKey(e);
    if (move && heldKeys.delete(move)) { e.preventDefault(); syncKeys(); }
  });
  window.addEventListener("pointerdown", () => sound.unlock(), { passive: true });
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("orientationchange", () => setTimeout(resizeCanvas, 150));

  function updatePreferences() {
    if (controlSelect) controlSelect.value = preferences.mode;
    if (durationInput) durationInput.value = preferences.duration;
    if (musicToggle) musicToggle.checked = preferences.music;
    if (sfxToggle) sfxToggle.checked = preferences.sfx;
    if (hintEl) hintEl.textContent = preferences.mode === "drag"
      ? "Drag anywhere on the YOU panel to move · release to stop · Settings for WASD"
      : "W A S D to move · release to stop · Settings for drag controls";
    canvas.dataset.controls = preferences.mode;
    canvas.setAttribute("aria-label", preferences.mode === "drag"
      ? "You versus AI. Drag on the player panel to move."
      : "You versus AI. Use W A S D to move.");
    sound.configure(preferences.music, preferences.sfx);
    try { localStorage.setItem("flatworld-pusht-settings", JSON.stringify(preferences)); } catch (_) {}
  }

  settingsBtn?.addEventListener("click", () => {
    zeroStickIfNeeded();
    sound.unlock();
    settingsDialog?.showModal();
  });
  function saveDuration() {
    if (!durationInput) return;
    const seconds = Number(durationInput.value);
    if (validDuration(seconds)) preferences.duration = seconds;
    updatePreferences();
  }
  document.getElementById("settings-close")?.addEventListener("click", () => {
    saveDuration();
    settingsDialog?.close();
  });
  settingsDialog?.addEventListener("click", (e) => {
    const rect = settingsDialog.getBoundingClientRect();
    if (e.target === settingsDialog && (e.clientX < rect.left || e.clientX > rect.right
        || e.clientY < rect.top || e.clientY > rect.bottom)) settingsDialog.close();
  });
  settingsDialog?.addEventListener("close", () => {
    saveDuration();
    canvas.focus({ preventScroll: true });
  });
  controlSelect?.addEventListener("change", () => {
    zeroStickIfNeeded();
    preferences.mode = controlSelect.value === "wasd" ? "wasd" : "drag";
    updatePreferences();
  });
  durationInput?.addEventListener("change", saveDuration);
  musicToggle?.addEventListener("change", () => {
    preferences.music = musicToggle.checked;
    sound.unlock();
    updatePreferences();
  });
  sfxToggle?.addEventListener("change", () => {
    preferences.sfx = sfxToggle.checked;
    sound.unlock();
    updatePreferences();
  });

  function receiveClock(msg) {
    const now = performance.now();
    const steps = [msg.left?.t, msg.right?.t].filter(value => Number.isFinite(value) && value >= 0);
    const step = steps.length ? Math.max(...steps) : null;
    let duration = msg.round_duration;
    let remaining = msg.time_remaining;
    const authoritative = Number.isFinite(duration) && duration > 0 && Number.isFinite(remaining);
    const source = authoritative ? "server" : "legacy";
    const reset = !clock || msg.phase === "countdown"
      || (msg.round_id != null && clock.roundId != null && msg.round_id !== clock.roundId)
      || (step != null && clock.step != null && step < clock.step)
      || (authoritative && clock.source === "server" && duration !== clock.duration);
    if (!authoritative) {
      // Old servers have a 1,100-step budget but may omit or repeat counters.
      // Carry elapsed time between messages instead of restarting the clock
      // on every snapshot. Step progress can move the estimate forward too.
      const stepsPerSecond = 30 * 3;
      duration = 1100 / stepsPerSecond;
      remaining = duration;
      if (!reset && clock.source === source) remaining = clockRemaining(now);
      if (step != null && msg.phase !== "countdown") {
        remaining = Math.min(remaining, duration - step / stepsPerSecond);
      }
    }
    // Repeated server timestamps must not undo progress between snapshots.
    if (!reset && clock.source === source && clock.running && msg.phase === "race") {
      remaining = Math.min(remaining, clockRemaining(now));
    }
    clock = { duration, remaining: Math.max(0, Math.min(duration, remaining)),
      receivedAt: now, running: msg.phase === "race" && !msg.paused,
      source, step, roundId: msg.round_id };
  }

  function clockRemaining(now) {
    // Stop extrapolating after a second without snapshots on a stale connection.
    const elapsed = clock.running ? Math.min(1, Math.max(0, (now - clock.receivedAt) / 1000)) : 0;
    return Math.max(0, clock.remaining - elapsed);
  }

  function updateClock() {
    if (!clockEl || !timeEl || !progressEl) return;
    if (!clock) {
      progressEl.removeAttribute("value");
      timeEl.textContent = "—";
      progressEl.setAttribute("aria-valuetext", "Waiting for game");
      clockEl.dataset.urgent = "false";
      return;
    }
    const remaining = clockRemaining(performance.now());
    progressEl.max = clock.duration;
    progressEl.value = remaining;
    timeEl.textContent = `${Math.ceil(remaining * 10) / 10}s${state?.paused ? " · paused" : ""}`;
    progressEl.setAttribute("aria-valuetext", `${timeEl.textContent} remaining`);
    clockEl.dataset.urgent = String(remaining <= 5 && clock.running);
  }

  function updateSound() {
    sound.setPlaying(!!state && !state.paused && state.phase !== "over", !document.hidden);
  }

  function playPushSound(msg) {
    if (msg.phase !== "race" || msg.paused || !msg.left?.states || !msg.left?.geom) return;
    const states = msg.left.states;
    const geom = msg.left.geom;
    const player = states[0];
    if (!player || !geom[0]) return;
    for (let i = 1; i < states.length; i++) {
      const box = states[i];
      if (!geom[i]) continue;
      const dx = player[0] - box[0], dy = player[1] - box[1];
      const c = Math.cos(box[2]), s = Math.sin(box[2]);
      const lx = Math.abs(c * dx + s * dy), ly = Math.abs(-s * dx + c * dy);
      const distance = Math.hypot(Math.max(0, lx - geom[i][0]), Math.max(0, ly - geom[i][1]));
      const speed = Math.hypot(box[3] || 0, box[4] || 0);
      const force = Math.hypot(stick.x, stick.y);
      if (distance <= geom[0][0] + 0.02 && (speed > 0.025 || force > 0.1)) {
        sound.push(Math.min(1, speed * 2 + force * 0.3));
        break;
      }
    }
  }

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
    if (panelAim && label.startsWith("YOU") && stickDragging && st && st.length > 0) {
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
    if (stickDragging && canMove()) steerToAim();
    flushStick(false);
    updateClock();
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
      // YOU: latest server pose (no interpolation delay) so drag tracks the finger.
      // AI: short motion buffer to hide network jitter.
      const rendered = motion.sample(performance.now()) || state;
      const youSide = predictYouSide(state.left);
      const aiSide = rendered.right || state.right;
      drawPanel(youRect.x, youRect.y, youRect.w, youRect.h,
        youSide, state.goal, preferences.mode === "drag" ? "YOU · drag" : "YOU · WASD", "#62cfbe", "#182430");
      drawPanel(aiRect.x, aiRect.y, aiRect.w, aiRect.h,
        aiSide, state.goal, "AI · WM+PPO", "#ba8cd6", "#1c2030");

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
  updatePreferences();
  updateClock();
  resizeCanvas();
  connect();
  requestAnimationFrame(frame);
})();

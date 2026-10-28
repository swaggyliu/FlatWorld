const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function client(saved = null) {
  const sockets = [];
  const callbacks = {};
  let now = 0;
  let nextFrame;
  const windowEvents = {};
  const documentEvents = {};
  const stored = new Map(saved ? [["flatworld-pusht-settings", JSON.stringify(saved)]] : []);
  function element() {
    return {
      dataset: {}, attributes: {}, listeners: {}, open: false,
      addEventListener(type, callback) { this.listeners[type] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      focus() {}, setPointerCapture() {}, releasePointerCapture() {},
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 1100, bottom: 560, width: 1100, height: 560 }),
      showModal() { this.open = true; },
      close() { this.open = false; this.listeners.close?.(); },
    };
  }
  const ctx = new Proxy({}, { get: () => () => {} });
  const canvas = {
    ...element(),
    width: 1100, height: 560, clientWidth: 1100,
    getContext: () => ctx,
  };
  const status = { textContent: "" };
  const elements = {
    c: canvas, status, score: {},
    rematch: element(), pause: element(),
  };
  for (const id of ["settings", "settings-dialog", "settings-close", "control-mode", "music-enabled",
    "sfx-enabled", "control-hint", "round-clock", "time-left", "time-progress"]) elements[id] = element();
  const sound = { pushes: [], unlocks: 0,
    unlock() { this.unlocks++; },
    configure(music, sfx) { this.music = music; this.sfx = sfx; },
    setPlaying(playing, visible) { this.playing = playing; this.visible = visible; },
    push(strength) { this.pushes.push(strength); },
  };
  class WebSocket {
    static OPEN = 1;
    static CLOSED = 3;
    readyState = 0;
    sent = [];
    constructor(url) { this.url = url; sockets.push(this); }
    send(message) { this.sent.push(JSON.parse(message)); }
  }
  const sandbox = {
    document: { hidden: false, getElementById: (id) => elements[id] || null,
      addEventListener: (type, callback) => { documentEvents[type] = callback; } },
    window: { PUSHT_WS: "wss://game.example/ws", innerWidth: 1100,
      PushTAudio: class { constructor() { return sound; } },
      devicePixelRatio: 1, matchMedia: () => ({ matches: false }),
      addEventListener: (type, callback) => { windowEvents[type] = callback; } },
    localStorage: { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) },
    location: { search: "" }, URLSearchParams, WebSocket,
    performance: { now: () => now },
    requestAnimationFrame: (callback) => { nextFrame = callback; },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../../docs/pusht/app.js"), "utf8"), sandbox);
  callbacks.rematch = elements.rematch.listeners.click;
  function event(extra) { return { preventDefault() {}, key: "", pointerId: 1, pointerType: "mouse", button: 0, ...extra }; }
  return { sockets, callbacks, status, elements, sound, stored,
    key(type, key, extra = {}) { windowEvents[type](event({ key, ...extra })); },
    pointer(type, x, y, extra = {}) { canvas.listeners[type](event({ clientX: x, clientY: y, ...extra })); },
    blur() { windowEvents.blur(); },
    hide() { sandbox.document.hidden = true; documentEvents.visibilitychange(); },
    snapshot(overrides = {}) {
      const side = { states: [[0.95, 0.55, 0, 0, 0], [1.2, 0.5, 0, 0, 0], [1.2, 0.8, 0, 0, 0]],
        geom: [[0.05, 0.05], [0.08, 0.28], [0.32, 0.08]] };
      sockets.at(-1).onmessage({ data: JSON.stringify({ type: "state", phase: "race", paused: false,
        round_duration: 22.222, time_remaining: 22.222, left: side, right: side, ...overrides }) });
    },
    open() { const ws = sockets.at(-1); ws.readyState = 1; ws.onopen(); },
    close(code, reason = "") { const ws = sockets.at(-1); ws.readyState = 3; ws.onclose({ code, reason }); },
    frame(time) { now = time; nextFrame(); },
  };
}

for (const code of [1000, 1008, 1009]) {
  test(`close ${code} stays disconnected until the player requests a rematch`, () => {
    const app = client();
    app.open();
    app.close(code, "Session idle. Press Rematch to reconnect.");
    app.frame(100000);
    assert.equal(app.sockets.length, 1);
    assert.match(app.status.textContent, /Rematch/);
    app.callbacks.rematch();
    assert.equal(app.sockets.length, 2);
    app.open();
    assert.equal(app.sockets[1].sent[0].type, "hello");
  });
}

test("network failures reconnect with increasing delay", () => {
  const app = client();
  app.open();
  app.close(1006);
  app.frame(1400);
  assert.equal(app.sockets.length, 1);
  app.frame(2100);
  assert.equal(app.sockets.length, 2);
  app.close(1006);
  app.frame(4500);
  assert.equal(app.sockets.length, 2);
  app.frame(5700);
  assert.equal(app.sockets.length, 3);
});

test("drag is the default and only dragging the player panel moves the player", () => {
  const app = client();
  app.open();
  app.snapshot();
  assert.equal(app.elements.c.dataset.controls, "drag");
  app.sockets[0].sent = [];
  app.key("keydown", "w", { code: "KeyW" });
  assert.equal(app.sockets[0].sent.length, 0);
  app.pointer("pointerdown", 850, 300);
  assert.equal(app.sockets[0].sent.length, 0);
  app.pointer("pointerdown", 350, 300);
  assert.ok(app.sockets[0].sent.at(-1).x > 0);
  app.pointer("pointerup", 350, 300);
  assert.equal(app.sockets[0].sent.at(-1).x, 0);
  assert.equal(app.sockets[0].sent.at(-1).y, 0);
});

test("WASD, diagonal movement, mode switches and blur never leave a held force", () => {
  const app = client();
  app.open();
  app.snapshot();
  app.elements["control-mode"].value = "wasd";
  app.elements["control-mode"].listeners.change();
  app.key("keydown", "w", { code: "KeyW" });
  app.key("keydown", "d", { code: "KeyD" });
  const diagonal = app.sockets[0].sent.at(-1);
  assert.ok(Math.abs(Math.hypot(diagonal.x, diagonal.y) - 1) < 1e-9);
  app.key("keyup", "w", { code: "KeyW" });
  assert.equal(app.sockets[0].sent.at(-1).x, 1);
  assert.equal(app.sockets[0].sent.at(-1).y, 0);
  app.blur();
  assert.equal(app.sockets[0].sent.at(-1).x, 0);
  app.key("keydown", "w");
  app.elements.settings.listeners.click();
  assert.equal(app.sockets[0].sent.at(-1).y, 0);
  const sent = app.sockets[0].sent.length;
  app.key("keydown", "d");
  app.key("keydown", " ");
  assert.equal(app.sockets[0].sent.length, sent);
  app.elements["settings-dialog"].close();
  app.elements["control-mode"].value = "drag";
  app.elements["control-mode"].listeners.change();
  app.key("keydown", "w");
  assert.equal(app.sockets[0].sent.at(-1).y, 0);
});

test("clock displays authoritative remaining time, pause and round reset", () => {
  const app = client();
  app.open();
  app.snapshot({ time_remaining: 4.2 });
  assert.equal(app.elements["time-progress"].value, 4.2);
  assert.equal(app.elements["round-clock"].dataset.urgent, "true");
  app.snapshot({ paused: true, time_remaining: 4.2 });
  assert.match(app.elements["time-left"].textContent, /paused/);
  assert.equal(app.elements.pause.textContent, "Resume");
  app.snapshot({ phase: "countdown" });
  assert.equal(app.elements["time-progress"].value, 22.222);
  assert.equal(app.elements["round-clock"].dataset.urgent, "false");
});

test("music and effects can be disabled independently and preferences persist", () => {
  const app = client();
  app.open();
  app.snapshot();
  app.elements["music-enabled"].checked = false;
  app.elements["music-enabled"].listeners.change();
  assert.equal(app.sound.music, false);
  assert.equal(app.sound.sfx, true);
  app.elements["sfx-enabled"].checked = false;
  app.elements["sfx-enabled"].listeners.change();
  assert.equal(app.sound.sfx, false);
  const restored = client(JSON.parse(app.stored.get("flatworld-pusht-settings")));
  assert.equal(restored.elements["music-enabled"].checked, false);
  assert.equal(restored.elements["sfx-enabled"].checked, false);
  app.hide();
  assert.equal(app.sound.visible, false);
});

test("push sound requires contact with the moving T, rather than free movement", () => {
  const app = client();
  app.open();
  const left = { states: [[-0.11, 0, 0, 0, 0], [0, 0, 0, 0.1, 0]], geom: [[0.05, 0.05], [0.08, 0.28]] };
  app.snapshot({ left });
  assert.equal(app.sound.pushes.length, 1);
  left.states[0][0] = -1;
  app.snapshot({ left });
  assert.equal(app.sound.pushes.length, 1);
  app.snapshot({ left, paused: true });
  assert.equal(app.sound.pushes.length, 1);
});

test("audio starts on interaction and music/effects/hidden-tab gates are independent", () => {
  const oscillators = [];
  const intervals = new Set();
  const param = () => ({ value: 0, setValueAtTime() {}, linearRampToValueAtTime() {},
    exponentialRampToValueAtTime() {}, setTargetAtTime(value) { this.value = value; } });
  class AudioContext {
    currentTime = 0;
    state = "running";
    destination = {};
    createGain() { return { gain: param(), connect() {}, disconnect() {} }; }
    createOscillator() {
      const osc = { frequency: param(), connect() {}, disconnect() {}, start() {}, stop() {} };
      oscillators.push(osc);
      return osc;
    }
  }
  const sandbox = { window: { AudioContext }, setInterval(callback) { intervals.add(callback); return callback; },
    clearInterval(callback) { intervals.delete(callback); } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../../docs/pusht/audio.js"), "utf8"), sandbox);
  const sound = new sandbox.window.PushTAudio();
  sound.setPlaying(true);
  assert.equal(sound.ctx, null);
  sound.unlock();
  assert.ok(oscillators.length > 0);
  assert.equal(intervals.size, 1);
  sound.configure(false, true);
  assert.equal(intervals.size, 0);
  const beforePush = oscillators.length;
  sound.push(0.5);
  assert.equal(oscillators.length, beforePush + 1);
  sound.configure(true, false);
  assert.equal(intervals.size, 1);
  const muted = oscillators.length;
  sound.ctx.currentTime = 1;
  sound.push(0.5);
  assert.equal(oscillators.length, muted);
  sound.setPlaying(true, false);
  assert.equal(intervals.size, 0);
  assert.equal(sound.music.gain.value, 0);
  assert.equal(sound.effects.gain.value, 0);
});

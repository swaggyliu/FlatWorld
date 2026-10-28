const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function client() {
  const sockets = [];
  const callbacks = {};
  let now = 0;
  let nextFrame;
  const ctx = new Proxy({}, { get: () => () => {} });
  const canvas = {
    width: 1100, height: 560, clientWidth: 1100,
    getContext: () => ctx, addEventListener() {},
  };
  const status = { textContent: "" };
  const elements = {
    c: canvas, status, score: {},
    rematch: { addEventListener: (type, callback) => { callbacks.rematch = callback; } },
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
    document: { getElementById: (id) => elements[id] || null, addEventListener() {} },
    window: { PUSHT_WS: "wss://game.example/ws", innerWidth: 1100,
      devicePixelRatio: 1, matchMedia: () => ({ matches: false }), addEventListener() {} },
    location: { search: "" }, URLSearchParams, WebSocket,
    performance: { now: () => now },
    requestAnimationFrame: (callback) => { nextFrame = callback; },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../../docs/pusht/app.js"), "utf8"), sandbox);
  return { sockets, callbacks, status,
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

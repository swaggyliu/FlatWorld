// Backend WebSocket. Override with ?ws=… or window.PUSHT_WS.
(function () {
  const params = new URLSearchParams(location.search);
  if (params.get("ws")) {
    window.PUSHT_WS = params.get("ws");
    return;
  }
  if (window.PUSHT_WS) return;

  // Same-origin when UI is served by game.web_server / local / CF tunnel.
  if (
    location.port === "8765" ||
    location.hostname === "127.0.0.1" ||
    location.hostname === "localhost" ||
    location.hostname.endsWith(".trycloudflare.com") ||
    location.hostname.endsWith(".sslip.io")
  ) {
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    window.PUSHT_WS = proto + location.host + "/ws";
    return;
  }

  // GitHub Pages is https → browser requires wss:// (plain ws:// is blocked).
  // Prefer Aliyun TLS hostname when Caddy binds :443; else Cloudflare tunnel
  // that terminates TLS in front of the same Aliyun :8765 process.
  window.PUSHT_WS = "wss://8.138.152.185.sslip.io/ws";
  window.PUSHT_WS_FALLBACK =
    "wss://immediate-pale-indicate-displayed.trycloudflare.com/ws";
})();

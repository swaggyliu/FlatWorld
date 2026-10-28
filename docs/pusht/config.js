// Endpoints are not shown in the UI. Optional override: ?ws=…
(function () {
  function dec(s) {
    try {
      return atob(s);
    } catch (_) {
      return "";
    }
  }
  const params = new URLSearchParams(location.search);
  if (params.get("ws")) {
    window.PUSHT_WS = params.get("ws");
    return;
  }
  if (window.PUSHT_WS) return;

  const host = location.hostname;
  if (
    location.port === "8765" ||
    host === "127.0.0.1" ||
    host === "localhost" ||
    host.endsWith(".trycloudflare.com") ||
    host.endsWith(".sslip.io")
  ) {
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    window.PUSHT_WS = proto + location.host + "/ws";
    return;
  }

  // Primary: flexphysics.com (ICP/TCP 备案) via gateway→node2 bridge.
  // Fallback: direct node2 sslip.io TLS.
  window.PUSHT_WS = dec(
    "d3NzOi8vd3d3LmZsZXhwaHlzaWNzLmNvbS9jYWwvdGFzay1kYXRhL3dzL19mdw=="
  );
  window.PUSHT_WS_FALLBACK = dec(
    "d3NzOi8vOC4xMzguMTUyLjE4NS5zc2xpcC5pby93cw=="
  );
})();

// Override with ?ws=wss://host/ws or window.PUSHT_WS before this script.
(function () {
  const params = new URLSearchParams(location.search);
  if (params.get("ws")) {
    window.PUSHT_WS = params.get("ws");
    return;
  }
  if (window.PUSHT_WS) return;

  // Same-origin when page is served by game.web_server / Cloudflare tunnel.
  if (
    location.port === "8765" ||
    location.hostname === "127.0.0.1" ||
    location.hostname === "localhost" ||
    location.hostname.endsWith(".trycloudflare.com")
  ) {
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    window.PUSHT_WS = proto + location.host + "/ws";
    return;
  }

  // GitHub Pages (https) → Cloudflare quick tunnel (WSS). Refresh via deploy script
  // if the trycloudflare hostname rotates. Direct IP needs Aliyun SG TCP 8765 + TLS.
  window.PUSHT_WS = "wss://immediate-pale-indicate-displayed.trycloudflare.com/ws";
})();

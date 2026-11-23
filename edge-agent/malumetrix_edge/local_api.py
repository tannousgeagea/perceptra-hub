"""
A small HTTP API on the device (127.0.0.1:7070 by default), so an inference
service in any language can use the agent without internet access or
credentials of its own.

    GET  /v1/status                 device, platform connection, models
    GET  /v1/models                 {model_id: current model} for every model
    GET  /v1/models/<model_id>      model_path, manifest_path, manifest, version
    POST /v1/health                 {"fps": 24.8, "latency_ms": 31, "errors": 0}
    POST /v1/samples                not yet (phase 2): 501

It listens on localhost only unless you set local_api to another address;
anything that can reach it can read model paths and report health.
"""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

log = logging.getLogger("malumetrix_edge")
MAX_BODY = 16_384


def serve(address: str, agent) -> Optional[ThreadingHTTPServer]:
    """Start in a daemon thread. `agent` needs .status(), .engine and .health."""
    if not address or address.lower() == "off":
        return None
    host, _, port = address.rpartition(":")

    class Handler(BaseHTTPRequestHandler):
        server_version = "malumetrix-edge"

        def log_message(self, fmt, *args):  # quiet: the service polls often
            log.debug("local api: " + fmt, *args)

        def _send(self, code: int, body) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/v1/status":
                return self._send(200, agent.status())
            if path == "/v1/models":
                models = {mid: agent.engine.current(mid) for mid in agent.store.state["models"]}
                return self._send(200, {k: v for k, v in models.items() if v})
            if path.startswith("/v1/models/"):
                info = agent.engine.current(path.split("/", 3)[3])
                return self._send(200, info) if info else self._send(404, {"detail": "No model with that id runs here yet"})
            self._send(404, {"detail": "Not found"})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"detail": f"Body over {MAX_BODY} bytes"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, {"detail": "Body must be JSON"})
            if path == "/v1/health":
                if not isinstance(body, dict):
                    return self._send(400, {"detail": "Send an object of metrics"})
                agent.health.report(**{str(k): v for k, v in body.items()})
                return self._send(200, {"ok": True})
            if path == "/v1/samples":
                return self._send(501, {"detail": "Image and prediction samples come in a later agent version"})
            self._send(404, {"detail": "Not found"})

    server = ThreadingHTTPServer((host or "127.0.0.1", int(port)), Handler)
    threading.Thread(target=server.serve_forever, name="malumetrix-local-api", daemon=True).start()
    log.info("local API on http://%s", address)
    return server

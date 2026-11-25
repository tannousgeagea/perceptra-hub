"""
A small HTTP API on the device (127.0.0.1:7070 by default), so an inference
service in any language can use the agent without internet access or
credentials of its own.

    GET  /v1/status                 device, platform connection, models
    GET  /v1/models                 {model_id: current model} for every model
    GET  /v1/models/<model_id>      model_path, manifest_path, manifest, version
    POST /v1/health                 {"fps": 24.8, "latency_ms": 31, "errors": 0}
    POST /v1/predictions            {"model_id", "camera"?, "predictions": [{"class", "confidence"}, …]}
                                    or "batch": [[…], […]], one list per inference (hourly summaries)
    POST /v1/samples/check          {"model_id", "predictions"} → {"keep": reason or null}; cheap, per frame
    POST /v1/samples                {"model_id", "image_base64", "predictions", "camera"?, "format"?}
                                    → {"kept": reason or null}; uploaded in the background

It listens on localhost only unless you set local_api to another address;
anything that can reach it can read model paths and report health.
"""
import base64
import binascii
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

log = logging.getLogger("malumetrix_edge")
MAX_BODY = 262_144   # a batch of predictions can be sizeable
MAX_SAMPLE_BODY = 24 * 1024 * 1024  # a base64 image


def serve(address: str, agent) -> Optional[ThreadingHTTPServer]:
    """Start in a daemon thread. `agent` needs .status(), .record(), .engine, .store, .health and .sampler."""
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
            limit = MAX_SAMPLE_BODY if path == "/v1/samples" else MAX_BODY
            if length > limit:
                return self._send(413, {"detail": f"Body over {limit} bytes"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, {"detail": "Body must be JSON"})
            if path == "/v1/health":
                if not isinstance(body, dict):
                    return self._send(400, {"detail": "Send an object of metrics"})
                agent.health.report(**{str(k): v for k, v in body.items()})
                return self._send(200, {"ok": True})
            if path == "/v1/predictions":
                # {"model_id", "camera"?, "predictions": [...]} for one inference, or
                # {"model_id", "camera"?, "batch": [[...], [...]]} for several (one list per inference).
                if not isinstance(body, dict) or not body.get("model_id"):
                    return self._send(400, {"detail": "Send {\"model_id\": …, \"predictions\": [...]} or \"batch\": [[...], …]"})
                runs = body.get("batch") if isinstance(body.get("batch"), list) else [body.get("predictions") or []]
                for preds in runs:
                    agent.record(str(body["model_id"]), preds if isinstance(preds, list) else [], camera=body.get("camera"))
                return self._send(200, {"recorded": len(runs)})
            if path == "/v1/samples/check":
                # Cheap, per frame: is it worth encoding and sending?
                if not isinstance(body, dict) or not body.get("model_id"):
                    return self._send(400, {"detail": "Send {\"model_id\": …, \"predictions\": [...]}"})
                return self._send(200, {"keep": agent.sampler.should_sample(str(body["model_id"]), body.get("predictions") or [])})
            if path == "/v1/samples":
                if not isinstance(body, dict) or not body.get("model_id") or not body.get("image_base64"):
                    return self._send(400, {"detail": "Send model_id, image_base64 (JPEG/PNG/WebP), predictions, and optionally camera, format, width, height"})
                try:
                    image = base64.b64decode(body["image_base64"], validate=True)
                    kept = agent.sampler.offer_sample(str(body["model_id"]), image, body.get("predictions") or [],
                                                      camera=body.get("camera"), fmt=body.get("format") or "jpg",
                                                      width=body.get("width"), height=body.get("height"))
                except (ValueError, binascii.Error) as e:
                    return self._send(400, {"detail": f"Bad sample: {e}"})
                return self._send(200, {"kept": kept})
            self._send(404, {"detail": "Not found"})

    server = ThreadingHTTPServer((host or "127.0.0.1", int(port)), Handler)
    threading.Thread(target=server.serve_forever, name="malumetrix-local-api", daemon=True).start()
    log.info("local API on http://%s", address)
    return server

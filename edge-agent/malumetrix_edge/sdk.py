"""
The agent as a library.

Sidecar (`malumetrix-edge run`) and SDK share the same engine; they differ
only in how a new model reaches the inference code: hooks (commands, URLs)
for the sidecar, a Python callback for the SDK.

    from malumetrix_edge import EdgeClient

    def load(update):                      # runs in the agent's thread
        detector.swap(update.model_path, update.manifest["classes"])
        return detector.self_test()        # False (or an exception) → rolled back

    edge = EdgeClient(url="https://malumetrix.example.com", code="K7QM-4XPA-9TWE",
                      on_model_update=load, accepts=["onnx"])
    edge.start()
    ...
    edge.report_health(fps=24.8, latency_ms=31, errors=0)
"""
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import __version__, hooks
from .config import Config, load
from .engine import Engine
from .health import Health
from .local_api import serve
from .protocol import Client
from .runner import Runner
from .store import Store

log = logging.getLogger("malumetrix_edge")


class NotEnrolled(Exception):
    pass


@dataclass
class ModelUpdate:
    model_id: str
    version_id: str
    model_path: str
    manifest_path: str
    manifest: dict


def enroll(cfg: Config, code: str) -> dict:
    """Trade a one-time code for this device's key and keep it in data_dir."""
    store = Store(cfg.data_path)
    answer = Client(cfg.url, timeout=cfg.request_timeout_s, ca_bundle=cfg.ca_bundle).enroll(
        code, accepts=cfg.accepts, hardware=Health.hardware())
    store.save_credentials(answer["device_id"], answer["device_key"], answer.get("name", ""))
    log.info("enrolled as %s (%s)", answer.get("name"), answer["device_id"])
    return answer


class Agent:
    def __init__(self, cfg: Config, hand_over=None, check_health=None):
        self.cfg = cfg
        self.store = Store(cfg.data_path)
        creds = self.store.credentials()
        if not creds:
            raise NotEnrolled("This device isn't enrolled yet: run `malumetrix-edge enroll --code <code from Fleet>`.")
        self.device = creds
        self.health = Health(cfg.data_path)
        self.client = Client(cfg.url, creds["device_key"], timeout=cfg.request_timeout_s, ca_bundle=cfg.ca_bundle)
        self.engine = Engine(cfg, self.store, hand_over or hooks.hand_over, check_health or hooks.healthy)
        self.runner = Runner(cfg, self.store, self.client, self.engine, self.health)
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def status(self) -> dict:
        r = self.runner
        return {"agent_version": __version__, "device_id": self.device.get("device_id"), "name": self.device.get("name"),
                "platform": self.cfg.url, "connected": r.failures == 0 and r.last_ok is not None,
                "last_heartbeat": r.last_ok, "last_error": r.last_error, "poll_seconds": r.poll_seconds,
                "models": self.engine.reports()}

    def start(self) -> "Agent":
        self._server = serve(self.cfg.local_api, self)
        self._thread = threading.Thread(target=self.runner.run_forever, name="malumetrix-heartbeat", daemon=True)
        self._thread.start()
        return self

    def run_forever(self) -> None:
        self._server = serve(self.cfg.local_api, self)
        self.runner.run_forever()

    def stop(self) -> None:
        self.runner.stop()
        if self._server:
            self._server.shutdown()


class EdgeClient:
    def __init__(self, url: Optional[str] = None, *, code: Optional[str] = None, data_dir: Optional[str] = None,
                 on_model_update: Optional[Callable[[ModelUpdate], Optional[bool]]] = None,
                 accepts: Optional[List[str]] = None, config_path: Optional[str] = None, local_api: str = "off",
                 health_url: Optional[str] = None, health_timeout_s: Optional[int] = None):
        self.cfg = load(config_path, url=url, data_dir=data_dir, accepts=accepts, local_api=local_api,
                        health_url=health_url, health_timeout_s=health_timeout_s)
        self._callback = on_model_update
        if not Store(self.cfg.data_path).credentials():
            if not code:
                raise NotEnrolled("Pass code= (from Fleet → Add device) the first time.")
            enroll(self.cfg, code)
        self.agent = Agent(self.cfg, hand_over=self._hand_over if on_model_update else None)

    def _hand_over(self, cfg: Config, info: dict):
        manifest = self.agent.store.state["models"].get(info["model_id"], {}).get("manifest") or {}
        try:
            with open(info["manifest_path"]) as f:
                manifest = json.load(f)
        except Exception:
            pass
        try:
            ok = self._callback(ModelUpdate(info["model_id"], info["version_id"], info["model_path"], info["manifest_path"], manifest))
        except Exception as e:
            log.exception("on_model_update raised")
            return False, f"on_model_update raised {type(e).__name__}: {e}"
        return (False, "on_model_update returned False") if ok is False else (True, "")

    def start(self) -> "EdgeClient":
        self.agent.start()
        return self

    def stop(self) -> None:
        self.agent.stop()

    def report_health(self, **metrics) -> None:
        """fps=, latency_ms=, errors=, frames=… sent with the next heartbeat."""
        self.agent.health.report(**metrics)

    def current(self, model_id: str) -> Optional[ModelUpdate]:
        info = self.agent.engine.current(model_id)
        if not info:
            return None
        return ModelUpdate(info["model_id"], info["version_id"], info["model_path"], info["manifest_path"], info.get("manifest") or {})

    def models(self) -> Dict[str, ModelUpdate]:
        return {mid: m for mid in self.agent.store.state["models"] if (m := self.current(mid))}

    def wait_for_model(self, model_id: str, timeout: float = 600) -> Optional[ModelUpdate]:
        """Block until the model runs here (first start), or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            m = self.current(model_id)
            if m and self.agent.store.state["models"][model_id].get("status") == "running":
                return m
            time.sleep(1)
        return None

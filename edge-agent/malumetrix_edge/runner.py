"""
The heartbeat loop. One call every poll_seconds (the platform decides, with
±10% jitter): report what runs and how it's doing, get back what should run.

Offline: models keep running, health is kept for later, retries back off
exponentially (with jitter, up to 5 minutes). Revoked key: models keep
running, the agent says so in its log and checks again every 5 minutes.
"""
import logging
import random
import threading
import time
from typing import Optional

from .config import Config
from .engine import Engine
from .health import Health
from .protocol import ApiError, Client, NetworkError, backoff_seconds
from .store import Store

log = logging.getLogger("malumetrix_edge")

HARDWARE_EVERY_S = 3600


class Runner:
    def __init__(self, cfg: Config, store: Store, client: Client, engine: Engine, health: Health):
        self.cfg, self.store, self.client, self.engine, self.health = cfg, store, client, engine, health
        self.poll_seconds = 30
        self.failures = 0
        self.last_ok: Optional[float] = None
        self.last_error: Optional[str] = None
        self._hardware_at = 0.0
        self._stop = threading.Event()

    def beat(self) -> Optional[dict]:
        """One heartbeat. Returns the answer, or None if the platform wasn't reached."""
        snapshot = self.health.snapshot()
        backlog = self.health.batch()
        payload = {"accepts": self.cfg.accepts, "health": snapshot, "health_backlog": backlog,
                   "models": self.engine.reports(), "state_token": self.store.state.get("state_token"),
                   "refresh": self.engine.needs_refresh}
        if time.monotonic() - self._hardware_at > HARDWARE_EVERY_S:
            payload["hardware"] = self.health.hardware()
        try:
            answer = self.client.heartbeat(payload)
        except NetworkError as e:
            self.health.keep(snapshot)
            self.failures += 1
            self.last_error = f"platform unreachable: {e}"
            log.warning("%s (attempt %s)", self.last_error, self.failures)
            return None
        except ApiError as e:
            self.health.keep(snapshot)
            self.failures += 1
            self.last_error = f"platform refused: {e.detail}"
            log.error(self.last_error)
            if e.status == 401:
                log.error("The device key was revoked or is unknown. Models keep running; "
                          "enroll again with a new code from Fleet: malumetrix-edge enroll --code …")
            return {"_error": e.status}
        self.failures, self.last_ok, self.last_error = 0, time.time(), None
        self.health.sent(backlog)
        if "hardware" in payload:
            self._hardware_at = time.monotonic()
        self.poll_seconds = int(answer.get("poll_seconds") or self.poll_seconds)
        if not answer.get("unchanged"):
            self.engine.needs_refresh = False
            self.store.state["state_token"] = answer.get("state_token")
            self.store.save()
            self.engine.set_desired(answer.get("models") or [])
        return answer

    def next_wait(self, answer: Optional[dict]) -> float:
        if answer is not None and answer.get("_error") == 401:
            return 300.0
        if self.failures:
            return backoff_seconds(self.failures, base=self.poll_seconds)
        return self.poll_seconds * random.uniform(0.9, 1.1)

    def run_forever(self) -> None:
        self.engine.start()
        while not self._stop.is_set():
            answer = self.beat()
            self._stop.wait(self.next_wait(answer))

    def stop(self) -> None:
        self._stop.set()
        self.engine.stop()

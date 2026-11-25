"""
The reconciler: makes what runs match what the platform wants.

For each model whose desired version differs from the running one:
  download (resumable, checksum-verified) → switch `current` in one step →
  hand over to the inference service → wait until it is healthy.
If the hand-over or the health check fails, it switches back to the version
that ran before and reports `rolled_back` with the failed version, so the
rollout pauses and nobody has to log in to the box. A version that failed is
not retried for the same rollout (no crash loops); a new rollout may retry it.

It runs in its own thread, so heartbeats keep going during a long download.
"""
import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

from . import downloader
from .config import Config
from .store import Store, write_atomic

log = logging.getLogger("malumetrix_edge")

HandOver = Callable[[Config, dict], Tuple[bool, str]]


class Engine:
    def __init__(self, cfg: Config, store: Store, hand_over: HandOver, check_health: HandOver):
        self.cfg, self.store = cfg, store
        self.hand_over, self.check_health = hand_over, check_health
        self._desired: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.needs_refresh = False
        self._thread: Optional[threading.Thread] = None
        self.listeners: List[Callable[[str, dict], None]] = []  # (model_id, current info) after a switch

    # ── input from the heartbeat ──
    def set_desired(self, models: List[dict]) -> None:
        with self._lock:
            self._desired = {m["model_id"]: m for m in models}
        self._wake.set()

    def sampling(self, model_id: str) -> Optional[dict]:
        """The platform's image sampling policy for a model (None until the first full heartbeat answer)."""
        with self._lock:
            entry = self._desired.get(model_id)
        return (entry or {}).get("sampling")

    # ── output for the heartbeat ──
    def reports(self) -> List[dict]:
        out = []
        for model_id, st in list(self.store.state["models"].items()):
            out.append({"model_id": model_id, "version_id": st.get("current"), "status": st.get("status") or "pending",
                        "failed_version_id": st.get("failed_version"), "error": (st.get("error") or None) and st["error"][:500]})
        return out

    def current(self, model_id: str) -> Optional[dict]:
        st = self.store.state["models"].get(model_id)
        if not st or not st.get("current") or not st.get("format"):
            return None
        d = self.store.version_dir(model_id, st["current"])
        return {"model_id": model_id, "version_id": st["current"], "model_path": str(d / f"model.{st['format']}"),
                "manifest_path": str(d / "manifest.json"), "manifest": st.get("manifest"), "status": st.get("status")}

    # ── the worker ──
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="malumetrix-reconciler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=60)
            self._wake.clear()
            self.reconcile_once()

    def reconcile_once(self) -> None:
        with self._lock:
            desired = list(self._desired.values())
        for entry in desired:
            if self._stop.is_set():
                return
            try:
                self._apply(entry)
            except Exception as e:  # one model's trouble never stops the others
                log.exception("reconcile %s failed", entry.get("model_id"))
                self.store.update_model(entry["model_id"], error=f"agent error: {e}"[:500])

    def _apply(self, entry: dict) -> None:
        mid, vid = entry["model_id"], entry["version_id"]
        st = self.store.model(mid)
        if st["current"] == vid and st["status"] in ("running", "rolled_back"):
            return  # rolled_back: already running this version, after refusing a newer one
        failed_for = st.get("failed_rollouts", {}).get(vid)
        if failed_for is not None and failed_for == entry.get("rollout_id"):
            return  # failed for this rollout already; wait for a new one
        art = entry.get("artifact")
        if not entry.get("ready") or not art:
            return  # the platform is still preparing the file

        cfg = self.cfg.for_model(mid)
        fmt = art["format"]
        was = st["status"]
        log.info("%s: updating %s → %s (%s)", entry.get("model_name", mid), st["current"], vid, fmt)
        self.store.update_model(mid, status="downloading", error=None)
        try:
            if not self.store.has_version(mid, vid, fmt):
                downloader.download(art["url"], self.store.model_file(mid, vid, fmt), art["sha256"], art.get("size_bytes"),
                                    timeout=max(60, self.cfg.request_timeout_s), ca_bundle=self.cfg.ca_bundle)
                write_atomic(self.store.version_dir(mid, vid) / "manifest.json", entry.get("manifest") or {})
        except downloader.LinkExpired:
            self.needs_refresh = True
            self.store.update_model(mid, status=was if was != "downloading" else "pending")
            return
        except Exception as e:  # network or checksum: transient, retried on the next heartbeat
            log.warning("%s: download failed: %s", mid, e)
            self.store.update_model(mid, status=was if was != "downloading" else "pending", error=f"download: {e}"[:500])
            return

        info = self.store.point_current(mid, vid, fmt)
        ok, err = self.hand_over(cfg, info)
        if ok:
            ok, err = self.check_health(cfg, info)
        if ok:
            previous = [v for v in [st["current"], *st["previous"]] if v and v != vid][: self.cfg.keep_versions - 1]
            self.store.update_model(mid, current=vid, previous=previous, status="running", error=None,
                                    failed_version=None, format=fmt, manifest=entry.get("manifest"))
            self.store.prune(mid, keep=[vid, *previous])
            log.info("%s: running %s", mid, vid)
            self._notify(mid)
            return

        # Failed: go back to what ran before, if anything did.
        log.error("%s: %s failed: %s", mid, vid, err)
        failed_rollouts = {**st.get("failed_rollouts", {}), vid: entry.get("rollout_id")}
        old = st["current"]
        if old and self.store.has_version(mid, old, st["format"]):
            back = self.store.point_current(mid, old, st["format"])
            ok_back, err_back = self.hand_over(cfg, back)
            self.store.update_model(mid, status="rolled_back" if ok_back else "failed", failed_version=vid,
                                    failed_rollouts=failed_rollouts,
                                    error=(err if ok_back else f"{err}; going back also failed: {err_back}")[:500])
            self._notify(mid)
        else:
            self.store.clear_current(mid)  # nothing ran before: don't leave the broken version as current
            self.store.update_model(mid, status="failed", failed_version=vid, failed_rollouts=failed_rollouts, error=err[:500])

    def _notify(self, model_id: str) -> None:
        info = self.current(model_id)
        for fn in list(self.listeners):
            try:
                fn(model_id, info or {})
            except Exception:
                log.exception("listener failed")

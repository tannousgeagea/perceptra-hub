"""
Health: what the machine says (CPU, memory, disk, temperature) plus what your
inference service reports (fps, latency, errors) through the local API or the
SDK. Reports made while the platform is unreachable are kept (one per minute,
up to a day) and sent in batches once it is back.
"""
import os
import platform
import shutil
import socket
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .store import read_json, write_atomic

APP_METRICS_TTL_S = 120     # app numbers older than this are stale, not reported
BACKLOG_MAX = 24 * 60       # a day of minutes
BACKLOG_BATCH = 120         # per heartbeat (the server's limit)


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text()
    except OSError:
        return None


class Health:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._app: Dict[str, object] = {}
        self._app_at = 0.0
        self._cpu_prev = None
        self._backlog_path = self.data_dir / "health_backlog.json"
        self._backlog: "OrderedDict[str, dict]" = OrderedDict(
            (p["at"], p) for p in read_json(self._backlog_path, []) if isinstance(p, dict) and "at" in p)

    # ── from the inference service ──
    def report(self, **metrics) -> None:
        """fps=, latency_ms=, errors=, frames=… Numbers or short strings."""
        with self._lock:
            self._app.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None})
            self._app_at = time.monotonic()

    # ── from the machine ──
    def _cpu_percent(self) -> Optional[float]:
        raw = _read("/proc/stat")
        if not raw:
            return None
        vals = [int(x) for x in raw.splitlines()[0].split()[1:]]
        idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
        prev, self._cpu_prev = self._cpu_prev, (idle, total)
        if not prev or total == prev[1]:
            return None
        return round(100.0 * (1 - (idle - prev[0]) / (total - prev[1])), 1)

    def _memory_percent(self) -> Optional[float]:
        raw = _read("/proc/meminfo")
        if not raw:
            return None
        info = {line.split(":")[0]: int(line.split()[1]) for line in raw.splitlines() if line.split()[1:]}
        if not info.get("MemTotal"):
            return None
        return round(100.0 * (1 - info.get("MemAvailable", info.get("MemFree", 0)) / info["MemTotal"]), 1)

    def _temperature_c(self) -> Optional[float]:
        raw = _read("/sys/class/thermal/thermal_zone0/temp")
        try:
            return round(int(raw) / 1000, 1) if raw else None
        except ValueError:
            return None

    def snapshot(self) -> dict:
        m: Dict[str, object] = {}
        cpu, mem, temp = self._cpu_percent(), self._memory_percent(), self._temperature_c()
        if cpu is not None:
            m["cpu_percent"] = cpu
        if mem is not None:
            m["memory_percent"] = mem
        if temp is not None:
            m["temperature_c"] = temp
        try:
            m["disk_free_gb"] = round(shutil.disk_usage(self.data_dir).free / 1e9, 2)
        except OSError:
            pass
        with self._lock:
            if self._app and time.monotonic() - self._app_at <= APP_METRICS_TTL_S:
                m.update(self._app)
        return m

    @staticmethod
    def hardware() -> dict:
        info = {"hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
                "machine": platform.machine(), "python": platform.python_version(), "cpus": os.cpu_count()}
        gpu = _read("/proc/device-tree/model")  # Jetson and other SBCs
        if gpu:
            info["board"] = gpu.strip("\x00 \n")
        return info

    # ── offline backlog ──
    def keep(self, metrics: dict) -> None:
        """Remember this minute's report until it can be sent."""
        at = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
        with self._lock:
            self._backlog[at] = {"at": at, "metrics": metrics}
            while len(self._backlog) > BACKLOG_MAX:
                self._backlog.popitem(last=False)
            write_atomic(self._backlog_path, list(self._backlog.values()))

    def batch(self) -> List[dict]:
        with self._lock:
            return list(self._backlog.values())[:BACKLOG_BATCH]

    def sent(self, batch: List[dict]) -> None:
        if not batch:
            return
        with self._lock:
            for p in batch:
                self._backlog.pop(p["at"], None)
            write_atomic(self._backlog_path, list(self._backlog.values()))

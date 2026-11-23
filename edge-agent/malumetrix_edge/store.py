"""
Everything the agent keeps on disk, under data_dir:

    credentials.json                 device id + key (mode 0600)
    state.json                       per model: current, previous, failed versions
    health_backlog.json              health reports not sent yet (offline)
    models/<model_id>/<version_id>/  model.<format> + manifest.json
    models/<model_id>/current        symlink to the running version's folder
    models/<model_id>/current.json   the same as a file: {version_id, model_path, manifest_path}

Writes go to a temp file, are flushed and renamed, so a power cut never
leaves a half-written state or a half-switched model.
"""
import json
import os
import shutil
import threading
from pathlib import Path
from typing import List, Optional


def write_atomic(path: Path, data, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return default


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.state = read_json(self.root / "state.json", {"models": {}, "state_token": None})

    # ── credentials ──
    def credentials(self) -> Optional[dict]:
        return read_json(self.root / "credentials.json", None)

    def save_credentials(self, device_id: str, device_key: str, name: str = "") -> None:
        write_atomic(self.root / "credentials.json", {"device_id": device_id, "device_key": device_key, "name": name}, mode=0o600)

    # ── state ──
    def save(self) -> None:
        with self._lock:
            write_atomic(self.root / "state.json", self.state)

    def model(self, model_id: str) -> dict:
        with self._lock:
            return self.state["models"].setdefault(model_id, {
                "current": None, "previous": [], "failed": [], "status": "pending", "error": None,
                "failed_version": None, "format": None, "manifest": None})

    def update_model(self, model_id: str, **changes) -> None:
        with self._lock:
            self.model(model_id).update(changes)
            self.save()

    # ── files ──
    def version_dir(self, model_id: str, version_id: str) -> Path:
        return self.root / "models" / model_id / version_id

    def model_file(self, model_id: str, version_id: str, fmt: str) -> Path:
        return self.version_dir(model_id, version_id) / f"model.{fmt}"

    def has_version(self, model_id: str, version_id: str, fmt: str) -> bool:
        d = self.version_dir(model_id, version_id)
        return (d / f"model.{fmt}").exists() and (d / "manifest.json").exists()

    def point_current(self, model_id: str, version_id: str, fmt: str) -> dict:
        """Switch `current` to the version in one step (symlink + pointer file)."""
        base = self.root / "models" / model_id
        target = self.version_dir(model_id, version_id)
        info = {"model_id": model_id, "version_id": version_id,
                "model_path": str(target / f"model.{fmt}"), "manifest_path": str(target / "manifest.json")}
        link, tmp = base / "current", base / ".current.tmp"
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            os.symlink(target.name, tmp)  # relative: survives moving data_dir
            os.replace(tmp, link)
        except OSError:
            pass  # no symlinks (some Windows/FAT setups): the pointer file is enough
        write_atomic(base / "current.json", info)
        return info

    def clear_current(self, model_id: str) -> None:
        base = self.root / "models" / model_id
        for p in (base / "current", base / "current.json"):
            if p.is_symlink() or p.exists():
                p.unlink()

    def prune(self, model_id: str, keep: List[str]) -> None:
        """Delete version folders not in `keep` (current + previous)."""
        base = self.root / "models" / model_id
        if not base.exists():
            return
        for child in base.iterdir():
            if child.is_dir() and not child.is_symlink() and child.name not in keep:
                shutil.rmtree(child, ignore_errors=True)

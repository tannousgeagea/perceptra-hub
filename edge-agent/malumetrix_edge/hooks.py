"""
How the sidecar hands a new model to *your* inference service, and how it
knows the service is happy with it. All optional; use what fits:

  reload_command   any shell command, e.g. `docker restart detector`,
                   `systemctl reload detector`, `kill -HUP $(cat /run/det.pid)`
  notify_url       POST {model_id, version_id, model_path, manifest_path}
                   to an endpoint of your service
  health_url       GET; any 2xx within health_timeout_s means healthy
  health_command   exit code 0 within health_timeout_s means healthy

Commands get MALUMETRIX_MODEL_ID, MALUMETRIX_VERSION_ID, MALUMETRIX_MODEL_PATH
and MALUMETRIX_MANIFEST_PATH in their environment. With no health check
configured, a model counts as healthy once its reload hook succeeded.
"""
import json
import os
import subprocess
import time
import urllib.request
from typing import Tuple

from .config import Config


def _env(info: dict) -> dict:
    return {**os.environ, "MALUMETRIX_MODEL_ID": info["model_id"], "MALUMETRIX_VERSION_ID": info["version_id"],
            "MALUMETRIX_MODEL_PATH": info["model_path"], "MALUMETRIX_MANIFEST_PATH": info["manifest_path"]}


def _run(cmd: str, info: dict, timeout: int) -> Tuple[bool, str]:
    try:
        p = subprocess.run(cmd, shell=True, env=_env(info), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"`{cmd}` took longer than {timeout}s"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or [""]
        return False, f"`{cmd}` exited with {p.returncode}: {tail[0][:200]}"
    return True, ""


def _get_ok(url: str, timeout: int) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def hand_over(cfg: Config, info: dict) -> Tuple[bool, str]:
    """Tell the inference service about the model now at info['model_path']."""
    if cfg.reload_command:
        ok, err = _run(cfg.reload_command, info, timeout=max(30, cfg.health_timeout_s))
        if not ok:
            return False, err
    if cfg.notify_url:
        req = urllib.request.Request(cfg.notify_url, data=json.dumps(info).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if not 200 <= resp.status < 300:
                    return False, f"{cfg.notify_url} answered {resp.status}"
        except Exception as e:
            return False, f"{cfg.notify_url}: {e}"
    return True, ""


def healthy(cfg: Config, info: dict) -> Tuple[bool, str]:
    """Wait until the service reports healthy, up to health_timeout_s."""
    if not cfg.health_url and not cfg.health_command:
        return True, ""
    deadline = time.monotonic() + cfg.health_timeout_s
    last = ""
    while time.monotonic() < deadline:
        if cfg.health_url and not _get_ok(cfg.health_url, timeout=10):
            last = f"{cfg.health_url} not healthy"
        elif cfg.health_command:
            ok, last = _run(cfg.health_command, info, timeout=30)
            if ok:
                return True, ""
        else:
            return True, ""
        time.sleep(min(5, max(1, cfg.health_timeout_s / 10)))
    return False, f"not healthy after {cfg.health_timeout_s}s ({last})"

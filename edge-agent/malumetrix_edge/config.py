"""
Settings: a JSON file, overridden by environment variables (handy in Docker).

    {
      "url": "https://malumetrix.example.com",
      "data_dir": "/var/lib/malumetrix",
      "accepts": ["onnx", "pt"],
      "reload_command": "docker restart detector",
      "notify_url": "http://127.0.0.1:8000/reload",
      "health_url": "http://127.0.0.1:8000/healthz",
      "health_command": null,
      "health_timeout_s": 60,
      "local_api": "127.0.0.1:7070",
      "keep_versions": 3,
      "models": {"<model_id>": {"reload_command": "systemctl reload line2-detector"}}
    }

Every hook can be set per model under "models"; the top level is the default.
"""
import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional

HOOK_FIELDS = ("reload_command", "notify_url", "health_url", "health_command", "health_timeout_s")

ENV = {
    "url": "MALUMETRIX_URL",
    "data_dir": "MALUMETRIX_DATA_DIR",
    "accepts": "MALUMETRIX_ACCEPTS",            # comma separated, best first
    "reload_command": "MALUMETRIX_RELOAD_COMMAND",
    "notify_url": "MALUMETRIX_NOTIFY_URL",
    "health_url": "MALUMETRIX_HEALTH_URL",
    "health_command": "MALUMETRIX_HEALTH_COMMAND",
    "health_timeout_s": "MALUMETRIX_HEALTH_TIMEOUT",
    "local_api": "MALUMETRIX_LOCAL_API",        # host:port, or "off"
    "keep_versions": "MALUMETRIX_KEEP_VERSIONS",
    "ca_bundle": "MALUMETRIX_CA_BUNDLE",
    "request_timeout_s": "MALUMETRIX_REQUEST_TIMEOUT",
}


@dataclass
class Config:
    url: str = ""
    data_dir: str = "/var/lib/malumetrix"
    accepts: List[str] = field(default_factory=lambda: ["onnx", "pt"])
    reload_command: Optional[str] = None
    notify_url: Optional[str] = None
    health_url: Optional[str] = None
    health_command: Optional[str] = None
    health_timeout_s: int = 60
    local_api: str = "127.0.0.1:7070"
    keep_versions: int = 3
    ca_bundle: Optional[str] = None
    request_timeout_s: int = 30
    models: Dict[str, dict] = field(default_factory=dict)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    def for_model(self, model_id: str) -> "Config":
        """This config with the model's own hooks applied over the defaults."""
        own = {k: v for k, v in (self.models.get(model_id) or {}).items() if k in HOOK_FIELDS}
        return replace(self, **own) if own else self


def _coerce(name: str, raw: str):
    if name == "accepts":
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if name in ("health_timeout_s", "keep_versions", "request_timeout_s"):
        return int(raw)
    return raw


def load(path: Optional[str] = None, **overrides) -> Config:
    """File (if any) < environment < explicit keyword overrides."""
    data: dict = {}
    path = path or os.getenv("MALUMETRIX_CONFIG")
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(Config)}
    values = {k: v for k, v in data.items() if k in known}
    for name, env in ENV.items():
        if os.getenv(env):
            values[name] = _coerce(name, os.environ[env])
    values.update({k: v for k, v in overrides.items() if v is not None and k in known})
    cfg = Config(**values)
    cfg.url = cfg.url.rstrip("/")
    cfg.keep_versions = max(2, cfg.keep_versions)  # the previous version must stay for rollback
    return cfg

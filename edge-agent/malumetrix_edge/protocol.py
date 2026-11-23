"""
The three calls of the device protocol, over plain HTTPS (stdlib only).

    POST /api/v1/edge/enroll       {code}                → {device_id, device_key, …}
    POST /api/v1/edge/heartbeat    report (X-Device-Key) → desired state
    POST /api/v1/edge/rotate-key   (X-Device-Key)        → {device_key}
"""
import json
import random
import ssl
import urllib.error
import urllib.request
from typing import Optional

from . import PROTOCOL, __version__


class ApiError(Exception):
    """The server answered with an error (4xx/5xx)."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class NetworkError(Exception):
    """The server couldn't be reached (offline, DNS, TLS, timeout)."""


def backoff_seconds(attempt: int, base: float, cap: float = 300.0) -> float:
    """Exponential with full jitter: thousands of devices coming back don't call at the same second."""
    return random.uniform(base / 2, min(cap, base * (2 ** max(0, attempt))))


class Client:
    def __init__(self, url: str, device_key: Optional[str] = None, timeout: int = 30, ca_bundle: Optional[str] = None):
        if not url:
            raise ValueError("The platform URL is missing (set MALUMETRIX_URL or 'url' in the config).")
        self.url = url.rstrip("/")
        self.device_key = device_key
        self.timeout = timeout
        self.ssl = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None

    def _post(self, path: str, body: dict, auth: bool = True) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": f"malumetrix-edge/{__version__}"}
        if auth:
            headers["X-Device-Key"] = self.device_key or ""
        req = urllib.request.Request(f"{self.url}/api/v1{path}", data=json.dumps(body, default=str).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read() or b"{}").get("detail", e.reason)
            except Exception:
                detail = e.reason
            raise ApiError(e.code, detail if isinstance(detail, str) else json.dumps(detail)) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise NetworkError(str(e)) from None

    def enroll(self, code: str, accepts=None, hardware=None) -> dict:
        return self._post("/edge/enroll", {"code": code, "agent_version": __version__,
                                           "accepts": accepts, "hardware": hardware}, auth=False)

    def heartbeat(self, payload: dict) -> dict:
        return self._post("/edge/heartbeat", {"protocol": PROTOCOL, "agent_version": __version__, **payload})

    def rotate_key(self) -> str:
        return self._post("/edge/rotate-key", {})["device_key"]

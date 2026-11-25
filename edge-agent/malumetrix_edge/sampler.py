"""
Field samples: which frames to keep, and getting them to the platform.

The platform sends a policy per model with the heartbeat (off until a project
admin turns it on; off at sites that refuse images). Your service asks cheaply
whether a frame is worth keeping (`should_sample`), and hands over the encoded
image only then (`offer_sample`). Reasons, best first:

    class            a detection of a class the policy names
    low_confidence   a detection below the policy's confidence
    random           a few unbiased frames an hour (evenly spaced)

At most `max_per_hour` frames per model are kept. Kept frames wait in a
bounded queue on disk and are uploaded in the background after heartbeats,
so an offline device catches up later; when the queue is full, the oldest go.
"""
import json
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .protocol import ApiError, NetworkError
from .store import write_atomic

log = logging.getLogger("malumetrix_edge")

MAX_QUEUE = 200                     # samples waiting on disk
MAX_QUEUE_BYTES = 500 * 1024 * 1024
UPLOADS_PER_BEAT = 5
FORMATS = ("jpg", "jpeg", "png", "webp")


def _objects(predictions) -> List[dict]:
    out = []
    for p in predictions or ():
        if not isinstance(p, dict):
            continue
        cls = p.get("class", p.get("label", p.get("class_name")))
        conf = p.get("confidence", p.get("score", p.get("conf")))
        try:
            conf = min(max(float(conf), 0.0), 1.0) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        box = p.get("bbox_xyxyn", p.get("bbox"))
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            box = None
        out.append({"class": str(cls)[:120] if cls is not None else None, "confidence": conf,
                    "bbox_xyxyn": [float(v) for v in box] if box else None})
    return out


class Sampler:
    def __init__(self, data_dir: Path, policy_of: Callable[[str], Optional[dict]], version_of: Callable[[str], Optional[str]]):
        self.dir = Path(data_dir) / "samples"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.policy_of, self.version_of = policy_of, version_of
        self._lock = threading.Lock()
        self._kept: Dict[str, deque] = {}      # model_id -> times kept in the last hour
        self._last_random: Dict[str, float] = {}

    # ── choosing ──
    def should_sample(self, model_id: str, predictions, now: Optional[float] = None) -> Optional[str]:
        """The reason this frame is worth keeping, or None. Cheap: call it for every frame."""
        policy = self.policy_of(model_id) or {}
        if not policy.get("enabled"):
            return None
        now = now if now is not None else time.time()
        with self._lock:
            kept = self._kept.setdefault(model_id, deque())
            while kept and now - kept[0] > 3600:
                kept.popleft()
            if len(kept) >= int(policy.get("max_per_hour") or 0):
                return None
            last_random = self._last_random.get(model_id, 0.0)
        objs = _objects(predictions)
        wanted = set(policy.get("classes") or [])
        if wanted and any(o["class"] in wanted for o in objs):
            return "class"
        below = float(policy.get("low_confidence_below") or 0)
        if below > 0 and any(o["confidence"] is not None and o["confidence"] < below for o in objs):
            return "low_confidence"
        per_hour = int(policy.get("random_per_hour") or 0)
        if per_hour > 0 and now - last_random >= 3600 / per_hour:
            return "random"
        return None

    def offer_sample(self, model_id: str, image: bytes, predictions, *, camera: Optional[str] = None, fmt: str = "jpg",
                     width: Optional[int] = None, height: Optional[int] = None, reason: Optional[str] = None,
                     now: Optional[float] = None) -> Optional[str]:
        """Keep this frame if the policy wants it (queued for upload). Returns the reason, or None if not kept."""
        fmt = (fmt or "jpg").lower()
        if fmt not in FORMATS or not image:
            raise ValueError(f"fmt must be one of {FORMATS} and the image non-empty")
        now = now if now is not None else time.time()
        reason = reason or self.should_sample(model_id, predictions, now)
        if reason is None:
            return None
        with self._lock:
            self._kept.setdefault(model_id, deque()).append(now)
            if reason == "random":
                self._last_random[model_id] = now
        sid = uuid.uuid4().hex
        meta = {"model_id": model_id, "version_id": self.version_of(model_id), "camera": (camera or "")[:80],
                "captured_at": datetime.fromtimestamp(now, timezone.utc).isoformat(), "reason": reason,
                "format": fmt, "size_bytes": len(image), "width": width, "height": height,
                "predictions": _objects(predictions)}
        (self.dir / f"{sid}.{fmt}").write_bytes(image)
        write_atomic(self.dir / f"{sid}.json", meta)
        self._trim()
        return reason

    # ── the queue ──
    def pending(self) -> List[Path]:
        return sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime)

    def _image_of(self, meta_path: Path, meta: dict) -> Path:
        return meta_path.with_suffix(f".{meta.get('format', 'jpg')}")

    def _drop(self, meta_path: Path) -> None:
        try:
            meta = json.loads(meta_path.read_text())
            self._image_of(meta_path, meta).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        meta_path.unlink(missing_ok=True)

    def _trim(self) -> None:
        items = self.pending()
        sizes = []
        for p in items:
            try:
                meta = json.loads(p.read_text())
                sizes.append((p, self._image_of(p, meta).stat().st_size))
            except (OSError, ValueError):
                sizes.append((p, 0))
        total = sum(s for _, s in sizes)
        while sizes and (len(sizes) > MAX_QUEUE or total > MAX_QUEUE_BYTES):
            p, s = sizes.pop(0)
            log.warning("sample queue full: dropping the oldest (%s)", p.stem)
            self._drop(p)
            total -= s

    # ── uploading ──
    def upload_some(self, client, limit: int = UPLOADS_PER_BEAT) -> int:
        """Send up to `limit` queued samples. Stops at the first network trouble; drops refused ones."""
        sent = 0
        for meta_path in self.pending()[:limit]:
            try:
                meta = json.loads(meta_path.read_text())
                data = self._image_of(meta_path, meta).read_bytes()
            except (OSError, ValueError):
                self._drop(meta_path)
                continue
            try:
                if not meta.get("_sample_id"):
                    answer = client.create_sample({k: v for k, v in meta.items() if not k.startswith("_")})
                    meta["_sample_id"], meta["_upload"] = answer["sample_id"], answer["upload"]
                    write_atomic(meta_path, meta)
                try:
                    client.upload(meta["_upload"], data)
                except ApiError as e:
                    if e.status in (401, 403) and not meta["_upload"].get("device_key"):
                        # the storage link expired: ask for a new one next time
                        meta.pop("_sample_id", None), meta.pop("_upload", None)
                        write_atomic(meta_path, meta)
                        continue
                    raise
                client.finish_sample(meta["_sample_id"])
                self._drop(meta_path)
                sent += 1
            except NetworkError as e:
                log.info("samples: platform unreachable (%s); %s waiting", e, len(self.pending()))
                break
            except ApiError as e:
                if e.status == 429 or e.status >= 500:
                    break  # over the platform's hourly cap, or its trouble: try later
                log.warning("samples: platform refused %s (%s): dropped", meta_path.stem, e.detail)
                self._drop(meta_path)
        return sent

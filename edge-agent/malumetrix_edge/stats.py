"""
Prediction summaries: every inference your service records, folded into one
small record per hour, model version and camera. Inferences, objects, a
10-bin confidence histogram and counts per class: about 1 KB an hour, sent
with the heartbeat, kept on disk while the platform is unreachable.

The platform compares each site with its own past to spot drift (confidence
dropping, the class mix shifting) without a single image leaving the device.
"""
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .store import read_json, write_atomic

BINS = 10                 # confidence histogram: [0, .1), [.1, .2) … [.9, 1]
MAX_CLASSES = 100         # per bucket; beyond that, counted as "(other)"
MAX_BUCKETS = 7 * 24 * 8  # a week of hours for a few models/cameras; oldest dropped first
SEND_BATCH = 48           # per heartbeat (the server's limit)


def _hour(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0).isoformat()


def _objects(predictions) -> Iterable[tuple]:
    """(class, confidence) from [{"class": "pallet", "confidence": 0.91}, …]. Also accepts
    "label"/"class_name" and "score"/"conf"; anything unreadable is skipped."""
    for p in predictions or ():
        if not isinstance(p, dict):
            continue
        cls = p.get("class", p.get("label", p.get("class_name")))
        conf = p.get("confidence", p.get("score", p.get("conf")))
        try:
            conf = min(max(float(conf), 0.0), 1.0) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        yield (str(cls)[:80] if cls is not None else None, conf)


class PredictionStats:
    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / "prediction_stats.json"
        self._lock = threading.Lock()
        saved = read_json(self._path, {})
        self._buckets: Dict[str, dict] = saved.get("buckets", {}) if isinstance(saved, dict) else {}
        self._sent: Dict[str, int] = {}   # bucket key -> its change counter when last sent

    def record(self, model_id: str, version_id: Optional[str], predictions, camera: Optional[str] = None,
               now: Optional[datetime] = None) -> None:
        """One inference (its list of detections, possibly empty)."""
        hour = _hour(now)
        camera = (camera or "")[:80]
        key = f"{hour}|{model_id}|{version_id or ''}|{camera}"
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = {"hour": hour, "model_id": model_id, "version_id": version_id, "camera": camera,
                                          "inferences": 0, "objects": 0, "confidence_hist": [0] * BINS, "classes": {},
                                          "_n": 0}
                self._trim()
            b["inferences"] += 1
            for cls, conf in _objects(predictions):
                b["objects"] += 1
                if conf is not None:
                    b["confidence_hist"][min(int(conf * BINS), BINS - 1)] += 1
                if cls is not None:
                    classes = b["classes"]
                    if cls not in classes and len(classes) >= MAX_CLASSES:
                        cls = "(other)"
                    classes[cls] = classes.get(cls, 0) + 1
            b["_n"] += 1

    def _trim(self) -> None:
        while len(self._buckets) > MAX_BUCKETS:
            self._buckets.pop(min(self._buckets, key=lambda k: self._buckets[k]["hour"]))

    def save(self) -> None:
        with self._lock:
            write_atomic(self._path, {"buckets": self._buckets})

    def batch(self) -> List[dict]:
        """Buckets changed since they were last sent (the current hour is re-sent as it grows)."""
        with self._lock:
            changed = [(k, b) for k, b in self._buckets.items() if self._sent.get(k) != b["_n"]]
            changed.sort(key=lambda kb: kb[1]["hour"])
            out = []
            for k, b in changed[:SEND_BATCH]:
                out.append({**{f: v for f, v in b.items() if f != "_n"}, "_key": k, "_n": b["_n"]})
            return out

    def sent(self, batch: List[dict]) -> None:
        """The platform has these; closed hours are dropped, the current one is kept to grow."""
        if not batch:
            return
        current = _hour()
        with self._lock:
            for item in batch:
                k = item["_key"]
                b = self._buckets.get(k)
                if b is None:
                    continue
                self._sent[k] = item["_n"]
                if b["hour"] < current and b["_n"] == item["_n"]:
                    self._buckets.pop(k, None)
                    self._sent.pop(k, None)
            write_atomic(self._path, {"buckets": self._buckets})

    @staticmethod
    def wire(batch: List[dict]) -> List[dict]:
        """What goes over the network: the batch without its bookkeeping."""
        return [{k: v for k, v in item.items() if not k.startswith("_")} for item in batch]

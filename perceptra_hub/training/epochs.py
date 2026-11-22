"""
The one place a finished training epoch is recorded.

Every way a run reports progress (the platform GPU trainer, a compute agent,
a cloud provider monitor, PATCH /training-sessions/{id}) calls record_epoch,
so they all keep the same history, best metrics and time-left estimate, and
the project's live stream gets a "training.epoch" event for each epoch.
"""
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from django.utils import timezone

# Enough for any real run; older epochs are dropped first beyond this.
HISTORY_LIMIT = 1000
# A run is "stalled" when no epoch arrived for this many typical epochs...
STALL_FACTOR = 3
# ...and never sooner than this (first epochs, very fast epochs).
STALL_MIN = timedelta(minutes=5)

ACTIVE_STATUSES = {"pending", "queued", "initializing", "running"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

_RAW_KEY = re.compile(r"^metrics/(?P<name>.+?)(?:\((?P<kind>[A-Z])\))?$")


def normalize_key(key: str) -> str:
    """Ultralytics' raw names -> the names model versions use.

    "metrics/mAP50(B)" -> "mAP50", "metrics/mAP50(M)" -> "mask_mAP50",
    "val/box_loss" -> "val_box_loss"; anything else is kept as it is.
    """
    m = _RAW_KEY.match(key)
    if m:
        return ("mask_" if m.group("kind") == "M" else "") + m.group("name")
    return key.replace("/", "_")


def lower_is_better(key: str) -> bool:
    return "loss" in key.lower()


def clean_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Finite numbers only, with normalized names; epoch counters are not metrics."""
    out: Dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if key in ("epoch", "total_epochs") or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[normalize_key(str(key))] = number
    return out


def _parse(at: Optional[str]) -> Optional[datetime]:
    if not at:
        return None
    try:
        return datetime.fromisoformat(at)
    except ValueError:
        return None


def epoch_seconds(session) -> Optional[float]:
    """Typical seconds per epoch: the recent average between recorded epochs."""
    history: List[dict] = session.epoch_history or []
    times = [t for t in (_parse(e.get("at")) for e in history[-20:]) if t]
    if len(times) >= 2:
        span = (times[-1] - times[0]).total_seconds()
        steps = len(times) - 1
        return span / steps if span > 0 else None
    if len(times) == 1 and session.started_at:
        span = (times[0] - session.started_at).total_seconds()
        first = history[-1].get("epoch") or 1
        return span / first if span > 0 else None
    return None


def last_epoch_at(session) -> Optional[datetime]:
    history = session.epoch_history or []
    return _parse(history[-1].get("at")) if history else None


def stalled_for(session, now: Optional[datetime] = None) -> Optional[timedelta]:
    """How long a running run has gone without an epoch, when that is unusually long."""
    if session.status != "running":
        return None
    now = now or timezone.now()
    since = last_epoch_at(session) or session.started_at
    if since is None:
        return None
    typical = epoch_seconds(session)
    limit = max(STALL_MIN, timedelta(seconds=STALL_FACTOR * typical)) if typical else STALL_MIN * 2
    quiet = now - since
    return quiet if quiet > limit else None


def best_epoch(session, key: str) -> Optional[dict]:
    """The history entry with the best value of `key`."""
    entries = [e for e in (session.epoch_history or []) if key in (e.get("metrics") or {})]
    if not entries:
        return None
    pick = min if lower_is_better(key) else max
    return pick(entries, key=lambda e: e["metrics"][key])


def _merge_best(best: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    merged = dict(best) if isinstance(best, dict) else {}
    for key, value in metrics.items():
        current = merged.get(key)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            merged[key] = value
        elif (value < current) if lower_is_better(key) else (value > current):
            merged[key] = value
    return merged


def record_epoch(
    session,
    epoch: int,
    metrics: Optional[Dict[str, Any]],
    *,
    total_epochs: Optional[int] = None,
    at: Optional[datetime] = None,
    publish: bool = True,
) -> dict:
    """Store one finished epoch on the run and tell the project's live stream.

    Re-reporting an epoch replaces it. Saves only the fields it changes, so a
    caller holding the session for the whole run (the platform trainer) can
    keep calling it.
    """
    at = at or timezone.now()
    clean = clean_metrics(metrics)
    entry = {"epoch": int(epoch), "at": at.isoformat(), "metrics": clean}

    history = [e for e in (session.epoch_history or []) if e.get("epoch") != entry["epoch"]]
    history.append(entry)
    history.sort(key=lambda e: e.get("epoch") or 0)
    session.epoch_history = history[-HISTORY_LIMIT:]

    fields = {"epoch_history", "current_metrics", "best_metrics", "current_epoch", "progress", "updated_at"}
    if total_epochs:
        session.total_epochs = int(total_epochs)
        fields.add("total_epochs")
    session.current_epoch = max(session.current_epoch or 0, entry["epoch"])
    if session.total_epochs:
        session.progress = min(100.0, session.current_epoch / session.total_epochs * 100)
    session.current_metrics = clean
    session.best_metrics = _merge_best(session.best_metrics, clean)

    # An epoch means the run is running, whatever the last status said.
    if session.status in ACTIVE_STATUSES and session.status != "running":
        session.status = "running"
        fields.add("status")
    if not session.started_at:
        session.started_at = at
        fields.add("started_at")

    per_epoch = epoch_seconds(session)
    if per_epoch and session.total_epochs:
        remaining = max(0, session.total_epochs - session.current_epoch)
        session.estimated_time_remaining = timedelta(seconds=per_epoch * remaining)
        fields.add("estimated_time_remaining")

    session.save(update_fields=sorted(fields))

    if publish:
        _publish(session, entry)
    return entry


def _publish(session, entry: dict) -> None:
    from realtime.bus import publish_on_commit

    try:
        project = session.model_version.model.project
    except Exception:  # pragma: no cover - a run always has a version and project
        return
    eta = session.estimated_time_remaining
    publish_on_commit(project.project_id, "training.epoch", {
        "session": session.id,
        "model_id": session.model_version.model.model_id,
        "version_id": session.model_version.version_id,
        "status": session.status,
        "epoch": entry["epoch"],
        "total_epochs": session.total_epochs,
        "progress": session.progress,
        "eta_seconds": eta.total_seconds() if eta else None,
        "at": entry["at"],
        "metrics": entry["metrics"],
    })


def publish_status(session) -> None:
    """Tell the live stream a run changed status (started, finished, failed)."""
    from realtime.bus import publish_on_commit

    try:
        project = session.model_version.model.project
    except Exception:  # pragma: no cover
        return
    publish_on_commit(project.project_id, "training.status", {
        "session": session.id,
        "model_id": session.model_version.model.model_id,
        "version_id": session.model_version.version_id,
        "status": session.status,
        "progress": session.progress,
    })

"""
How models do in the field, from hourly prediction summaries sent by devices.

A "place" is a site (all its devices together) or, for a device without a
site, the device itself. Each place is compared with its own past on the same
model version: the baseline is the first BASELINE_HOURS after that version
started reporting there, the recent window is the last RECENT_HOURS. That
catches slow drift as well as sudden change, and never compares a site with
another one (lighting, products and cameras differ).

Signals (all from predictions, no ground truth needed):
    field_confidence      mean confidence fell (relative)
    field_low_confidence  share of objects below 0.5 confidence rose (points)
    field_class_mix       the mix of predicted classes shifted (total variation)
    field_silent          an online device stopped reporting inferences
    field_fps             frames per second fell (health reports)
They become MetricAlerts with a site/device, so the Alerts page,
notifications, the project overview and the "alert" retraining trigger all
see them. Thresholds can be overridden per project in
Project.settings["field_thresholds"].
"""
import logging
from collections import defaultdict
from datetime import timedelta
from statistics import median
from typing import Dict, Iterable, List, Optional

from django.db.models import Sum
from django.utils import timezone

from fleet.models import Device, DeviceModel, HealthSample, PredictionStat

logger = logging.getLogger(__name__)

BINS = PredictionStat.BINS
RECENT_HOURS = 24
BASELINE_HOURS = 72
MIN_INFERENCES = 200        # per window, below that the numbers are noise
MIN_OBJECTS_FOR_MIX = 200
SILENT_HOURS = 6
REALERT_HOURS = 24          # one alert per place, version and signal per day at most

THRESHOLDS = {
    # signal: (warning, critical)
    "field_confidence": (0.10, 0.20),      # relative drop of mean confidence
    "field_low_confidence": (0.10, 0.20),  # rise of the share below 0.5, in points
    "field_class_mix": (0.20, 0.35),       # total variation distance of class shares
    "field_fps": (0.30, 0.50),             # relative drop of median fps
}


# ── storing what devices send ────────────────────────────────────────────────

def store_summaries(device: Device, items: Iterable[dict], now) -> int:
    """Upsert hourly summaries (the current hour arrives several times as it grows)."""
    from ml_models.models import Model, ModelVersion

    models: Dict[str, Optional[Model]] = {}
    versions: Dict[tuple, Optional[ModelVersion]] = {}
    oldest, newest = now - timedelta(days=7), now + timedelta(hours=1)
    stored = 0
    for item in items:
        hour = item.get("hour")
        if hour is None or timezone.is_naive(hour) or not (oldest <= hour <= newest):
            continue
        hour = hour.replace(minute=0, second=0, microsecond=0)
        mid = item.get("model_id")
        if mid not in models:
            models[mid] = Model.objects.filter(model_id=mid, organization_id=device.organization_id, is_deleted=False).first()
        model = models[mid]
        if model is None:
            continue
        vid = item.get("version_id")
        if (mid, vid) not in versions:
            versions[(mid, vid)] = ModelVersion.objects.filter(model=model, version_id=vid).first() if vid else None
        hist = [max(0, int(n)) for n in (item.get("confidence_hist") or [])][:BINS]
        hist += [0] * (BINS - len(hist))
        PredictionStat.objects.update_or_create(
            device=device, model=model, version=versions[(mid, vid)], camera=(item.get("camera") or "")[:80], hour=hour,
            defaults={"inferences": int(item.get("inferences") or 0), "object_count": int(item.get("objects") or 0),
                      "confidence_hist": hist, "class_counts": item.get("classes") or {}})
        stored += 1
    return stored


# ── numbers from summaries ───────────────────────────────────────────────────

def summarize(rows: Iterable[PredictionStat]) -> dict:
    """Totals and derived numbers for a set of hourly summaries."""
    inferences = objects = 0
    hist = [0] * BINS
    classes: Dict[str, int] = defaultdict(int)
    for r in rows:
        inferences += r.inferences
        objects += r.object_count
        for i, n in enumerate((r.confidence_hist or [])[:BINS]):
            hist[i] += n
        for c, n in (r.class_counts or {}).items():
            classes[c] += n
    scored = sum(hist)
    total_cls = sum(classes.values())
    return {
        "inferences": inferences,
        "objects": objects,
        "objects_per_inference": round(objects / inferences, 3) if inferences else None,
        "mean_confidence": round(sum(n * (i + 0.5) / BINS for i, n in enumerate(hist)) / scored, 4) if scored else None,
        "low_confidence_share": round(sum(hist[: BINS // 2]) / scored, 4) if scored else None,
        "confidence_hist": hist,
        "class_share": {c: round(n / total_cls, 4) for c, n in sorted(classes.items(), key=lambda kv: -kv[1])} if total_cls else {},
        "class_objects": total_cls,
    }


def class_shift(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Total variation distance between two class mixes: 0 same, 1 nothing in common."""
    return round(0.5 * sum(abs(a.get(c, 0.0) - b.get(c, 0.0)) for c in set(a) | set(b)), 4)


def _place_of(device: Device):
    return ("site", device.site_id) if device.site_id else ("device", device.pk)


def _thresholds(project) -> dict:
    custom = ((getattr(project, "settings", None) or {}).get("field_thresholds") or {})
    out = dict(THRESHOLDS)
    for k, v in custom.items():
        if k in out and isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = (float(v[0]), float(v[1]))
    return out


def _severity(value: Optional[float], limits) -> Optional[str]:
    if value is None:
        return None
    return "critical" if value >= limits[1] else "warning" if value >= limits[0] else None


# ── per place: baseline against recent ───────────────────────────────────────

def compare_place(model, version, devices: List[Device], now) -> Optional[dict]:
    """The version's baseline at this place against its last RECENT_HOURS, or None when not enough data yet."""
    qs = PredictionStat.objects.filter(model=model, version=version, device__in=devices)
    first = qs.order_by("hour").values_list("hour", flat=True).first()
    if first is None:
        return None
    base_end = first + timedelta(hours=BASELINE_HOURS)
    recent_start = now - timedelta(hours=RECENT_HOURS)
    if base_end > recent_start:
        return None  # the version hasn't run here long enough to have a separate baseline
    baseline = summarize(qs.filter(hour__gte=first, hour__lt=base_end))
    recent = summarize(qs.filter(hour__gte=recent_start))
    if baseline["inferences"] < MIN_INFERENCES or recent["inferences"] < MIN_INFERENCES:
        return None
    out = {"baseline": baseline, "recent": recent, "baseline_from": first, "baseline_to": base_end}
    bc, rc = baseline["mean_confidence"], recent["mean_confidence"]
    out["confidence_drop"] = round((bc - rc) / bc, 4) if bc and rc is not None else None
    bl, rl = baseline["low_confidence_share"], recent["low_confidence_share"]
    out["low_confidence_rise"] = round(rl - bl, 4) if bl is not None and rl is not None else None
    out["class_mix_shift"] = (class_shift(baseline["class_share"], recent["class_share"])
                              if baseline["class_objects"] >= MIN_OBJECTS_FOR_MIX and recent["class_objects"] >= MIN_OBJECTS_FOR_MIX else None)
    return out


def _version_now(model, devices: List[Device], now):
    """The version with most inferences at this place in the recent window."""
    row = (PredictionStat.objects.filter(model=model, device__in=devices, hour__gte=now - timedelta(hours=RECENT_HOURS),
                                         version__isnull=False)
           .values("version").annotate(n=Sum("inferences")).order_by("-n").first())
    if row is None:
        return None
    from ml_models.models import ModelVersion
    return ModelVersion.objects.filter(pk=row["version"]).first()


def _vname(v) -> str:
    if v is None:
        return "?"
    return v.version_name or f"v{v.version_number}"


# ── alerts ───────────────────────────────────────────────────────────────────

def _recent_alert(project, version, site, device, metric, now) -> bool:
    from temporal_analysis.models import MetricAlert
    from django.db.models import Q
    return MetricAlert.objects.filter(project=project, model_version=version, site=site, device=device, metric_name=metric).filter(
        Q(is_acknowledged=False) | Q(alert_date__gte=now - timedelta(hours=REALERT_HOURS))).exists()


def _raise(created, *, project, version, site, device, metric, severity, current, previous, threshold, change, message, now):
    from temporal_analysis.models import MetricAlert
    if _recent_alert(project, version, site, device, metric, now):
        return
    created.append(MetricAlert.objects.create(
        project=project, model_version=version, site=site, device=device, severity=severity, metric_name=metric,
        current_value=current, previous_value=previous, threshold_value=threshold, change_percent=change, message=message))


def check_model(model, now=None) -> list:
    """Compare every place that runs this model with its own baseline; returns new alerts."""
    now = now or timezone.now()
    project = model.project
    limits = _thresholds(project)
    created: list = []
    recent_devices = Device.objects.filter(
        prediction_stats__model=model, prediction_stats__hour__gte=now - timedelta(days=8), revoked_at__isnull=True,
    ).select_related("site").distinct()
    places: Dict[tuple, List[Device]] = defaultdict(list)
    for d in recent_devices:
        places[_place_of(d)].append(d)

    for (kind, _), devices in places.items():
        site = devices[0].site if kind == "site" else None
        device = devices[0] if kind == "device" else None
        where = site.name if site else device.name
        version = _version_now(model, devices, now)
        if version is None:
            continue
        cmp = compare_place(model, version, devices, now)
        if cmp is None:
            continue
        base, rec = cmp["baseline"], cmp["recent"]
        label = f"{model.name} {_vname(version)}"
        sev = _severity(cmp["confidence_drop"], limits["field_confidence"])
        if sev:
            _raise(created, project=project, version=version, site=site, device=device, metric="field_confidence",
                   severity=sev, current=rec["mean_confidence"], previous=base["mean_confidence"],
                   threshold=limits["field_confidence"][0], change=round(-100 * cmp["confidence_drop"], 1), now=now,
                   message=f"{where}: {label} mean confidence fell from {base['mean_confidence']:.2f} to "
                           f"{rec['mean_confidence']:.2f} ({-100 * cmp['confidence_drop']:+.0f} %) in the last {RECENT_HOURS} h")
        sev = _severity(cmp["low_confidence_rise"], limits["field_low_confidence"])
        if sev:
            _raise(created, project=project, version=version, site=site, device=device, metric="field_low_confidence",
                   severity=sev, current=rec["low_confidence_share"], previous=base["low_confidence_share"],
                   threshold=limits["field_low_confidence"][0], change=round(100 * cmp["low_confidence_rise"], 1), now=now,
                   message=f"{where}: {label} detections below 0.5 confidence rose from "
                           f"{base['low_confidence_share']:.0%} to {rec['low_confidence_share']:.0%}")
        sev = _severity(cmp["class_mix_shift"], limits["field_class_mix"])
        if sev:
            moved = sorted(set(base["class_share"]) | set(rec["class_share"]),
                           key=lambda c: -abs(rec["class_share"].get(c, 0) - base["class_share"].get(c, 0)))[:2]
            detail = ", ".join(f"{c} {base['class_share'].get(c, 0):.0%} → {rec['class_share'].get(c, 0):.0%}" for c in moved)
            _raise(created, project=project, version=version, site=site, device=device, metric="field_class_mix",
                   severity=sev, current=cmp["class_mix_shift"], previous=0.0, threshold=limits["field_class_mix"][0],
                   change=None, now=now, message=f"{where}: the mix of classes {label} sees has shifted ({detail})")

    # Per device: stopped reporting while online.
    from fleet import services
    for d in recent_devices:
        if services.device_status(d, now) != "online":
            continue
        had = PredictionStat.objects.filter(model=model, device=d, hour__gte=now - timedelta(hours=RECENT_HOURS + SILENT_HOURS),
                                            hour__lt=now - timedelta(hours=SILENT_HOURS)).aggregate(n=Sum("inferences"))["n"] or 0
        lately = PredictionStat.objects.filter(model=model, device=d, hour__gte=now - timedelta(hours=SILENT_HOURS)).aggregate(n=Sum("inferences"))["n"] or 0
        if had >= MIN_INFERENCES and lately == 0:
            dm = DeviceModel.objects.filter(device=d, model=model).select_related("reported_version").first()
            _raise(created, project=project, version=dm.reported_version if dm else None, site=d.site, device=d,
                   metric="field_silent", severity="warning", current=0, previous=had, threshold=0, change=-100.0, now=now,
                   message=f"{d.name}: online, but {model.name} reported no inferences in the last {SILENT_HOURS} h "
                           f"(it reported {had:,} the day before). Is the inference service running?")
    return created


def fps_drop(device: Device, now) -> Optional[dict]:
    """Median fps of the last day against days 2–8 before; None without enough reports."""
    def fps(since, until):
        vals = [m.get("fps") for m in HealthSample.objects.filter(device=device, at__gte=since, at__lt=until).values_list("metrics", flat=True)]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return median(vals) if len(vals) >= 30 else None
    recent = fps(now - timedelta(hours=RECENT_HOURS), now)
    base = fps(now - timedelta(days=8), now - timedelta(days=2))
    if not recent or not base:
        return None
    return {"recent": round(recent, 2), "baseline": round(base, 2), "drop": round((base - recent) / base, 4)}


def check_fps(device: Device, now=None) -> list:
    now = now or timezone.now()
    drop = fps_drop(device, now)
    created: list = []
    if drop is None:
        return created
    for dm in DeviceModel.objects.filter(device=device, model__is_deleted=False).select_related("model__project", "reported_version"):
        sev = _severity(drop["drop"], _thresholds(dm.model.project)["field_fps"])
        if sev:
            _raise(created, project=dm.model.project, version=dm.reported_version, site=device.site, device=device,
                   metric="field_fps", severity=sev, current=drop["recent"], previous=drop["baseline"],
                   threshold=_thresholds(dm.model.project)["field_fps"][0], change=round(-100 * drop["drop"], 1), now=now,
                   message=f"{device.name}: frames per second fell from {drop['baseline']:.1f} to {drop['recent']:.1f} "
                           f"({-100 * drop['drop']:+.0f} %) while running {dm.model.name}")
    return created


def check_all(now=None) -> int:
    """For the periodic task: every model with recent summaries, every device with health."""
    from ml_models.models import Model
    from api.tasks.evaluation.alert import send_alert_notifications

    now = now or timezone.now()
    created = []
    for model in Model.objects.filter(prediction_stats__hour__gte=now - timedelta(days=8), is_deleted=False).select_related("project").distinct():
        try:
            created += check_model(model, now)
        except Exception:  # one model's trouble never stops the others
            logger.exception("field check failed for model %s", model.model_id)
    for device in Device.objects.filter(revoked_at__isnull=True, enrolled_at__isnull=False, last_seen_at__gte=now - timedelta(days=1)).select_related("site"):
        try:
            created += check_fps(device, now)
        except Exception:
            logger.exception("fps check failed for device %s", device.device_id)
    by_project = defaultdict(list)
    for a in created:
        by_project[a.project_id].append(a.id)
    for project_id, ids in by_project.items():
        send_alert_notifications.apply_async((project_id, ids), queue="evaluation")
    return len(created)


# ── for the console ──────────────────────────────────────────────────────────

def model_field(model, days: int = 14, now=None) -> dict:
    """Per place: daily numbers over `days`, and the comparison with its baseline."""
    now = now or timezone.now()
    since = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = list(PredictionStat.objects.filter(model=model, hour__gte=since).select_related("device__site", "version"))
    devices_by_place: Dict[tuple, Dict[int, Device]] = defaultdict(dict)
    rows_by_place: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = _place_of(r.device)
        devices_by_place[key][r.device.pk] = r.device
        rows_by_place[key].append(r)

    limits = _thresholds(model.project)
    places = []
    for key, prow in rows_by_place.items():
        devices = list(devices_by_place[key].values())
        first = devices[0]
        days_rows: Dict[str, list] = defaultdict(list)
        for r in prow:
            days_rows[r.hour.date().isoformat()].append(r)
        daily = []
        for day in sorted(days_rows):
            s = summarize(days_rows[day])
            daily.append({"date": day, "inferences": s["inferences"], "mean_confidence": s["mean_confidence"],
                          "low_confidence_share": s["low_confidence_share"], "objects_per_inference": s["objects_per_inference"]})
        version = _version_now(model, devices, now)
        cmp = compare_place(model, version, devices, now) if version else None
        recent = summarize([r for r in prow if r.hour >= now - timedelta(hours=RECENT_HOURS)])
        places.append({
            "kind": key[0],
            "site": {"site_id": str(first.site.site_id), "name": first.site.name} if key[0] == "site" else None,
            "device": {"device_id": str(first.device_id), "name": first.name} if key[0] == "device" else None,
            "devices": sorted(d.name for d in devices),
            "version": {"version_id": version.version_id, "version_name": _vname(version)} if version else None,
            "recent": recent,
            "baseline": cmp["baseline"] if cmp else None,
            "baseline_from": cmp["baseline_from"].isoformat() if cmp else None,
            "baseline_to": cmp["baseline_to"].isoformat() if cmp else None,
            "confidence_drop": cmp["confidence_drop"] if cmp else None,
            "low_confidence_rise": cmp["low_confidence_rise"] if cmp else None,
            "class_mix_shift": cmp["class_mix_shift"] if cmp else None,
            # Same thresholds as the alerts (the project's own, if set): warning, critical or None.
            "signals": {
                "confidence": _severity(cmp["confidence_drop"], limits["field_confidence"]) if cmp else None,
                "low_confidence": _severity(cmp["low_confidence_rise"], limits["field_low_confidence"]) if cmp else None,
                "class_mix": _severity(cmp["class_mix_shift"], limits["field_class_mix"]) if cmp else None,
            },
            "daily": daily,
        })
    places.sort(key=lambda p: -(p["recent"]["inferences"] or 0))
    return {"model_id": model.model_id, "model_name": model.name, "days": days, "recent_hours": RECENT_HOURS,
            "baseline_hours": BASELINE_HOURS, "min_inferences": MIN_INFERENCES, "places": places}

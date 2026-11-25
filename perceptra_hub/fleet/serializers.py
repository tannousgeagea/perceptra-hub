"""Fleet objects as the console sees them (plain dicts, ISO dates)."""
from django.utils import timezone

from fleet import services
from fleet.models import DeviceModel, Rollout, RolloutTarget

# The few health numbers lists show; the device page gets the full report.
HEALTH_SUMMARY_KEYS = ("fps", "latency_ms", "cpu_percent", "gpu_percent", "memory_percent", "temperature_c", "disk_free_gb", "errors")


def iso(dt):
    return dt.isoformat() if dt else None


def version_ref(v):
    return {"version_id": v.version_id, "version_name": v.version_name, "version_number": v.version_number} if v else None


def site(s, counts=None):
    out = {"site_id": str(s.site_id), "name": s.name, "location": s.location, "timezone": s.timezone,
           "allow_images": s.allow_images, "created_at": iso(s.created_at)}
    if counts is not None:
        out["devices"] = counts
    return out


def device_model(dm: DeviceModel):
    in_sync = services.in_sync(dm)
    return {
        "model_id": dm.model.model_id, "model_name": dm.model.name, "project_id": str(dm.model.project.project_id),
        "desired": version_ref(dm.desired_version), "reported": version_ref(dm.reported_version),
        "reported_status": dm.reported_status, "reported_error": dm.reported_error or None,
        "reported_at": iso(dm.reported_at), "in_sync": in_sync,
        "rollout_id": str(dm.rollout.rollout_id) if dm.rollout_id else None,
    }


def device(d, now=None, *, detail=False):
    now = now or timezone.now()
    out = {
        "device_id": str(d.device_id), "name": d.name, "labels": d.labels or [],
        "site": {"site_id": str(d.site.site_id), "name": d.site.name} if d.site_id else None,
        "status": services.device_status(d, now), "last_seen_at": iso(d.last_seen_at),
        "enrolled_at": iso(d.enrolled_at), "agent_version": d.agent_version or None,
        "enrollment_pending": bool(d.enrollment_code_hash),
        "enrollment_expires_at": iso(d.enrollment_expires_at),
        "health": {k: d.health.get(k) for k in HEALTH_SUMMARY_KEYS if k in (d.health or {})},
        "models": [device_model(dm) for dm in d.models.all()],
        "camera_count": len(d.cameras.all()),
    }
    if detail:
        out.update({
            "hardware": d.hardware or {}, "accepts": d.accepts or [], "health": d.health or {},
            "last_ip": d.last_ip, "protocol": d.protocol, "poll_seconds": services.poll_seconds(d),
            "revoked_at": iso(d.revoked_at), "created_at": iso(d.created_at),
            "cameras": [{"camera_id": str(c.camera_id), "name": c.name, "position": c.position,
                         "order": c.order, "settings": c.settings or {}} for c in d.cameras.all()],
        })
    return out


def rollout(r: Rollout, now=None, *, targets=False):
    out = {
        "rollout_id": str(r.rollout_id), "status": r.status, "status_reason": r.status_reason or None,
        "model_id": r.model.model_id, "model_name": r.model.name, "project_id": str(r.model.project.project_id),
        "version": version_ref(r.version), "previous_version": version_ref(r.previous_version),
        "is_rollback": r.is_rollback, "strategy": r.strategy, "canary_count": r.canary_count,
        "canary_minutes": r.canary_minutes, "max_failures": r.max_failures, "selector": r.selector or {},
        "created_by": (r.created_by.get_full_name() or r.created_by.email) if r.created_by_id else None,
        "created_at": iso(r.created_at), "canary_healthy_since": iso(r.canary_healthy_since),
        "completed_at": iso(r.completed_at), "progress": services.rollout_counts(r, now),
    }
    if targets:
        now = now or timezone.now()
        out["targets"] = [{
            "device_id": str(t.device.device_id), "device_name": t.device.name,
            "site": t.device.site.name if t.device.site_id else None,
            "device_status": services.device_status(t.device, now),
            "phase": t.phase, "state": t.state, "error": t.error or None,
            "assigned_at": iso(t.assigned_at), "finished_at": iso(t.finished_at),
            # This device's own change: what it ran when asked, and what it was asked to run.
            "from_version": version_ref(t.from_version), "version": version_ref(t.to_version or r.version),
        } for t in r.targets.select_related("device__site", "from_version", "to_version").order_by("phase", "device__name")
            if t.state != RolloutTarget.State.SKIPPED or targets == "all"]
    return out

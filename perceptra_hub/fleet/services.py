"""
The fleet's rules, in one place.

Heartbeat (the whole device protocol, apart from enrolling):
    device → what it runs and how healthy it is
    server → what it should run (+ a download link and checksum when it differs)
The server never pushes. Desired state is held per (device, model), so lost
messages, restarts and long outages fix themselves on the next heartbeat.

Rollouts move devices to a version by setting their desired state: canary
devices first (optionally), then the rest once the canaries have run the new
version for `canary_minutes`. A device that fails the version rolls itself
back; its desired state follows (no retry loop), and the rollout pauses once
more devices failed than it tolerates.
"""
import hashlib
import os
from collections import Counter
from datetime import timedelta
from typing import Iterable, List, Optional

from django.db import transaction
from django.utils import timezone

from fleet import artifacts, keys
from fleet.models import Device, DeviceModel, HealthSample, Rollout, RolloutTarget, Site

DEFAULT_POLL_SECONDS = int(os.getenv("FLEET_POLL_SECONDS", "30"))
ONLINE_MISSED_POLLS = 3            # online while seen within 3 intervals
STALE_AFTER = timedelta(minutes=10)  # then stale, then offline
PROTOCOL = 1

ACTIVE = (Rollout.Status.PREPARING, Rollout.Status.CANARY, Rollout.Status.ROLLING, Rollout.Status.PAUSED)
OPEN_TARGET = (RolloutTarget.State.WAITING, RolloutTarget.State.ASSIGNED)


class FleetError(Exception):
    """A request that breaks a fleet rule; the message is shown to people."""


# ── Devices ──────────────────────────────────────────────────────────────────

def poll_seconds(device: Device) -> int:
    return device.poll_seconds or DEFAULT_POLL_SECONDS


def device_status(device: Device, now=None) -> str:
    """pending (never enrolled), revoked, online, stale or offline."""
    if device.revoked_at:
        return "revoked"
    if not device.enrolled_at:
        return "pending"
    if not device.last_seen_at:
        return "offline"
    age = (now or timezone.now()) - device.last_seen_at
    if age <= timedelta(seconds=poll_seconds(device) * ONLINE_MISSED_POLLS):
        return "online"
    return "stale" if age <= STALE_AFTER else "offline"


def issue_enrollment_code(device: Device) -> str:
    """A new one-time code for the device; any earlier unused code stops working."""
    code, code_hash, expires = keys.new_enrollment_code()
    device.enrollment_code_hash, device.enrollment_expires_at = code_hash, expires
    device.revoked_at = None
    device.save(update_fields=["enrollment_code_hash", "enrollment_expires_at", "revoked_at"])
    return code


def enroll(code: str, *, agent_version: str = "", hardware: Optional[dict] = None,
           accepts: Optional[list] = None, ip: Optional[str] = None):
    """Trade a valid code for a device key. Re-enrolling replaces the old key."""
    now = timezone.now()
    with transaction.atomic():
        device = (Device.objects.select_for_update()
                  .filter(enrollment_code_hash=keys.code_hash(code), enrollment_expires_at__gt=now)
                  .exclude(enrollment_code_hash="").first())
        if device is None:
            raise FleetError("This enrollment code is wrong, used or expired. Create a new one in Fleet.")
        key, prefix, secret_hash = keys.new_device_key()
        device.key_prefix, device.key_hash = prefix, secret_hash
        device.enrollment_code_hash, device.enrollment_expires_at = "", None
        device.enrolled_at, device.revoked_at = now, None
        device.agent_version = agent_version[:40]
        device.hardware = hardware or device.hardware
        device.accepts = accepts or device.accepts
        device.last_seen_at, device.last_ip = now, ip
        device.save()
    return device, key


def authenticate(key: str) -> Optional[Device]:
    parts = keys.split_key(key)
    if not parts:
        return None
    prefix, secret = parts
    for device in Device.objects.filter(key_prefix=prefix, revoked_at__isnull=True, enrolled_at__isnull=False).select_related("organization"):
        if keys.secret_matches(secret, device.key_hash):
            return device
    return None


def rotate_key(device: Device) -> str:
    key, prefix, secret_hash = keys.new_device_key()
    device.key_prefix, device.key_hash = prefix, secret_hash
    device.save(update_fields=["key_prefix", "key_hash"])
    return key


def revoke(device: Device) -> None:
    """The device's key stops working at once; it drops out of open rollouts."""
    with transaction.atomic():
        device.revoked_at = timezone.now()
        device.key_hash, device.enrollment_code_hash, device.enrollment_expires_at = "", "", None
        device.save(update_fields=["revoked_at", "key_hash", "enrollment_code_hash", "enrollment_expires_at"])
        RolloutTarget.objects.filter(device=device, state__in=OPEN_TARGET).update(state=RolloutTarget.State.SKIPPED)


# ── Heartbeat ────────────────────────────────────────────────────────────────

def _minute(dt):
    return dt.replace(second=0, microsecond=0)


def _apply_report(device: Device, report: dict, now) -> Optional[Rollout]:
    """Record what the device says about one model; returns the rollout it affects."""
    from ml_models.models import Model, ModelVersion

    model = Model.objects.filter(model_id=report.get("model_id"), organization_id=device.organization_id).first()
    if model is None:
        return None
    version = ModelVersion.objects.filter(model=model, version_id=report.get("version_id")).first() if report.get("version_id") else None
    failed = ModelVersion.objects.filter(model=model, version_id=report.get("failed_version_id")).first() if report.get("failed_version_id") else None
    status = report.get("status") or DeviceModel.Reported.RUNNING
    if status not in DeviceModel.Reported.values:
        status = DeviceModel.Reported.RUNNING

    dm, _ = DeviceModel.objects.select_for_update().get_or_create(device=device, model=model)
    dm.reported_version, dm.reported_status = version, status
    dm.reported_error = (report.get("error") or "")[:500]
    dm.reported_at = now
    if status == DeviceModel.Reported.FAILED and version is not None and failed is None:
        failed = version
    if failed is not None and dm.desired_version_id == failed.id:
        # The device gave up on the version (and, if it could, went back to the
        # one it ran before). Desired state follows reality: no retry loop.
        dm.desired_version = version if version != failed else None
    dm.save()

    target = (RolloutTarget.objects.select_related("rollout")
              .filter(device=device, rollout__model=model, rollout__status__in=ACTIVE)
              .exclude(state=RolloutTarget.State.SKIPPED).order_by("-rollout__created_at").first())
    if target is None:
        return None
    rv = target.rollout.version_id
    if failed is not None and failed.id == rv and target.state != RolloutTarget.State.FAILED:
        target.state, target.error, target.finished_at = RolloutTarget.State.FAILED, dm.reported_error or "Failed on the device", now
        target.save(update_fields=["state", "error", "finished_at"])
    elif (version is not None and version.id == rv and target.state == RolloutTarget.State.ASSIGNED
          and status in (DeviceModel.Reported.RUNNING, DeviceModel.Reported.ROLLED_BACK)):
        # rolled_back still means "running this version" (after refusing another one).
        target.state, target.error, target.finished_at = RolloutTarget.State.UPDATED, "", now
        target.save(update_fields=["state", "error", "finished_at"])
    return target.rollout


def heartbeat(device: Device, payload: dict, *, ip: Optional[str] = None, request_base: str = "") -> dict:
    """Record the device's report and answer with what it should run."""
    now = timezone.now()
    touched = set()
    with transaction.atomic():
        device = Device.objects.select_for_update().get(pk=device.pk)
        device.last_seen_at, device.last_ip = now, ip
        device.agent_version = (payload.get("agent_version") or device.agent_version)[:40]
        device.protocol = int(payload.get("protocol") or PROTOCOL)
        if payload.get("accepts"):
            device.accepts = [str(f).lower() for f in payload["accepts"]][:10]
        if payload.get("hardware"):
            device.hardware = payload["hardware"]
        health = payload.get("health")
        if health:
            device.health = {**health, "at": now.isoformat()}
            HealthSample.objects.update_or_create(device=device, at=_minute(now), defaults={"metrics": health})
        # Reports kept while offline: stored at their own minute, within the retention window, never in the future.
        oldest = now - timedelta(days=7)
        for point in (payload.get("health_backlog") or [])[:120]:
            at = point.get("at")
            if at is None or timezone.is_naive(at) or not (oldest <= at <= now):
                continue
            HealthSample.objects.update_or_create(device=device, at=_minute(at), defaults={"metrics": point.get("metrics") or {}})
        device.save()
        for report in payload.get("models") or []:
            rollout = _apply_report(device, report, now)
            if rollout is not None:
                touched.add(rollout.pk)
    for pk in touched:
        advance(Rollout.objects.get(pk=pk))
    return desired_state(device, state_token=payload.get("state_token"), refresh=bool(payload.get("refresh")), request_base=request_base)


def desired_state(device: Device, *, state_token: Optional[str] = None, refresh: bool = False, request_base: str = "") -> dict:
    """Every model the device should run. Cheap 'unchanged' answer when nothing moved."""
    entries = []
    for dm in (DeviceModel.objects.filter(device=device, desired_version__isnull=False)
               .select_related("model", "desired_version__storage_profile", "desired_version__model__task",
                               "desired_version__model__framework").order_by("model__name")):
        art = artifacts.ready_artifact(dm.desired_version, device.accepts)
        entries.append((dm, art))
    token = hashlib.sha1("|".join(
        f"{dm.model.model_id}:{dm.desired_version.version_id}:{art.sha256 if art else 'pending'}" for dm, art in entries
    ).encode()).hexdigest()[:16]
    base = {"protocol": PROTOCOL, "device_id": str(device.device_id), "poll_seconds": poll_seconds(device),
            "state_token": token, "server_time": timezone.now().isoformat()}
    if state_token == token and not refresh:
        return {**base, "unchanged": True}
    models = []
    for dm, art in entries:
        v = dm.desired_version
        entry = {"model_id": dm.model.model_id, "model_name": dm.model.name, "version_id": v.version_id,
                 "version_name": v.version_name, "rollout_id": str(dm.rollout.rollout_id) if dm.rollout_id else None,
                 "ready": art is not None, "artifact": None, "manifest": None}
        if art is not None:
            url, expires = artifacts.download_url(art, request_base)
            entry["artifact"] = {"format": art.format, "url": url, "sha256": art.sha256,
                                 "size_bytes": art.size_bytes, "expires_at": expires.isoformat()}
            entry["manifest"] = art.manifest
        models.append(entry)
    return {**base, "unchanged": False, "models": models}


# ── Rollouts ─────────────────────────────────────────────────────────────────

def select_devices(organization, *, model=None, device_ids: Iterable = (), site_ids: Iterable = (),
                   labels: Iterable = (), running_model: bool = False) -> List[Device]:
    """Enrolled, unrevoked devices matching any of the selectors."""
    from django.db.models import Q

    q = Q(pk__in=[])
    if device_ids:
        q |= Q(device_id__in=list(device_ids))
    if site_ids:
        q |= Q(site__site_id__in=list(site_ids))
    for label in labels or ():
        q |= Q(labels__contains=[label])
    if running_model and model is not None:
        q |= Q(models__model=model, models__desired_version__isnull=False) | Q(models__model=model, models__reported_version__isnull=False)
    return list(Device.objects.filter(q, organization=organization, enrolled_at__isnull=False, revoked_at__isnull=True)
                .distinct().order_by("name"))


def _supersede(model, except_pk=None):
    for old in Rollout.objects.select_for_update().filter(model=model, status__in=ACTIVE).exclude(pk=except_pk):
        old.status, old.status_reason, old.completed_at = Rollout.Status.SUPERSEDED, "A newer rollout of this model took over.", timezone.now()
        old.save(update_fields=["status", "status_reason", "completed_at"])
        old.targets.filter(state__in=OPEN_TARGET).update(state=RolloutTarget.State.SKIPPED)


def create_rollout(organization, model, version, devices: List[Device], *, user=None, strategy: str = "all",
                   canary_count: int = 1, canary_minutes: int = 15, max_failures: int = 0,
                   selector: Optional[dict] = None, is_rollback: bool = False, previous_version=None) -> Rollout:
    if version.model_id != model.id:
        raise FleetError("That version belongs to another model.")
    if not artifacts.available_formats(version):
        raise FleetError(f"{version.version_name} has no model file to deploy (no ONNX or checkpoint).")
    if not devices:
        raise FleetError("No devices match. Pick at least one enrolled device, site or label.")
    if strategy not in Rollout.Strategy.values:
        raise FleetError("Strategy must be 'all' or 'canary'.")
    if strategy == Rollout.Strategy.CANARY and canary_count >= len(devices):
        strategy = Rollout.Strategy.ALL  # a canary of everything is just "all at once"

    if previous_version is None:
        current = Counter(DeviceModel.objects.filter(device__in=devices, model=model, desired_version__isnull=False)
                          .exclude(desired_version=version).values_list("desired_version", flat=True))
        if current:
            from ml_models.models import ModelVersion
            previous_version = ModelVersion.objects.filter(pk=current.most_common(1)[0][0]).first()

    now = timezone.now()
    with transaction.atomic():
        _supersede(model)
        rollout = Rollout.objects.create(
            organization=organization, model=model, version=version, previous_version=previous_version,
            is_rollback=is_rollback, strategy=strategy, canary_count=canary_count, canary_minutes=canary_minutes,
            max_failures=max_failures, selector=selector or {}, created_by=user)
        # Canaries: devices that are online right now first, so the canary actually runs.
        ordered = sorted(devices, key=lambda d: (device_status(d, now) != "online", d.name))
        n_canary = canary_count if strategy == Rollout.Strategy.CANARY else 0
        RolloutTarget.objects.bulk_create([
            RolloutTarget(rollout=rollout, device=d, phase=RolloutTarget.Phase.CANARY if i < n_canary else RolloutTarget.Phase.MAIN)
            for i, d in enumerate(ordered)])
    if artifacts.is_prepared(version):
        start(rollout)
    else:
        from api.tasks.fleet import prepare_rollout
        transaction.on_commit(lambda: prepare_rollout.apply_async(args=[rollout.pk]))
    return rollout


def start(rollout: Rollout) -> Rollout:
    """After the files are prepared: assign the first wave, or fail if nothing can be downloaded."""
    rollout.refresh_from_db()
    if rollout.status != Rollout.Status.PREPARING:
        return rollout
    if artifacts.ready_artifact(rollout.version, None) is None:
        errors = [a.error for a in rollout.version.fleet_artifacts.all() if a.error]
        rollout.status = Rollout.Status.FAILED
        rollout.status_reason = f"Couldn't prepare the model file: {errors[0] if errors else 'no file found'}"
        rollout.completed_at = timezone.now()
        rollout.save(update_fields=["status", "status_reason", "completed_at"])
        return rollout
    canary = rollout.strategy == Rollout.Strategy.CANARY
    rollout.status = Rollout.Status.CANARY if canary else Rollout.Status.ROLLING
    rollout.save(update_fields=["status"])
    _assign(rollout, rollout.targets.filter(state=RolloutTarget.State.WAITING,
                                            phase=RolloutTarget.Phase.CANARY if canary else RolloutTarget.Phase.MAIN))
    return advance(rollout)


def _assign(rollout: Rollout, targets) -> None:
    now = timezone.now()
    for t in targets.select_related("device"):
        dm, _ = DeviceModel.objects.get_or_create(device=t.device, model=rollout.model)
        dm.desired_version, dm.rollout = rollout.version, rollout
        dm.save(update_fields=["desired_version", "rollout", "updated_at"])
        already = (dm.reported_version_id == rollout.version_id
                   and dm.reported_status in (DeviceModel.Reported.RUNNING, DeviceModel.Reported.ROLLED_BACK))
        t.state = RolloutTarget.State.UPDATED if already else RolloutTarget.State.ASSIGNED
        t.assigned_at = now
        t.finished_at = now if already else None
        t.save(update_fields=["state", "assigned_at", "finished_at"])


def advance(rollout: Rollout, now=None) -> Rollout:
    """Move a running rollout forward: pause on failures, promote past the canary, complete."""
    now = now or timezone.now()
    with transaction.atomic():
        rollout = Rollout.objects.select_for_update().get(pk=rollout.pk)
        if rollout.status not in (Rollout.Status.CANARY, Rollout.Status.ROLLING):
            return rollout
        targets = list(rollout.targets.all())
        failed = [t for t in targets if t.state == RolloutTarget.State.FAILED]
        if len(failed) > rollout.max_failures:
            rollout.status = Rollout.Status.PAUSED
            rollout.status_reason = (f"Paused: {len(failed)} device{'s' if len(failed) != 1 else ''} failed to run "
                                     f"{rollout.version.version_name}. First error: {failed[0].error}")[:500]
            rollout.save(update_fields=["status", "status_reason"])
            return rollout

        if rollout.status == Rollout.Status.CANARY:
            canaries = [t for t in targets if t.phase == RolloutTarget.Phase.CANARY and t.state != RolloutTarget.State.SKIPPED]
            if canaries and all(t.state == RolloutTarget.State.UPDATED for t in canaries):
                if rollout.canary_healthy_since is None:
                    rollout.canary_healthy_since = max(t.finished_at or now for t in canaries)
                    rollout.save(update_fields=["canary_healthy_since"])
                if now - rollout.canary_healthy_since >= timedelta(minutes=rollout.canary_minutes):
                    rollout.status, rollout.status_reason = Rollout.Status.ROLLING, ""
                    rollout.save(update_fields=["status", "status_reason"])
                    _assign(rollout, rollout.targets.filter(state=RolloutTarget.State.WAITING))
                    targets = list(rollout.targets.all())
            elif not canaries:  # every canary dropped out: go on with the rest
                rollout.status = Rollout.Status.ROLLING
                rollout.save(update_fields=["status"])
                _assign(rollout, rollout.targets.filter(state=RolloutTarget.State.WAITING))
                targets = list(rollout.targets.all())

        if rollout.status == Rollout.Status.ROLLING and not any(t.state in OPEN_TARGET for t in targets):
            rollout.status, rollout.completed_at = Rollout.Status.COMPLETED, now
            rollout.status_reason = ""
            rollout.save(update_fields=["status", "completed_at", "status_reason"])
    return rollout


def pause(rollout: Rollout, reason: str = "Paused by hand.") -> Rollout:
    if rollout.status not in (Rollout.Status.CANARY, Rollout.Status.ROLLING):
        raise FleetError("Only a rollout in progress can be paused.")
    rollout.status, rollout.status_reason = Rollout.Status.PAUSED, reason[:500]
    rollout.save(update_fields=["status", "status_reason"])
    return rollout


def resume(rollout: Rollout) -> Rollout:
    """Go on. Failures so far are accepted (the tolerance rises to cover them)."""
    if rollout.status != Rollout.Status.PAUSED:
        raise FleetError("Only a paused rollout can be resumed.")
    failed = rollout.targets.filter(state=RolloutTarget.State.FAILED).count()
    canary_left = rollout.targets.filter(phase=RolloutTarget.Phase.CANARY, state=RolloutTarget.State.ASSIGNED).exists()
    main_assigned = rollout.targets.filter(phase=RolloutTarget.Phase.MAIN).exclude(state=RolloutTarget.State.WAITING).exists()
    rollout.max_failures = max(rollout.max_failures, failed)
    rollout.status = Rollout.Status.CANARY if (rollout.strategy == Rollout.Strategy.CANARY and (canary_left or not main_assigned)) else Rollout.Status.ROLLING
    rollout.status_reason = ""
    rollout.save(update_fields=["max_failures", "status", "status_reason"])
    if rollout.status == Rollout.Status.ROLLING:
        _assign(rollout, rollout.targets.filter(state=RolloutTarget.State.WAITING))
    return advance(rollout)


def rollback(rollout: Rollout, user=None) -> Rollout:
    """Put every device this rollout reached back on the version it replaced."""
    if rollout.previous_version is None:
        raise FleetError("There is no earlier version to go back to for these devices.")
    devices = [t.device for t in rollout.targets.select_related("device")
               .filter(state__in=(RolloutTarget.State.ASSIGNED, RolloutTarget.State.UPDATED, RolloutTarget.State.FAILED))
               .filter(device__revoked_at__isnull=True)]
    if not devices:
        raise FleetError("No device got this version yet, so there is nothing to roll back.")
    return create_rollout(rollout.organization, rollout.model, rollout.previous_version, devices, user=user,
                          strategy=Rollout.Strategy.ALL, selector={"rollback_of": str(rollout.rollout_id)},
                          is_rollback=True, previous_version=rollout.version)


def advance_all() -> int:
    """For the periodic task: canary timers run out even when no device calls."""
    n = 0
    for r in Rollout.objects.filter(status__in=(Rollout.Status.CANARY, Rollout.Status.ROLLING)):
        advance(r)
        n += 1
    return n


def rollout_counts(rollout: Rollout, now=None) -> dict:
    """Progress for people: updated / failed / waiting, and how many targets are offline."""
    now = now or timezone.now()
    c = Counter()
    for t in rollout.targets.select_related("device"):
        c[t.state] += 1
        if t.state in OPEN_TARGET and device_status(t.device, now) in ("offline", "stale"):
            c["offline"] += 1
    total = sum(v for k, v in c.items() if k in RolloutTarget.State.values) - c[RolloutTarget.State.SKIPPED]
    return {"total": total, "updated": c[RolloutTarget.State.UPDATED], "failed": c[RolloutTarget.State.FAILED],
            "assigned": c[RolloutTarget.State.ASSIGNED], "waiting": c[RolloutTarget.State.WAITING],
            "skipped": c[RolloutTarget.State.SKIPPED], "offline": c["offline"]}

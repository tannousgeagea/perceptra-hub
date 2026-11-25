"""
The deployment fleet: where models actually run.

    Site ─┬─ Device ─┬─ Camera
          │          └─ DeviceModel   one per (device, model): the version it
          │                           should run (desired) and the version it
          │                           says it runs (reported)
    Rollout ── RolloutTarget          moves a set of devices to a version

Devices pull: every heartbeat reports what runs and gets back what should run
(fleet/services.py). Nothing is ever pushed, so a device that was offline
simply catches up on its next heartbeat.
"""
import uuid

from django.conf import settings
from django.db import models


class Site(models.Model):
    """A physical place devices are installed in: a plant, a store, a line."""
    site_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='fleet_sites')
    name = models.CharField(max_length=120)
    location = models.CharField(max_length=255, blank=True, default='')
    timezone = models.CharField(max_length=64, blank=True, default='', help_text="IANA name, e.g. Europe/Berlin")
    # Off: devices here send no images (privacy), only prediction summaries.
    allow_images = models.BooleanField(default=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'fleet_site'
        ordering = ['name']
        constraints = [models.UniqueConstraint(fields=['organization', 'name'], name='fleet_site_unique_name')]

    def __str__(self):
        return self.name


class Device(models.Model):
    """
    One machine that runs models. It authenticates with its own device key
    (never a person's login or an API key), obtained once with an enrollment
    code. Only the key's hash is stored.
    """
    device_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='fleet_devices')
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name='devices')
    name = models.CharField(max_length=120)
    labels = models.JSONField(default=list, blank=True, help_text="Free-form tags used to target rollouts, e.g. [\"line-2\", \"gpu\"]")

    # Enrollment: a one-time code (hashed) until the device trades it for a key.
    enrollment_code_hash = models.CharField(max_length=64, blank=True, default='')
    enrollment_expires_at = models.DateTimeField(null=True, blank=True)
    enrolled_at = models.DateTimeField(null=True, blank=True)
    key_prefix = models.CharField(max_length=16, blank=True, default='', db_index=True)
    key_hash = models.CharField(max_length=64, blank=True, default='')
    revoked_at = models.DateTimeField(null=True, blank=True)

    # Last reported by the device.
    last_seen_at = models.DateTimeField(null=True, blank=True, db_index=True)
    last_ip = models.GenericIPAddressField(null=True, blank=True)
    agent_version = models.CharField(max_length=40, blank=True, default='')
    protocol = models.PositiveSmallIntegerField(default=1)
    hardware = models.JSONField(default=dict, blank=True)
    accepts = models.JSONField(default=list, blank=True, help_text="Model formats the device can load, best first")
    health = models.JSONField(default=dict, blank=True, help_text="The latest health report")

    poll_seconds = models.PositiveIntegerField(null=True, blank=True, help_text="Heartbeat interval; empty uses the default")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'fleet_device'
        ordering = ['name']
        constraints = [models.UniqueConstraint(fields=['organization', 'name'], name='fleet_device_unique_name')]

    def __str__(self):
        return self.name


class Camera(models.Model):
    """A camera or sensor attached to a device."""
    camera_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='cameras')
    name = models.CharField(max_length=120)
    position = models.CharField(max_length=120, blank=True, default='', help_text="e.g. front, top, bunker")
    order = models.PositiveIntegerField(default=0)
    settings = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'fleet_camera'
        ordering = ['order', 'name']
        constraints = [models.UniqueConstraint(fields=['device', 'name'], name='fleet_camera_unique_name')]

    def __str__(self):
        return f"{self.name} ({self.device})"


class Rollout(models.Model):
    """Moves a set of devices to one version of a model, all at once or canary first."""

    class Strategy(models.TextChoices):
        ALL = 'all', 'All at once'
        CANARY = 'canary', 'Canary first'

    class Status(models.TextChoices):
        PREPARING = 'preparing', 'Preparing'      # artifacts being checksummed
        CANARY = 'canary', 'Canary'               # canary devices updating / being watched
        ROLLING = 'rolling', 'Rolling out'        # every target assigned
        PAUSED = 'paused', 'Paused'
        COMPLETED = 'completed', 'Completed'
        SUPERSEDED = 'superseded', 'Superseded'   # a newer rollout of the model took over
        FAILED = 'failed', 'Failed'               # could not start (e.g. no artifact)

    rollout_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='fleet_rollouts')
    model = models.ForeignKey('ml_models.Model', on_delete=models.CASCADE, related_name='rollouts')
    version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.CASCADE, related_name='rollouts')
    previous_version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    is_rollback = models.BooleanField(default=False)

    strategy = models.CharField(max_length=10, choices=Strategy.choices, default=Strategy.ALL)
    canary_count = models.PositiveIntegerField(default=1)
    canary_minutes = models.PositiveIntegerField(default=15, help_text="How long canaries must stay healthy before the rest follow")
    max_failures = models.PositiveIntegerField(default=0, help_text="Failed devices tolerated before the rollout pauses")
    selector = models.JSONField(default=dict, blank=True, help_text="How targets were chosen: sites, devices, labels")

    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PREPARING, db_index=True)
    status_reason = models.CharField(max_length=500, blank=True, default='')
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)
    canary_healthy_since = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'fleet_rollout'
        ordering = ['-created_at']
        indexes = [models.Index(fields=['model', 'status'])]

    def __str__(self):
        return f"{self.model} → {self.version.version_name} ({self.status})"


class RolloutTarget(models.Model):
    class Phase(models.TextChoices):
        CANARY = 'canary', 'Canary'
        MAIN = 'main', 'Main'

    class State(models.TextChoices):
        WAITING = 'waiting', 'Waiting'      # not assigned yet (behind the canary)
        ASSIGNED = 'assigned', 'Assigned'   # told to run the version, not confirmed
        UPDATED = 'updated', 'Updated'      # reports running the version
        FAILED = 'failed', 'Failed'         # reported a failure for the version
        SKIPPED = 'skipped', 'Skipped'      # taken over by a newer rollout or removed

    rollout = models.ForeignKey(Rollout, on_delete=models.CASCADE, related_name='targets')
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='rollout_targets')
    phase = models.CharField(max_length=8, choices=Phase.choices, default=Phase.MAIN)
    state = models.CharField(max_length=10, choices=State.choices, default=State.WAITING, db_index=True)
    error = models.CharField(max_length=500, blank=True, default='')
    assigned_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    # What the device ran when this rollout asked it to update (set at assignment).
    # Rollback sends each device back to its own from_version.
    from_version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    # Only for rollbacks, where devices go back to different versions. Empty: the rollout's version.
    to_version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')

    class Meta:
        db_table = 'fleet_rollout_target'
        constraints = [models.UniqueConstraint(fields=['rollout', 'device'], name='fleet_rollout_target_unique')]


class DeviceModel(models.Model):
    """What one device should run for one model, and what it says it runs."""

    class Reported(models.TextChoices):
        PENDING = 'pending', 'Pending'
        DOWNLOADING = 'downloading', 'Downloading'
        RUNNING = 'running', 'Running'
        FAILED = 'failed', 'Failed'
        ROLLED_BACK = 'rolled_back', 'Rolled back'

    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='models')
    model = models.ForeignKey('ml_models.Model', on_delete=models.CASCADE, related_name='device_models')
    desired_version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    rollout = models.ForeignKey(Rollout, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    reported_version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    reported_status = models.CharField(max_length=12, choices=Reported.choices, default=Reported.PENDING)
    reported_error = models.CharField(max_length=500, blank=True, default='')
    reported_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'fleet_device_model'
        constraints = [models.UniqueConstraint(fields=['device', 'model'], name='fleet_device_model_unique')]


class ModelArtifact(models.Model):
    """
    A version's file in one format, with the checksum devices verify and the
    manifest they read (classes, input size…). Prepared once per version and
    format, before any device is told to download it.
    """
    version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.CASCADE, related_name='fleet_artifacts')
    format = models.CharField(max_length=16)  # onnx, pt
    storage_key = models.CharField(max_length=1024)
    sha256 = models.CharField(max_length=64, blank=True, default='')
    size_bytes = models.BigIntegerField(null=True, blank=True)
    manifest = models.JSONField(default=dict, blank=True)
    error = models.CharField(max_length=500, blank=True, default='')
    prepared_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'fleet_model_artifact'
        constraints = [models.UniqueConstraint(fields=['version', 'format'], name='fleet_artifact_unique')]


class HealthSample(models.Model):
    """One health report per device per minute, kept for the charts (pruned daily)."""
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='health_samples')
    at = models.DateTimeField()  # truncated to the minute
    metrics = models.JSONField(default=dict)

    class Meta:
        db_table = 'fleet_health_sample'
        constraints = [models.UniqueConstraint(fields=['device', 'at'], name='fleet_health_unique_minute')]
        indexes = [models.Index(fields=['at'])]


class PredictionStat(models.Model):
    """
    What one device's model saw in one hour, on one camera: every inference the
    inference service recorded, summarised on the device (no images). The
    current hour is re-sent as it grows and replaced here.
    """
    BINS = 10  # confidence histogram: [0, .1) … [.9, 1]

    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='prediction_stats')
    model = models.ForeignKey('ml_models.Model', on_delete=models.CASCADE, related_name='prediction_stats')
    # Null when the device ran a version this platform doesn't know (or none yet).
    version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='prediction_stats')
    camera = models.CharField(max_length=80, blank=True, default='')
    hour = models.DateTimeField()
    inferences = models.PositiveIntegerField(default=0)
    object_count = models.PositiveIntegerField(default=0)  # detections ("objects" would clash with the manager)
    confidence_hist = models.JSONField(default=list)   # BINS counts of objects by confidence
    class_counts = models.JSONField(default=dict)      # {class name: objects}
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'fleet_prediction_stat'
        constraints = [models.UniqueConstraint(fields=['device', 'model', 'version', 'camera', 'hour'],
                                               name='fleet_prediction_stat_unique_hour')]
        indexes = [models.Index(fields=['model', 'hour']), models.Index(fields=['hour'])]


class SamplingPolicy(models.Model):
    """
    Which field images devices keep for a model (sent to them in the desired
    state). Off until a project admin turns it on: images leave customer sites.
    A site can refuse images whatever the policy (Site.allow_images).
    """
    model = models.OneToOneField('ml_models.Model', on_delete=models.CASCADE, related_name='sampling_policy')
    enabled = models.BooleanField(default=False)
    low_confidence_below = models.FloatField(default=0.4, help_text="Keep frames with a detection below this confidence (0 = off)")
    classes = models.JSONField(default=list, blank=True, help_text="Keep frames with any of these classes")
    random_per_hour = models.PositiveIntegerField(default=2, help_text="Unbiased frames per hour and device")
    max_per_hour = models.PositiveIntegerField(default=10, help_text="Cap per hour and device, all reasons together")
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'fleet_sampling_policy'


class FieldSample(models.Model):
    """
    One image a device kept, with what the model found in it. Uploaded straight
    to storage when it can mint upload links, else through the API; then
    registered into the model's project (prediction labels to review), where
    active learning ranks it. Unreviewed samples are deleted after 30 days.
    """
    class Reason(models.TextChoices):
        LOW_CONFIDENCE = 'low_confidence', 'Low confidence'
        CLASS = 'class', 'Class of interest'
        RANDOM = 'random', 'Random'

    class Status(models.TextChoices):
        UPLOADING = 'uploading', 'Waiting for the file'
        REGISTERED = 'registered', 'In the project'
        FAILED = 'failed', 'Failed'

    sample_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='field_samples')
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='field_samples')
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name='field_samples')  # where it was taken
    model = models.ForeignKey('ml_models.Model', on_delete=models.CASCADE, related_name='field_samples')
    version = models.ForeignKey('ml_models.ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, related_name='field_samples')
    camera = models.CharField(max_length=80, blank=True, default='')
    captured_at = models.DateTimeField()
    reason = models.CharField(max_length=20, choices=Reason.choices)
    predictions = models.JSONField(default=list)
    storage_profile = models.ForeignKey('storage.StorageProfile', on_delete=models.PROTECT, related_name='+')
    storage_key = models.CharField(max_length=500)
    file_format = models.CharField(max_length=10, default='jpg')
    size_bytes = models.PositiveIntegerField(default=0)
    # From the device, or read from the bytes on an API upload; else probed from storage when registering.
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.UPLOADING, db_index=True)
    error = models.CharField(max_length=500, blank=True, default='')
    image = models.ForeignKey('images.Image', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    project_image = models.ForeignKey('projects.ProjectImage', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    registered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'fleet_field_sample'
        ordering = ['-captured_at']
        indexes = [models.Index(fields=['model', '-captured_at']), models.Index(fields=['device', 'created_at'])]

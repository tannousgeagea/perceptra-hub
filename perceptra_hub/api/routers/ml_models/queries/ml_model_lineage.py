"""
Where a model version came from, and what changed between two versions.

    GET /api/v1/models/{model_id}/versions/{version_id}/lineage
    GET /api/v1/models/{model_id}/compare?a={version_id}&b={version_id}

Read-only and built from what is already stored: the version's metrics and
settings, its dataset version (images and splits), its training run and its
deployments. Objects per class are counted from the active annotations on the
dataset version's images. Anyone who can view the model's project.
"""
from typing import Any, Dict, List, Optional

from django.db.models import Count
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import RequestContext, get_request_context
from projects.dependencies import require_project_permission

router = APIRouter(prefix="/models")

# Metrics where a smaller number is better; anything else named here is
# bookkeeping rather than quality, so it gets no verdict.
LOWER_IS_BETTER = ("loss",)
NO_VERDICT = {"epoch", "learning_rate", "lr"}
# Changes smaller than this are "the same" (metric noise between runs).
SAME_EPSILON = 1e-4


# ---------------------------------------------------------------- schemas

class VersionRef(BaseModel):
    version_id: str
    version_number: int
    label: str
    status: str
    deployment_status: str
    created_at: str


class ClassCount(BaseModel):
    name: str
    count: int


class DatasetSummary(BaseModel):
    id: int
    version_number: int
    label: str
    total_images: int
    total_annotations: int
    train_count: int
    val_count: int
    test_count: int
    created_at: str
    classes: List[ClassCount]


class TrainingSummary(BaseModel):
    id: int
    status: str
    started_by: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    epochs_run: Optional[int] = None
    total_epochs: Optional[int] = None
    best_metrics: Dict[str, Any] = {}
    compute_resource: Optional[str] = None


class DeploymentSummary(BaseModel):
    target_env: str
    deployed_at: str
    undeployed_at: Optional[str] = None
    is_active: bool
    deployed_by: Optional[str] = None


class VersionDetail(VersionRef):
    model_size: str
    created_by: Optional[str] = None
    deployed_at: Optional[str] = None
    tags: List[str]
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    error_message: Optional[str] = None


class LineageResponse(BaseModel):
    version: VersionDetail
    parent: Optional[VersionRef] = None
    children: List[VersionRef]
    previous: Optional[VersionRef] = None
    next: Optional[VersionRef] = None
    production: Optional[VersionRef] = None
    dataset: Optional[DatasetSummary] = None
    training: Optional[TrainingSummary] = None
    deployments: List[DeploymentSummary]


class MetricChange(BaseModel):
    key: str
    a: Optional[float] = None
    b: Optional[float] = None
    delta: Optional[float] = None
    # "higher" | "lower" | None (no better direction)
    better: Optional[str] = None
    # "improved" | "worse" | "same" | None (missing on one side, or no direction)
    verdict: Optional[str] = None


class SettingChange(BaseModel):
    key: str
    a: Any = None
    b: Any = None


class ClassChange(BaseModel):
    name: str
    a: int
    b: int
    delta: int
    # Objects of this class on the images only B trained on.
    in_added_images: int


class DataChange(BaseModel):
    same_dataset: bool
    a: Optional[DatasetSummary] = None
    b: Optional[DatasetSummary] = None
    # Images, by project image, between the two dataset versions.
    only_a: int = 0
    only_b: int = 0
    shared: int = 0
    # Shared images that sit in a different split (e.g. train -> val).
    moved_split: int = 0
    classes: List[ClassChange] = []


class CompareResponse(BaseModel):
    a: VersionDetail
    b: VersionDetail
    a_training: Optional[TrainingSummary] = None
    b_training: Optional[TrainingSummary] = None
    production_version_id: Optional[str] = None
    metrics: List[MetricChange]
    settings_changed: List[SettingChange]
    settings_identical: int
    data: DataChange


# ---------------------------------------------------------------- helpers

def _person(user) -> Optional[str]:
    if user is None:
        return None
    return f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username or user.email


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def _label(version) -> str:
    return version.version_name or f"v{version.version_number}"


def _ref(version) -> VersionRef:
    return VersionRef(
        version_id=version.version_id,
        version_number=version.version_number,
        label=_label(version),
        status=version.status,
        deployment_status=version.deployment_status,
        created_at=version.created_at.isoformat(),
    )


def _detail(version) -> VersionDetail:
    return VersionDetail(
        **_ref(version).model_dump(),
        model_size=version.model_size or "",
        created_by=_person(version.created_by),
        deployed_at=_iso(version.deployed_at),
        tags=sorted(t.name for t in version.tags.all()),
        metrics=version.metrics or {},
        config=version.config or {},
        error_message=version.error_message or None,
    )


def _load_model(model_id: str, ctx: RequestContext):
    from ml_models.models import Model

    model = (Model.objects.select_related("project")
             .filter(model_id=model_id, organization=ctx.organization, is_deleted=False).first())
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    require_project_permission(ctx, model.project, "view", "Viewing model versions")
    return model


def _load_version(model, version_id: str):
    version = (model.versions.select_related("dataset_version", "parent_version", "created_by")
               .filter(version_id=version_id, is_deleted=False).first())
    if version is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found in this model")
    return version


def _class_counts(project_image_ids) -> Dict[str, int]:
    """Active objects per class on these project images."""
    from annotations.models import Annotation

    rows = (Annotation.objects
            .filter(project_image_id__in=project_image_ids, is_active=True)
            .values("annotation_class__name")
            .annotate(n=Count("id")))
    return {r["annotation_class__name"] or "unlabelled": r["n"] for r in rows}


def _dataset(ds, counts: Optional[Dict[str, int]] = None) -> Optional[DatasetSummary]:
    if ds is None:
        return None
    from projects.models import VersionImage

    if counts is None:
        counts = _class_counts(VersionImage.objects.filter(version=ds).values("project_image_id"))
    return DatasetSummary(
        id=ds.id,
        version_number=ds.version_number,
        label=ds.version_name or f"v{ds.version_number}",
        total_images=ds.total_images,
        total_annotations=ds.total_annotations,
        train_count=ds.train_count,
        val_count=ds.val_count,
        test_count=ds.test_count,
        created_at=ds.created_at.isoformat(),
        classes=[ClassCount(name=k, count=v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
    )


def _training(version) -> Optional[TrainingSummary]:
    from training.models import TrainingSession

    s = (TrainingSession.objects.select_related("triggered_by")
         .filter(model_version=version).order_by("-created_at").first())
    if s is None:
        return None
    duration = (s.completed_at - s.started_at).total_seconds() if s.started_at and s.completed_at else None
    return TrainingSummary(
        id=s.id,
        status=s.status,
        started_by=_person(s.triggered_by),
        started_at=_iso(s.started_at),
        completed_at=_iso(s.completed_at),
        duration_seconds=duration,
        epochs_run=s.current_epoch,
        total_epochs=s.total_epochs,
        best_metrics=s.best_metrics or {},
        compute_resource=s.compute_resource or None,
    )


def _number(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _direction(key: str) -> Optional[str]:
    k = key.lower()
    if k in NO_VERDICT:
        return None
    if any(word in k for word in LOWER_IS_BETTER):
        return "lower"
    return "higher"


def _metric_changes(a: Dict[str, Any], b: Dict[str, Any]) -> List[MetricChange]:
    keys = [k for k in a if k in b] + [k for k in a if k not in b] + [k for k in b if k not in a]
    out = []
    for key in keys:
        va, vb = _number(a.get(key)), _number(b.get(key))
        if va is None and vb is None:
            continue
        better = _direction(key)
        delta = verdict = None
        if va is not None and vb is not None:
            delta = vb - va
            if better:
                if abs(delta) < SAME_EPSILON:
                    verdict = "same"
                elif (delta > 0) == (better == "higher"):
                    verdict = "improved"
                else:
                    verdict = "worse"
        out.append(MetricChange(key=key, a=va, b=vb, delta=delta, better=better, verdict=verdict))
    # Quality metrics first, bookkeeping (epoch, learning rate) last.
    out.sort(key=lambda m: m.better is None)
    return out


def _settings(a, b):
    ca = dict(a.config or {}, model_size=a.model_size or None)
    cb = dict(b.config or {}, model_size=b.model_size or None)
    changed, same = [], 0
    for key in sorted(set(ca) | set(cb)):
        if ca.get(key) == cb.get(key):
            same += 1
        else:
            changed.append(SettingChange(key=key, a=ca.get(key), b=cb.get(key)))
    return changed, same


def _data_change(a, b) -> DataChange:
    from projects.models import VersionImage

    da, db = a.dataset_version, b.dataset_version
    if da is None or db is None:
        return DataChange(same_dataset=da == db and da is not None, a=_dataset(da), b=_dataset(db))

    split_a = dict(VersionImage.objects.filter(version=da).values_list("project_image_id", "split"))
    split_b = (split_a if da.id == db.id else
               dict(VersionImage.objects.filter(version=db).values_list("project_image_id", "split")))
    ids_a, ids_b = set(split_a), set(split_b)
    shared = ids_a & ids_b
    added = ids_b - ids_a

    counts_a = _class_counts(ids_a)
    counts_b = counts_a if da.id == db.id else _class_counts(ids_b)
    counts_added = _class_counts(added) if added else {}

    classes = [
        ClassChange(name=n, a=counts_a.get(n, 0), b=counts_b.get(n, 0),
                    delta=counts_b.get(n, 0) - counts_a.get(n, 0),
                    in_added_images=counts_added.get(n, 0))
        for n in set(counts_a) | set(counts_b)
    ]
    classes.sort(key=lambda c: (-abs(c.delta), -c.b, c.name))

    return DataChange(
        same_dataset=da.id == db.id,
        a=_dataset(da, counts_a),
        b=_dataset(db, counts_b),
        only_a=len(ids_a - ids_b),
        only_b=len(added),
        shared=len(shared),
        moved_split=sum(1 for i in shared if split_a[i] != split_b[i]),
        classes=classes,
    )


# ---------------------------------------------------------------- routes

@router.get("/{model_id}/versions/{version_id}/lineage", response_model=LineageResponse)
def get_version_lineage(
    model_id: str,
    version_id: str,
    ctx: RequestContext = Depends(get_request_context),
):
    from inferences.models import InferenceDeployment

    model = _load_model(model_id, ctx)
    version = _load_version(model, version_id)
    siblings = model.versions.filter(is_deleted=False)

    previous = siblings.filter(version_number__lt=version.version_number).order_by("-version_number").first()
    following = siblings.filter(version_number__gt=version.version_number).order_by("version_number").first()
    production = siblings.filter(deployment_status="production").order_by("-deployed_at").first()
    parent = version.parent_version if version.parent_version and not version.parent_version.is_deleted else None

    deployments = (InferenceDeployment.objects.filter(model_version=version)
                   .select_related("deployed_by").order_by("-deployed_at")[:50])

    return LineageResponse(
        version=_detail(version),
        parent=_ref(parent) if parent else None,
        children=[_ref(v) for v in version.derived_versions.filter(is_deleted=False).order_by("version_number")],
        previous=_ref(previous) if previous else None,
        next=_ref(following) if following else None,
        production=_ref(production) if production else None,
        dataset=_dataset(version.dataset_version),
        training=_training(version),
        deployments=[
            DeploymentSummary(
                target_env=d.target_env,
                deployed_at=d.deployed_at.isoformat(),
                undeployed_at=_iso(d.undeployed_at),
                is_active=d.is_active,
                deployed_by=_person(d.deployed_by),
            ) for d in deployments
        ],
    )


@router.get("/{model_id}/compare", response_model=CompareResponse)
def compare_versions(
    model_id: str,
    a: str = Query(..., description="Version id of the reference version (usually production)"),
    b: str = Query(..., description="Version id of the candidate version"),
    ctx: RequestContext = Depends(get_request_context),
):
    model = _load_model(model_id, ctx)
    va, vb = _load_version(model, a), _load_version(model, b)
    changed, same = _settings(va, vb)
    production = (model.versions.filter(is_deleted=False, deployment_status="production")
                  .order_by("-deployed_at").first())
    return CompareResponse(
        a=_detail(va),
        b=_detail(vb),
        a_training=_training(va),
        b_training=_training(vb),
        production_version_id=production.version_id if production else None,
        metrics=_metric_changes(va.metrics or {}, vb.metrics or {}),
        settings_changed=changed,
        settings_identical=same,
        data=_data_change(va, vb),
    )

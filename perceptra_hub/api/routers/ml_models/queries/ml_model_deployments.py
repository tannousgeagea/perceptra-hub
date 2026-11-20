"""
A model's deployment history: which version ran where, from when to when,
who put it there, and how it ended.

    GET /api/v1/models/{model_id}/deployments

Read from inference_deployment, which every deploy and undeploy already
writes (inferences/orchestrator.py). Anyone who can view the model's project.
"""
from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import RequestContext, get_request_context
from projects.dependencies import require_project_permission

router = APIRouter(prefix="/models")

# A deployment that ended within this long of another one starting was
# ended by it (promotion, or a newer version taking production).
SAME_MOMENT = timedelta(seconds=5)


class DeploymentEntry(BaseModel):
    id: int
    version_id: str
    version_number: int
    version_name: str
    target_env: str
    deployed_by: Optional[str] = None
    deployed_at: str
    undeployed_at: Optional[str] = None
    is_active: bool
    # "moved_to_production" | "replaced" | "undeployed" | None while active
    ended_by: Optional[str] = None
    replaced_by_version: Optional[str] = None


def _name(user) -> Optional[str]:
    if user is None:
        return None
    return f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username


@router.get("/{model_id}/deployments", response_model=List[DeploymentEntry])
def list_model_deployments(
    model_id: str,
    limit: int = Query(50, ge=1, le=200),
    ctx: RequestContext = Depends(get_request_context),
):
    from inferences.models import InferenceDeployment
    from ml_models.models import Model

    model = (Model.objects.select_related("project")
             .filter(model_id=model_id, organization=ctx.organization, is_deleted=False).first())
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    require_project_permission(ctx, model.project, "view", "Viewing deployments")

    rows = list(
        InferenceDeployment.objects.filter(model_version__model=model)
        .select_related("model_version", "deployed_by")
        .order_by("-deployed_at")[:limit]
    )

    def label(version) -> str:
        return version.version_name or f"v{version.version_number}"

    entries = []
    for d in rows:
        ended_by = replaced_by = None
        if d.undeployed_at:
            # What started at the moment this one ended?
            successor = next(
                (o for o in rows if o.id != d.id and abs(o.deployed_at - d.undeployed_at) <= SAME_MOMENT),
                None,
            )
            if successor is not None and successor.model_version_id == d.model_version_id:
                ended_by = f"moved_to_{successor.target_env}"
            elif successor is not None and successor.target_env == d.target_env:
                ended_by, replaced_by = "replaced", label(successor.model_version)
            else:
                ended_by = "undeployed"
        v = d.model_version
        entries.append(DeploymentEntry(
            id=d.id,
            version_id=v.version_id,
            version_number=v.version_number,
            version_name=label(v),
            target_env=d.target_env,
            deployed_by=_name(d.deployed_by),
            deployed_at=d.deployed_at.isoformat(),
            undeployed_at=d.undeployed_at.isoformat() if d.undeployed_at else None,
            is_active=d.is_active,
            ended_by=ended_by,
            replaced_by_version=replaced_by,
        ))
    return entries

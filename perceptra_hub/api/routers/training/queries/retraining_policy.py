"""
Retraining policy management API.

POST  /api/v1/models/{model_id}/retraining-policies
GET   /api/v1/models/{model_id}/retraining-policies
PATCH /api/v1/models/{model_id}/retraining-policies/{policy_id}
DELETE /api/v1/models/{model_id}/retraining-policies/{policy_id}
POST  /api/v1/models/{model_id}/retraining-policies/{policy_id}/trigger
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from ml_models.models import Model
from training.models import RetrainingPolicy

router = APIRouter()


class RetrainingPolicyCreate(BaseModel):
    trigger_type: str = "annotation_count"
    min_new_annotations: int = Field(default=100, ge=1)
    min_correction_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_days_since_training: Optional[int] = Field(default=None, ge=1)
    lookback_days: int = Field(default=30, ge=1)
    auto_create_dataset_version: bool = True
    auto_submit_training: bool = True
    compute_profile_id: Optional[str] = None
    min_hours_between_runs: int = Field(default=24, ge=1)


class RetrainingPolicyUpdate(BaseModel):
    trigger_type: Optional[str] = None
    min_new_annotations: Optional[int] = Field(default=None, ge=1)
    min_correction_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_days_since_training: Optional[int] = Field(default=None, ge=1)
    lookback_days: Optional[int] = Field(default=None, ge=1)
    auto_create_dataset_version: Optional[bool] = None
    auto_submit_training: Optional[bool] = None
    compute_profile_id: Optional[str] = None
    min_hours_between_runs: Optional[int] = Field(default=None, ge=1)
    is_active: Optional[bool] = None


def _model_or_404(model_id: str):
    try:
        return Model.objects.get(model_id=model_id)
    except Model.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")


def _policy_to_dict(p: RetrainingPolicy) -> dict:
    return {
        "policy_id": p.policy_id,
        "model_id": str(p.model.model_id),
        "is_active": p.is_active,
        "trigger_type": p.trigger_type,
        "min_new_annotations": p.min_new_annotations,
        "min_correction_rate": p.min_correction_rate,
        "max_days_since_training": p.max_days_since_training,
        "lookback_days": p.lookback_days,
        "auto_create_dataset_version": p.auto_create_dataset_version,
        "auto_submit_training": p.auto_submit_training,
        "compute_profile_id": p.compute_profile.profile_id if p.compute_profile else None,
        "min_hours_between_runs": p.min_hours_between_runs,
        "last_triggered_at": p.last_triggered_at.isoformat() if p.last_triggered_at else None,
        "created_at": p.created_at.isoformat(),
    }


@router.post("/models/{model_id}/retraining-policies", status_code=status.HTTP_201_CREATED)
def create_policy(model_id: str, body: RetrainingPolicyCreate, user=Depends(get_current_user)):
    model = _model_or_404(model_id)

    compute_profile = None
    if body.compute_profile_id:
        from compute.models import ComputeProfile
        try:
            compute_profile = ComputeProfile.objects.get(
                profile_id=body.compute_profile_id,
                organization=model.organization,
            )
        except ComputeProfile.DoesNotExist:
            raise HTTPException(status_code=404, detail="Compute profile not found")

    policy = RetrainingPolicy.objects.create(
        model=model,
        trigger_type=body.trigger_type,
        min_new_annotations=body.min_new_annotations,
        min_correction_rate=body.min_correction_rate,
        max_days_since_training=body.max_days_since_training,
        lookback_days=body.lookback_days,
        auto_create_dataset_version=body.auto_create_dataset_version,
        auto_submit_training=body.auto_submit_training,
        compute_profile=compute_profile,
        min_hours_between_runs=body.min_hours_between_runs,
        created_by=user,
    )
    return _policy_to_dict(policy)


@router.get("/models/{model_id}/retraining-policies")
def list_policies(model_id: str, user=Depends(get_current_user)):
    model = _model_or_404(model_id)
    policies = RetrainingPolicy.objects.filter(model=model).order_by("-created_at")
    return {"count": policies.count(), "results": [_policy_to_dict(p) for p in policies]}


@router.patch("/models/{model_id}/retraining-policies/{policy_id}")
def update_policy(
    model_id: str, policy_id: str, body: RetrainingPolicyUpdate, user=Depends(get_current_user)
):
    model = _model_or_404(model_id)
    try:
        policy = RetrainingPolicy.objects.get(policy_id=policy_id, model=model)
    except RetrainingPolicy.DoesNotExist:
        raise HTTPException(status_code=404, detail="Policy not found")

    update_data = body.dict(exclude_none=True)
    compute_profile_id = update_data.pop("compute_profile_id", None)

    if compute_profile_id is not None:
        from compute.models import ComputeProfile
        try:
            policy.compute_profile = ComputeProfile.objects.get(
                profile_id=compute_profile_id, organization=model.organization
            )
        except ComputeProfile.DoesNotExist:
            raise HTTPException(status_code=404, detail="Compute profile not found")

    for field, value in update_data.items():
        setattr(policy, field, value)
    policy.save()

    return _policy_to_dict(policy)


@router.delete("/models/{model_id}/retraining-policies/{policy_id}", status_code=204)
def delete_policy(model_id: str, policy_id: str, user=Depends(get_current_user)):
    model = _model_or_404(model_id)
    deleted, _ = RetrainingPolicy.objects.filter(policy_id=policy_id, model=model).delete()
    if not deleted:
        raise HTTPException(status_code=404, detail="Policy not found")


@router.post("/models/{model_id}/retraining-policies/{policy_id}/trigger")
def manually_trigger_policy(model_id: str, policy_id: str, user=Depends(get_current_user)):
    """Force-fire a retraining policy regardless of its threshold conditions."""
    model = _model_or_404(model_id)
    try:
        policy = RetrainingPolicy.objects.get(policy_id=policy_id, model=model)
    except RetrainingPolicy.DoesNotExist:
        raise HTTPException(status_code=404, detail="Policy not found")

    from training.retraining_service import RetrainingService

    try:
        result = RetrainingService()._trigger_retraining(policy, reason="manual_trigger")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result

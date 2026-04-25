"""
Champion/challenger evaluation API.

GET  /api/v1/model-evaluations/{evaluation_id}       → evaluation status + metrics
GET  /api/v1/models/{model_id}/evaluations           → evaluation history for a model
POST /api/v1/model-evaluations/{evaluation_id}/promote → manually promote challenger
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from api.dependencies import get_current_user
from inferences.models import ModelEvaluation
from ml_models.models import Model

router = APIRouter()


class EvaluationSummary(BaseModel):
    evaluation_id: str
    status: str
    challenger_version_id: Optional[str]
    champion_version_id: Optional[str]
    challenger_metrics: dict
    champion_metrics: dict
    improvement_delta: Optional[float]
    recommendation: Optional[str]
    auto_promoted: bool
    created_at: str
    completed_at: Optional[str]


def _to_summary(ev: ModelEvaluation) -> EvaluationSummary:
    return EvaluationSummary(
        evaluation_id=ev.evaluation_id,
        status=ev.status,
        challenger_version_id=str(ev.challenger.version_id) if ev.challenger_id else None,
        champion_version_id=str(ev.champion.version_id) if ev.champion_id else None,
        challenger_metrics=ev.challenger_metrics or {},
        champion_metrics=ev.champion_metrics or {},
        improvement_delta=ev.improvement_delta,
        recommendation=ev.recommendation,
        auto_promoted=ev.auto_promoted,
        created_at=ev.created_at.isoformat(),
        completed_at=ev.completed_at.isoformat() if ev.completed_at else None,
    )


@router.get("/model-evaluations/{evaluation_id}", response_model=EvaluationSummary)
def get_evaluation(evaluation_id: str, user=Depends(get_current_user)):
    try:
        ev = ModelEvaluation.objects.select_related("challenger", "champion").get(
            evaluation_id=evaluation_id
        )
    except ModelEvaluation.DoesNotExist:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _to_summary(ev)


@router.get("/models/{model_id}/evaluations")
def list_model_evaluations(
    model_id: str,
    limit: int = 20,
    offset: int = 0,
    user=Depends(get_current_user),
):
    try:
        model = Model.objects.get(model_id=model_id)
    except Model.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")

    qs = (
        ModelEvaluation.objects.filter(challenger__model=model)
        .select_related("challenger", "champion")
        .order_by("-created_at")[offset: offset + limit]
    )

    return {
        "count": ModelEvaluation.objects.filter(challenger__model=model).count(),
        "results": [_to_summary(ev) for ev in qs],
    }


@router.post(
    "/model-evaluations/{evaluation_id}/promote",
    status_code=status.HTTP_200_OK,
)
def manually_promote_challenger(evaluation_id: str, user=Depends(get_current_user)):
    """Human-override: promote the challenger regardless of automatic recommendation."""
    try:
        ev = ModelEvaluation.objects.select_related("challenger__model__organization").get(
            evaluation_id=evaluation_id
        )
    except ModelEvaluation.DoesNotExist:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    if ev.challenger.status != "trained":
        raise HTTPException(
            status_code=422,
            detail=f"Challenger status is '{ev.challenger.status}', must be 'trained' to promote.",
        )

    from inferences.orchestrator import InferenceOrchestrator

    try:
        orchestrator = InferenceOrchestrator(ev.challenger)
        orchestrator.deploy(target_env="production", user=user)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))

    ev.recommendation = "promote"
    ev.auto_promoted = False   # manual, not auto
    ev.save(update_fields=["recommendation", "auto_promoted"])

    return {
        "evaluation_id": evaluation_id,
        "challenger_version_id": str(ev.challenger.version_id),
        "deployment_status": ev.challenger.deployment_status,
    }

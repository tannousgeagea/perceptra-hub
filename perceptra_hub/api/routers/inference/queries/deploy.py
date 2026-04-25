"""
Deploy / undeploy a trained ModelVersion to the perceptra-inference service.

POST /api/v1/model-versions/{version_id}/deploy   → deploy to staging or production
POST /api/v1/model-versions/{version_id}/undeploy → retire from inference service
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from api.dependencies import get_current_user
from inferences.orchestrator import InferenceOrchestrator
from ml_models.models import ModelVersion

router = APIRouter()


class DeployRequest(BaseModel):
    target_env: str = "production"   # "staging" | "production"
    class_names: Optional[list[str]] = None


class DeployResponse(BaseModel):
    version_id: str
    deployment_status: str
    target_env: str
    deployment_id: int


@router.post(
    "/model-versions/{version_id}/deploy",
    response_model=DeployResponse,
    status_code=status.HTTP_200_OK,
)
def deploy_model_version(
    version_id: str,
    body: DeployRequest,
    user=Depends(get_current_user),
):
    try:
        model_version = ModelVersion.objects.select_related(
            "model__organization", "model__task", "storage_profile"
        ).get(version_id=version_id)
    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model version not found")

    try:
        orchestrator = InferenceOrchestrator(model_version)
        deployment = orchestrator.deploy(
            target_env=body.target_env,
            user=user,
            class_names=body.class_names,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return DeployResponse(
        version_id=version_id,
        deployment_status=model_version.deployment_status,
        target_env=body.target_env,
        deployment_id=deployment.id,
    )


@router.post(
    "/model-versions/{version_id}/undeploy",
    status_code=status.HTTP_200_OK,
)
def undeploy_model_version(
    version_id: str,
    user=Depends(get_current_user),
):
    try:
        model_version = ModelVersion.objects.select_related(
            "model__organization"
        ).get(version_id=version_id)
    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model version not found")

    try:
        orchestrator = InferenceOrchestrator(model_version)
        orchestrator.undeploy(user=user)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"version_id": version_id, "deployment_status": model_version.deployment_status}

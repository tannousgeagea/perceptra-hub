import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List
from pydantic import BaseModel
from ml_models.models import Model
from projects.models import Version as DatasetVersion

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

class ArtifactOut(BaseModel):
    onnx: Optional[str] = None
    weights: Optional[str] = None
    logs: Optional[str] = None

class DatasetUsedOut(BaseModel):
    id: str
    name: str
    itemCount: int
    createdAt: datetime

class ModelVersionOut(BaseModel):
    id: str
    versionNumber: int
    createdAt: datetime
    createdBy: str
    status: str
    metrics: dict
    tags: List[str]
    datasetUsed: Optional[DatasetUsedOut]
    artifacts: ArtifactOut

class ModelDetailOut(BaseModel):
    id: str
    name: str
    description: str
    type: str
    createdAt: datetime
    updatedAt: datetime
    createdBy: str
    projectId: str
    versions: List[ModelVersionOut]
    currentProductionVersion: Optional[str]
    tags: List[str]

@router.get("/models/{model_id}", response_model=ModelDetailOut)
def get_model_by_id(model_id: str):
    try:
        model = Model.objects.prefetch_related(
            "versions__dataset_version__project",
            "tags",
            "versions__created_by"
        ).select_related("task", "framework", "project").get(id=model_id)

        versions = []
        for version in model.versions.all().order_by("-created_at"):
            dataset = version.dataset_version
            dataset_out = None
            if dataset:
                dataset_out = {
                    "id": str(dataset.id),
                    "name": dataset.project.name,
                    "itemCount": dataset.project.project_images.count(),
                    "createdAt": dataset.created_at
                }

            versions.append({
                "id": f"{version.id}",
                "versionNumber": int(version.version),
                "createdAt": version.created_at,
                "createdBy": version.created_by.email if version.created_by else "unknown",
                "status": version.status,
                "metrics": version.metrics or {},
                "tags": version.config.get("tags", []) if version.config else [],
                "datasetUsed": dataset_out,
                "artifacts": {
                    "onnx": version.checkpoint.url if version.checkpoint else None,
                    "weights": version.checkpoint.url if version.checkpoint else None,
                    "logs": version.logs.url if version.logs else None,
                }
            })

        return {
            "id": str(model.id),
            "name": model.name,
            "description": model.description or "",
            "type": model.task.name,
            "createdAt": model.created_at,
            "updatedAt": model.updated_at,
            "createdBy": model.versions.first().created_by.email if model.versions.exists() else "unknown",
            "projectId": str(model.project.id),
            "versions": versions,
            "currentProductionVersion": None,  # Implement this logic if needed
            "tags": [tag.name for tag in model.tags.all()]
        }

    except Model.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Dict, Optional
from pydantic import BaseModel
from ml_models.models import ModelVersion

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

class ModelVersionOut(BaseModel):
    id: int
    version_number: int
    model_id: int
    model_name: str
    status: str
    metrics: Optional[Dict] = {}
    config: Optional[Dict] = {}
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_url: Optional[str] = None
    artifacts: Optional[Dict] = {}

    class Config:
        from_attributes = True

@router.get("/model-versions/{version_id}", response_model=ModelVersionOut)
def get_model_version(version_id: int):
    try:
        version = ModelVersion.objects.select_related("model", "dataset_version").get(id=version_id)

        return {
            "id": version.id,
            "version_number": version.version,
            "model_id": version.model.id,
            "model_name": version.model.name,
            "status": version.status,
            "metrics": version.metrics or {},
            "config": version.config or {},
            "dataset_id": version.dataset_version.id if version.dataset_version else None,
            "dataset_name": version.dataset_version.version_name if version.dataset_version else None,
            "dataset_url": version.dataset_version.version_file.url if version.dataset_version else None,
            "artifacts": {
                "weights": version.checkpoint.url if version.checkpoint else None,
                "logs": version.logs.url if version.logs else None
            }
        }

    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model version not found")

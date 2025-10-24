import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Literal, Dict
from pydantic import BaseModel
from ml_models.models import ModelVersion
from projects.models import Project
from django.core.files.base import ContentFile

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


class ModelVersionUpdate(BaseModel):
    status: Optional[Literal["training", "trained", "failed"]] = None
    metrics: Optional[Dict[str, float]] = None
    config: Optional[Dict] = None
    error_message: Optional[str] = None


@router.api_route(
    "/model-versions/{version_id}", methods=["PATCH"]
)
def update_model_version(version_id: int, data: ModelVersionUpdate):
    try:
        version = ModelVersion.objects.get(id=version_id)

        if data.status:
            version.status = data.status
        if data.metrics:
            version.metrics = data.metrics
        if data.config:
            version.config = data.config
        if data.error_message:
            version.logs.save("error.log", ContentFile(data.error_message.encode("utf-8")))  # optional

        version.save()
        return {"message": "Model version updated successfully"}

    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model version not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
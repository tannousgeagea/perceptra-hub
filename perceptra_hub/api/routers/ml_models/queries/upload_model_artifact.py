import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Literal, Dict
from ml_models.models import ModelVersion
from django.core.files.base import ContentFile
from fastapi import UploadFile, File, Form

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

@router.api_route(
    "/model-versions/{version_id}/upload-artifact", methods=["POST"]
)
def upload_artifact(
    version_id: int,
    file: UploadFile = File(...),
    type: Literal["checkpoint", "logs"] = Form(...)
):
    try:
        version = ModelVersion.objects.get(id=version_id)
        if type == "checkpoint":
            version.checkpoint.save(file.filename, ContentFile(file.file.read()))
        elif type == "logs":
            version.logs.save(file.filename, ContentFile(file.file.read()))
        else:
            raise HTTPException(400, "Invalid artifact type")

        version.save()
        return {"message": f"{type} uploaded successfully"}

    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model version not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

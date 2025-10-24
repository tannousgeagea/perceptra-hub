# routes/datasets.py
import time
from fastapi.routing import APIRoute
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from typing import List, Callable
from projects.models import Version, Project
from django.db.models import Count
from pydantic import BaseModel
from datetime import datetime

class DatasetOut(BaseModel):
    id: str
    name: str
    itemCount: int
    createdAt: datetime

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

@router.get("/datasets/{project_id}", response_model=List[DatasetOut])
def list_project_datasets(project_id: str):
    try:
        # Check if the project exists
        try:
            project = Project.objects.get(name=project_id)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found.")

        # Get dataset-like versions (linked to ProjectImages)
        dataset_versions = (
            Version.objects
            .filter(project=project)
            .annotate(item_count=Count('version_images'))
        )

        result = [
            DatasetOut(
                id=str(v.id),
                name=v.version_name,
                itemCount=v.item_count,
                createdAt=v.created_at,
            )
            for v in dataset_versions
        ]

        return result

    except HTTPException:
        raise  # re-raise known HTTP errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

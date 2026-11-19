# routes/datasets.py
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import RequestContext, get_request_context
from typing import List
from projects.models import Version, Project
from django.db.models import Count
from pydantic import BaseModel
from datetime import datetime

class DatasetOut(BaseModel):
    id: str
    name: str
    itemCount: int
    createdAt: datetime

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

@router.get("/datasets/{project_id}", response_model=List[DatasetOut])
def list_project_datasets(project_id: str, ctx: RequestContext = Depends(get_request_context)):
    try:
        # Check if the project exists
        try:
            project = Project.objects.get(
                name=project_id, organization=ctx.organization, is_deleted=False
            )
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

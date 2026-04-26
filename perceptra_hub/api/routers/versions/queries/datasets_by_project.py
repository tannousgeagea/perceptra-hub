from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from asgiref.sync import sync_to_async

from api.dependencies import get_project_context, ProjectContext
from projects.models import Version

router = APIRouter(prefix="/projects")


class DatasetResponse(BaseModel):
    id: str
    name: str
    version: Optional[int] = None
    item_count: Optional[int] = None
    created_at: Optional[str] = None


@router.get(
    "/{project_id}/datasets",
    response_model=List[DatasetResponse],
    summary="List exported dataset versions for a project",
)
async def list_project_datasets(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
    only_ready: bool = Query(True, description="Return only fully exported versions"),
):
    @sync_to_async
    def fetch(project, only_ready):
        qs = Version.objects.filter(project=project).order_by("-version_number")
        if only_ready:
            qs = qs.filter(export_status="completed")
        return list(qs)

    versions = await fetch(project_ctx.project, only_ready)

    return [
        DatasetResponse(
            id=str(v.version_id),
            name=v.version_name,
            version=v.version_number,
            item_count=v.total_images,
            created_at=v.created_at.isoformat() if v.created_at else None,
        )
        for v in versions
    ]

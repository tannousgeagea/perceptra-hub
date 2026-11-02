"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from uuid import UUID
import logging
from asgiref.sync import sync_to_async

from api.dependencies import get_request_context, RequestContext
from projects.models import (
    Project
)

from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete Project")
async def delete_project(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
    hard_delete: bool = Query(False, description="Permanently delete project")
):
    """Soft or hard delete a project."""
    ctx.require_role('admin', 'owner')
    
    @sync_to_async
    def delete_project_record(org, project_id, user, hard_delete):
        try:
            project = Project.objects.get(
                project_id=project_id,
                organization=org
            )
        except Project.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        if hard_delete:
            project.delete()
        else:
            project.soft_delete(user=user)
    
    await delete_project_record(ctx.organization, project_id, ctx.user, hard_delete)



@router.post("/{project_id}/restore", summary="Restore a Deleted Project")
async def restore_project(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
):
    """Undo a soft delete. Same permission as deleting."""
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def restore(org, project_id):
        project = Project.objects.filter(project_id=project_id, organization=org).first()
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        if project.is_deleted:
            project.restore()
        return project

    project = await restore(ctx.organization, project_id)
    return {"project_id": str(project.project_id), "is_deleted": project.is_deleted}

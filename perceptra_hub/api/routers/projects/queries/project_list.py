"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
import logging
from asgiref.sync import sync_to_async

from api.dependencies import get_request_context, RequestContext
from projects.models import (
    Project
)
from django.contrib.auth import get_user_model
from api.routers.projects.schemas import ProjectResponse, UserBasicInfo, ProjectStatistics, ProjectListItem


User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("", summary="List Projects")
async def list_projects(
    ctx: RequestContext = Depends(get_request_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: Optional[bool] = Query(None),
    project_type_id: Optional[int] = Query(None)
):
    """List all projects for organization."""
    
    @sync_to_async
    def get_projects(org, skip, limit, is_active, project_type_id, user):
        queryset = Project.objects.filter(
            organization=org,
            is_deleted=False
        ).select_related(
            'project_type',
            'visibility',
            'created_by',
            'updated_by',
            'organization'
        ).prefetch_related(
            'project_images',
            'annotation_groups',
            'memberships__role'
        ).distinct()
            
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)
        
        if project_type_id:
            queryset = queryset.filter(project_type_id=project_type_id)
        
        total = queryset.count()
        projects = list(queryset.select_related('project_type', 'visibility')[skip:skip + limit])
        
        results = []
        for project in projects:
            # Get user's role in this project
            membership = project.memberships.filter(user=user).first()
            user_role = membership.role.name if membership else 'member'
            
            # Get statistics
            image_count = project.project_images.count()
            annotation_group_count = project.annotation_groups.count()
            
            thumbnail = project.thumbnail_url
            if not thumbnail:
                first_image = project.project_images.first()
                if first_image and first_image.image:
                    try:
                        thumbnail = first_image.image.get_download_url(expiration=3600)
                        project.thumbnail_url = thumbnail
                        project.save(updated_field=['thumbnail_url'])
                    except Exception as e:
                        logger.warning(f"Failed to get thumbnail for project {project.id}: {e}")
                        thumbnail = None
            
            # Build creator info
            created_by = None
            if project.created_by:
                created_by = UserBasicInfo(
                    id=project.created_by.id,
                    username=project.created_by.username,
                    email=project.created_by.email,
                    first_name=project.created_by.first_name,
                    last_name=project.created_by.last_name
                )
            
            # Build updater info
            updated_by = None
            if project.updated_by:
                updated_by = UserBasicInfo(
                    id=project.updated_by.id,
                    username=project.updated_by.username,
                    email=project.updated_by.email,
                    first_name=project.updated_by.first_name,
                    last_name=project.updated_by.last_name
                )
                
            results.append(ProjectListItem(
                id=project.id,
                project_id=str(project.project_id),
                name=project.name,
                description=project.description,
                thumbnail_url=thumbnail,
                project_type_name=project.project_type.name,
                visibility_name=project.visibility.name,
                is_active=project.is_active,
                statistics=ProjectStatistics(
                    total_images=image_count,
                    total_annotations=0,  # Add when annotations are implemented
                    annotation_groups=annotation_group_count
                ),
                created_by=created_by,
                updated_by=updated_by,
                created_at=project.created_at.isoformat(),
                last_edited=project.last_edited.isoformat(),
                user_role=user_role
            ))
                
        return {"total": total, "projects": results}
    
    result = await get_projects(ctx.organization, skip, limit, is_active, project_type_id, ctx.user)
    
    return result
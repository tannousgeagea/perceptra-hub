"""
FastAPI routes for annotation management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging

from django.utils import timezone
from asgiref.sync import sync_to_async
import uuid

from api.dependencies import get_project_context, ProjectContext
from api.routers.annotations.schemas import AnnotationCreate, AnnotationResponse, AnnotationBatchCreate, AnnotationCreateResponse, AnnotationAuditConfig
from projects.models import ProjectImage
from django.db import transaction
from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationType,
    AnnotationGroup
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")

@router.post(
    "/{project_id}/images/{project_image_id}/annotations",
    response_model=AnnotationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Annotation"
)
async def create_annotation(
    project_id: UUID,
    project_image_id: int,
    data: AnnotationCreate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Create a new annotation for a project image."""
    
    @sync_to_async
    def create_annotation_record(project, project_image_id, data, user):
        # Get project image
        try:
            project_image = ProjectImage.objects.get(
                id=project_image_id,
                project=project,
                is_active=True
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )
        
        # Validate annotation type
        try:
            annotation_type = AnnotationType.objects.get(name=data.annotation_type)
        except AnnotationType.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Annotation type not found"
            )
        
        # Validate annotation class belongs to project
        try:
            annotation_class = None
            if data.annotation_class_id:
                annotation_class = AnnotationClass.objects.select_related('annotation_group').get(
                    class_id=data.annotation_class_id,
                    annotation_group__project=project
                )
            elif data.annotation_class_name:
                annotation_class = AnnotationClass.objects.select_related('annotation_group').get(
                    name=data.annotation_class_name,
                    annotation_group__project=project
                )
            elif data.annotation_id:
                annotation_class = AnnotationClass.objects.select_related('annotation_group').get(
                    id=data.annotation_id,
                    annotation_group__project=project
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="either annotation_class_id or annotation_class_name must be given"
                )
        
        except AnnotationClass.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Annotation class not found or doesn't belong to this project"
            )
        
        # Validate data format
        if len(data.data) != 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data must be [xmin, ymin, xmax, ymax]"
            )
        
        with transaction.atomic():
            annotation = None
            if data.annotation_uid:
                annotation = Annotation.objects.filter(
                    annotation_uid=data.annotation_uid,
                    project_image=project_image
                ).first()
                
            if annotation:
                # UPDATE existing
                annotation.annotation_type = annotation_type
                annotation.annotation_class = annotation_class
                annotation.data = data.data
                annotation.updated_by = user
                annotation.save()
                created = False
            else:   
                # Create annotation
                annotation = Annotation(
                    project_image=project_image,
                    annotation_type=annotation_type,
                    annotation_class=annotation_class,
                    data=data.data,
                    annotation_uid=data.annotation_uid or str(uuid.uuid4()),
                    annotation_source=data.annotation_source,
                    confidence=data.confidence,
                    created_by=user,
                    updated_by=user
                )
                created = True
                
                if data.annotation_time_seconds:
                    annotation._duration_seconds = data.annotation_time_seconds
                
                annotation.save()
                
            # Update project image status
            if not project_image.annotated:
                project_image.annotated = True
                project_image.status = 'annotated'
                project_image.save(update_fields=['annotated', 'status'])
        
        return {
            "message": "Annotation created successfully" if created else "Annotation update successffully",
            "annotation": AnnotationResponse(
                id=str(annotation.id),
                annotation_uid=annotation.annotation_uid,
                type=annotation.annotation_type.name,
                class_id=annotation.annotation_class.class_id,
                class_name=annotation.annotation_class.name,
                color=annotation.annotation_class.color,
                data=annotation.data,
                source=annotation.annotation_source,
                confidence=annotation.confidence,
                reviewed=annotation.reviewed,
                is_active=annotation.is_active,
                created_at=annotation.created_at.isoformat(),
                created_by=annotation.created_by.username if annotation.created_by else None
            )
        }
    
    return await create_annotation_record(
        project_ctx.project,
        project_image_id,
        data,
        project_ctx.user
    )

@router.post(
    "/{project_id}/annotations/batch",
    status_code=status.HTTP_201_CREATED,
    summary="Batch Create Annotations"
)
async def batch_create_annotations(
    project_id: UUID,
    data: AnnotationBatchCreate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Create multiple annotations for a project image at once."""
    
    results = []
    for ann_data in data.annotations:
        result = await create_annotation(
            project_id=project_id,
            data=ann_data,
            project_ctx=project_ctx
        )
        results.append(result)
    
    return {
        "message": f"Created {len(results)} annotations",
        "count": len(results),
        "annotations": results,
    }
        
        
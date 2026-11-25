
import os
import django
django.setup()
from fastapi import APIRouter, Depends, Query
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from typing import List, Optional
import datetime
from fastapi import status as http_status
from asgiref.sync import sync_to_async
from api.dependencies import get_project_context, ProjectContext

# Import your Django models.
from annotations.models import (
    Annotation,
    AnnotationClass, 
    AnnotationGroup,
)

from projects.models import Project, ProjectImage
from django.db import transaction
from django.utils import timezone
from typing import Literal

router = APIRouter(
    prefix="/projects",
)


def _refresh_flags(project_image_ids) -> None:
    """Images left with no annotation become unannotated (status too, unless reviewed or finalized)."""
    if not project_image_ids:
        return
    still = set(Annotation.objects.filter(project_image_id__in=project_image_ids, is_active=True, is_deleted=False)
                .values_list("project_image_id", flat=True))
    empty = [pk for pk in project_image_ids if pk not in still]
    if not empty:
        return
    now = timezone.now()
    ProjectImage.objects.filter(pk__in=empty).update(annotated=False, updated_at=now)
    ProjectImage.objects.filter(pk__in=empty, reviewed=False, finalized=False).update(status="unannotated")

@router.delete(
    "/{project_id}/classes/{class_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Annotation Class",
    description=(
        "Delete an annotation class. A class no annotation uses is deleted straight away. "
        "A class in use needs a choice: annotations=move&move_to=<class_id> moves its annotations to "
        "another class of the project first; annotations=delete deletes them with the class. "
        "Without a choice it answers 409 with the number of annotations."
    )
)
async def delete_annotation_class(
    project_id: str,
    class_id: int,
    annotations: Optional[Literal["move", "delete"]] = Query(None, description="What happens to the class's annotations"),
    move_to: Optional[int] = Query(None, description="class_id that receives them (annotations=move)"),
    hard_delete: bool = Query(False, description="Older name for annotations=delete"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Delete an annotation class from a project’s annotation group.
    Needs the editor role or above in the project.
    """
    project_ctx.require_edit_permission()
    if hard_delete and annotations is None:
        annotations = "delete"
    if annotations == "move" and move_to is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Say which class receives the annotations (move_to).")

    @sync_to_async
    def delete_class_record(project, class_id: int, choice: Optional[str], move_to: Optional[int]):
        # Ensure annotation group exists for this project
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        if not annotation_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project.project_id} does not have an annotation group."
            )

        # Locate class by class_id
        annotation_class = AnnotationClass.objects.filter(
            class_id=class_id,
            annotation_group=annotation_group
        ).first()

        if not annotation_class:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Annotation class {class_id} not found in project {project.project_id}."
            )

        # Every annotation row on the class (removed ones too: they'd go with it).
        rows = Annotation.objects.filter(annotation_class=annotation_class)
        in_use = rows.filter(is_active=True, is_deleted=False).count()
        name = annotation_class.name

        # Only annotations still in use need a choice; removed ones just go with the class.
        if in_use > 0 and choice is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"'{name}' is used by {in_use} annotation{'' if in_use == 1 else 's'}. "
                       "Move them to another class or delete them with it.",
            )

        with transaction.atomic():
            moved = deleted = 0
            if choice == "move":
                target = AnnotationClass.objects.filter(
                    annotation_group__project=project, class_id=move_to
                ).exclude(pk=annotation_class.pk).first()
                if target is None:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                        detail=f"No other class {move_to} in this project to move them to.")
                # A relabel, not an annotator's correction: plain update, no edit history.
                rows.update(annotation_class=target, updated_at=timezone.now())
                moved = in_use
                message = f"Deleted '{name}'; moved {in_use} annotation{'' if in_use == 1 else 's'} to '{target.name}'."
            else:
                touched = list(rows.values_list("project_image_id", flat=True).distinct())
                deleted = in_use
                rows.delete()
                _refresh_flags(touched)
                message = f"Deleted '{name}' and its {in_use} annotation{'' if in_use == 1 else 's'}."
            annotation_class.delete()
        return {"success": True, "message": message, "moved": moved, "deleted": deleted}

    try:
        return await delete_class_record(project_ctx.project, class_id, annotations, move_to)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

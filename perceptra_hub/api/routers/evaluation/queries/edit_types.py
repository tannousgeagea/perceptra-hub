"""
How reviewers changed the model's predictions.

    GET /api/v1/evaluation/projects/{project_id}/edit-types

Counts the project's prediction annotations that a reviewer has handled
(reviewed, or deleted as wrong) by how they were changed: kept as is, minor
or major box change, class changed, deleted. Annotation.edit_type is set
when a prediction is edited (annotations/models.py).
"""
from typing import List
from uuid import UUID

from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import ProjectContext, get_project_context

router = APIRouter(prefix="/evaluation")


class EditTypeCount(BaseModel):
    type: str      # label, e.g. "Class Changed"
    key: str       # edit_type value, e.g. "class_change"
    count: int
    percentage: float


@router.get("/projects/{project_id}/edit-types", response_model=List[EditTypeCount])
async def prediction_edit_types(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    from annotations.models import Annotation

    @sync_to_async
    def _counts():
        rows = dict(
            Annotation.objects.filter(
                project_image__project=project_ctx.project,
                annotation_source="prediction",
            )
            .filter(Q(reviewed=True) | Q(is_deleted=True, edit_type=Annotation.EditType.DELETED))
            .values_list("edit_type")
            .annotate(n=Count("id"))
            .values_list("edit_type", "n")
        )
        total = sum(rows.values())
        return [
            EditTypeCount(
                type=label,
                key=value,
                count=rows.get(value, 0),
                percentage=round(rows.get(value, 0) * 100 / total, 1) if total else 0.0,
            )
            for value, label in Annotation.EditType.choices
            if rows.get(value)
        ]

    return await _counts()

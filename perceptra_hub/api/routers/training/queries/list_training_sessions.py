# routes/training_sessions.py

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from training.models import TrainingSession
from django.utils.timezone import localtime
from django.db.models import Q

router = APIRouter()

class TrainingSessionOut(BaseModel):
    id: str
    modelName: str
    projectName: str
    status: str
    createdAt: str
    updatedAt: str
    progress: float
    metrics: Optional[dict]
    configuration: Optional[dict]
    logs: Optional[List[str]]

class TrainingSessionList(BaseModel):
    total: int
    results: List[TrainingSessionOut]

@router.get("/training-sessions", response_model=TrainingSessionList)
def list_training_sessions(
    project_id: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(10, ge=1),
    offset: int = Query(0, ge=0)
):
    try:
        queryset = TrainingSession.objects.select_related(
            "model_version__model", "model_version__model__project"
        )

        if project_id:
            queryset = queryset.filter(model_version__model__project__name__icontains=project_id)

        if model_id:
            queryset = queryset.filter(model_version__model__name__icontains=model_id)

        if search:
            queryset = queryset.filter(
                Q(model_version__model__name__icontains=search) |
                Q(model_version__model__project__name__icontains=search)
            )

        total = queryset.count()
        queryset = queryset.order_by("-created_at")[offset:offset + limit]

        results = []
        for session in queryset:
            mv = session.model_version
            model = mv.model
            project = model.project

            if isinstance(session.metrics, list):
                metrics = session.metrics[-1]
            else:
                metrics = session.metrics
            
            results.append(TrainingSessionOut(
                id=str(session.id),
                modelName=model.name,
                projectName=project.name,
                status=session.status,
                createdAt=localtime(session.created_at).isoformat(),
                updatedAt=localtime(session.updated_at).isoformat(),
                progress=session.progress,
                metrics=metrics,
                configuration=session.config,
                logs=session.logs.splitlines() if session.logs else []
            ))

        return {"total": total, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

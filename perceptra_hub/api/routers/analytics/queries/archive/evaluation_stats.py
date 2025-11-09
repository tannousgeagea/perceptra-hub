# app/api/routes/evaluation.py

import time
from fastapi.routing import APIRoute
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Callable
from django.db.models import Count, Q
from annotations.models import AnnotationAudit, Annotation
from projects.models import Project

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
)

class ConfusionEntry(BaseModel):
    class_name: str
    TP: int
    FP: int
    FN: int

class EvaluationStats(BaseModel):
    total: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    mean_average_precision: float
    confusion_matrix: list[ConfusionEntry]

@router.api_route("/analytics/evaluationstats", methods=["GET"], response_model=EvaluationStats)
def get_evaluation_stats(project_id: str):
    try:
        project = Project.objects.get(name=project_id)
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")

    audits = AnnotationAudit.objects.filter(annotation__project_image__project=project)

    tp = audits.filter(evaluation_status="TP").count()
    fp = audits.filter(evaluation_status="FP").count()
    fn = audits.filter(evaluation_status="FN").count()
    total = tp + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Mean Average Precision (mock for now)
    mean_ap = 0.75

    # Confusion matrix per class
    confusion = []
    classes = Annotation.objects.filter(project_image__project=project).values("annotation_class__name").distinct()
    for entry in classes:
        class_name = entry["annotation_class__name"]
        confusion.append(ConfusionEntry(
            class_name=class_name,
            TP=audits.filter(evaluation_status="TP", annotation__annotation_class__name=class_name).count(),
            FP=audits.filter(evaluation_status="FP", annotation__annotation_class__name=class_name).count(),
            FN=audits.filter(evaluation_status="FN", annotation__annotation_class__name=class_name).count(),
        ))

    return EvaluationStats(
        total=total,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_average_precision=mean_ap,
        confusion_matrix=confusion
    )

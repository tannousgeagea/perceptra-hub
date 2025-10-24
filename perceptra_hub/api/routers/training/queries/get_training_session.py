# routes/models.py
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Dict
from pydantic import BaseModel
from django.utils.timezone import localtime
from training.models import TrainingSession

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
    metricsData: Optional[List[Dict]]
    model_version: Optional[Dict]

@router.get("/training-sessions/{session_id}")
def get_training_session(session_id: int) -> TrainingSessionOut:
    try:
        session = TrainingSession.objects.select_related("model_version").get(id=session_id)
        mv = session.model_version
        model = mv.model
        project = model.project

        if isinstance(session.metrics, list):
            metrics = session.metrics[-1]
        else:
            metrics = session.metrics
            
        return TrainingSessionOut(
                id=str(session.id),
                modelName=model.name,
                projectName=project.name,
                status=session.status,
                createdAt=localtime(session.created_at).isoformat(),
                updatedAt=localtime(session.updated_at).isoformat(),
                progress=session.progress,
                metrics=metrics,
                configuration=session.config,
                logs=session.logs.splitlines() if session.logs else [],
                model_version={
                    "id": mv.id,
                    "version": mv.version,
                },
                metricsData=[
                    {
                        "epoch": i + 1,
                        **met,
                    } for i, met in enumerate(session.metrics)
                ],
            )
    except TrainingSession.DoesNotExist:
        raise HTTPException(404, "Session not found")

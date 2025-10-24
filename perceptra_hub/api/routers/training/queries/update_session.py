

# routes/models.py
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Dict
from pydantic import BaseModel
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


class TrainingUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = None
    logs: Optional[str] = None
    log_path: Optional[str] = None
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None

@router.patch("/training-sessions/{session_id}")
def update_session(session_id: int, data: TrainingUpdate):
    try:
        session = TrainingSession.objects.get(id=session_id)

        if data.status:
            session.status = data.status
        if data.progress is not None:
            session.progress = data.progress
        if data.log_path:
            session.log_path = data.log_path
        if data.logs:
            if session.logs is None:
                session.logs = ""
            session.logs += data.logs + "\n"
        if data.metrics:
            if session.metrics is None:
                session.metrics = []
            session.metrics.append(data.metrics)
        if data.error_message:
            session.error_message = data.error_message
            session.status = "failed"

        if data.status in ["completed", "failed"]:
            session.completed_at = datetime.now()

        session.save()
        return {"message": "Training session updated"}

    except TrainingSession.DoesNotExist:
        raise HTTPException(404, "Session not found")

# routes/models.py
import os
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List
from pydantic import BaseModel
from training.models import TrainingSession
from fastapi.responses import StreamingResponse

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

@router.get("/training-sessions/{session_id}/logs/stream")
def stream_logs(session_id: int):
    log_path = f"/media/session_{session_id}.log"

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    def iterfile():
        with open(log_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                yield line

    return StreamingResponse(iterfile(), media_type="text/plain")
import os
import json
import time
import uuid
import importlib
import django
django.setup()
from pathlib import Path
from fastapi import Body
from fastapi import Request
from datetime import datetime
from pydantic import BaseModel
from django.core.cache import cache
from fastapi import HTTPException, status
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi import FastAPI, Depends, APIRouter, Request, Header, Response
from typing import Callable, Union, Any, Dict, AnyStr, Optional, List
from common_utils.progress.core import get_progress

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


@router.api_route(
    "/progress/{task_id}", methods=["GET"], tags=["Progress"]
)
def get_task_progress(task_id: str):
    progress = get_progress(task_id)
    if not progress:
        return {
            "taskId": task_id,
            "percentage": 0,
            "status": "Initiating ...",
        }
        # raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "taskId": task_id,
        "percentage": progress.get("percentage", 0),
        "message": progress.get("message", ""),
        "status": progress.get("status", "idle"),
        "timeRemaining": progress.get("time_remaining"),
        "startTime": progress.get("start_time"),
        "lastUpdated": progress.get("last_updated")
    }
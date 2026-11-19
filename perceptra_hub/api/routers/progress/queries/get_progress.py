import os
import json
import uuid
import importlib
import django
django.setup()
from pathlib import Path
from fastapi import Body
from datetime import datetime
from pydantic import BaseModel
from django.core.cache import cache
from fastapi import HTTPException, status
from starlette.responses import JSONResponse
from fastapi import FastAPI, Depends, APIRouter, Header
from typing import Union, Any, Dict, AnyStr, Optional, List
from common_utils.progress.core import get_progress
from users.auth import get_current_user

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


@router.api_route(
    "/progress/{task_id}", methods=["GET"], tags=["Progress"]
)
def get_task_progress(task_id: str, _user=Depends(get_current_user)):
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
        "isComplete": progress.get("isComplete", False),
        "timeRemaining": progress.get("time_remaining"),
        "startTime": progress.get("start_time"),
        "lastUpdated": progress.get("last_updated"),
    }
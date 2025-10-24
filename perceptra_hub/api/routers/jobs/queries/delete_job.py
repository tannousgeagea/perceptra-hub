# api/jobs.py

import time
from fastapi import APIRouter, HTTPException, Depends
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable, Optional
from pydantic import BaseModel, Field
from typing import Optional
from jobs.models import Job  # Django model
from asgiref.sync import sync_to_async
from api.routers.auth.queries.dependencies import job_project_editor_or_admin_dependency

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
    route_class=TimedRoute
)

@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: int):
    try:
        job = await sync_to_async(Job.objects.get)(id=job_id)
    except Job.DoesNotExist:
        raise HTTPException(status_code=404, detail="Job not found")

    await sync_to_async(job.delete)()  # Deletes the job and related JobImages (via CASCADE)
    return None  # 204 No Content

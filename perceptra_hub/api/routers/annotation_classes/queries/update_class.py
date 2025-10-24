
import os
import time
import django
django.setup()
from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, status
from fastapi import Request, Response
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Callable, Optional
from typing import List, Optional
import datetime
from fastapi import status as http_status

# Import your Django models.
from annotations.models import (
    Annotation,
    AnnotationClass, 
    AnnotationGroup,
)

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


class AnnotationClassOut(BaseModel):
    id: int
    name: str
    color: str
    count: int

class AnnotationClassUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/classes/{id}", methods=["PATCH"], tags=["Annotation Classes"])
def update_class(id: int, updates: AnnotationClassUpdate) -> AnnotationClassOut:
    try:
        annotation_class = AnnotationClass.objects.filter(id=id).first()
        if not annotation_class:
            raise HTTPException(status_code=404, detail="Annotation class not found")
        
        for key, value in updates.dict(exclude_unset=True).items():
            setattr(annotation_class, key, value)
        annotation_class.save()
        return AnnotationClassOut(
            id=annotation_class.id,
            name=annotation_class.name,
            color=annotation_class.color,
            count=Annotation.objects.filter(annotation_class=annotation_class, is_active=True).count()
        )
    
    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unexpected Error: {e}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )
    

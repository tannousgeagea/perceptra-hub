# routes/models.py
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List
from pydantic import BaseModel
from ml_models.models import ModelTag

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

class TagOut(BaseModel):
    id: int
    name: str
    description: Optional[str]

    class Config:
        from_attributes = True

class TagIn(BaseModel):
    name: str
    description: Optional[str] = None

@router.get("/tags", response_model=List[TagOut])
def list_tags():
    return ModelTag.objects.all()

@router.post("/tags", response_model=TagOut)
def create_tag(data: TagIn):
    if ModelTag.objects.filter(name=data.name).exists():
        raise HTTPException(status_code=400, detail="Tag already exists")

    tag = ModelTag.objects.create(name=data.name, description=data.description)
    return tag

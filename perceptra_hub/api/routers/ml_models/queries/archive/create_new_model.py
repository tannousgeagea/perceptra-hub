# routes/models.py
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Literal
from pydantic import BaseModel
from ml_models.models import Model, ModelTag, ModelFramework, ModelTask
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
    responses={404: {"description": "Not found"}},
)

class ModelCreateIn(BaseModel):
    name: str
    description: Optional[str]
    task: Literal["object-detection", "segmentation", "classification"]
    framework: Literal["yolo", "rf-detr", "pytorch", "tensorflow", "onnx"] 
    project_id: str
    tags: Optional[List[str]] = []

class ModelCreateOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    type: str
    projectId: str
    tags: List[str]

    class Config:
        from_attributes = True

@router.api_route(
    "/models", methods=["POST"], response_model=ModelCreateOut, 
)
def create_model(data: ModelCreateIn):
    try:
        task = ModelTask.objects.get(name=data.task)
        framework = ModelFramework.objects.get(name=data.framework)
        project = Project.objects.get(name=data.project_id)

        if Model.objects.filter(name=data.name).exists():
            raise HTTPException(status_code=400, detail="Model name already exists")

        model = Model.objects.create(
            name=data.name,
            task=task,
            framework=framework,
            description=data.description,
            project=project,
        )

        if data.tags:
            tags = ModelTag.objects.filter(name__in=data.tags)
            model.tags.set(tags)

        return ModelCreateOut(
            id=model.id,
            name=model.name,
            description=model.description,
            type=task.name,
            projectId=project.name,
            tags=[tag.name for tag in model.tags.all()]
        )

    except (ModelTask.DoesNotExist, ModelFramework.DoesNotExist, Project.DoesNotExist) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

import os
import time
import importlib
from glob import glob
from typing import Callable
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.images.queries.{f.replace('/', '.')[:-3]}" 
    for f in os.listdir(QUERIES_DIR) 
    if not os.path.isdir(f)
    if f.endswith('.py') 
    if not f.endswith('__.py')
    ]

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
    prefix="/api/v1",
    tags=["Images"],
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)


for Q in QUERIES:
    module = importlib.import_module(Q)
    router.include_router(module.router)
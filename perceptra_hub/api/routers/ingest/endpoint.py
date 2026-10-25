import os
import time
import importlib
from typing import Callable
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi.routing import APIRoute

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.ingest.queries.{f.replace('/', '.')[:-3]}"
    for f in os.listdir(QUERIES_DIR)
    if f.endswith('.py')
    if not f.endswith('__.py')
    ]

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            response.headers["X-Response-Time"] = str(time.time() - before)
            return response

        return custom_route_handler


router = APIRouter(
    prefix="/api/v1",
    route_class=TimedRoute,
    tags=["Ingest"],
    responses={404: {"description": "Not found"}},
)

for Q in QUERIES:
    module = importlib.import_module(Q)
    router.include_router(module.router)

import importlib
import os

from fastapi import APIRouter

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.fleet.queries.{f[:-3]}"
    for f in sorted(os.listdir(QUERIES_DIR))
    if f.endswith(".py") and not f.endswith("__.py")
]

router = APIRouter(
    prefix="/api/v1",
    tags=["Fleet"],
    responses={404: {"description": "Not found"}},
)

for Q in QUERIES:
    module = importlib.import_module(Q)
    router.include_router(module.router)

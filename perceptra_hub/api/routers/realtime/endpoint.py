import os
import importlib
from fastapi import APIRouter

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.realtime.queries.{f[:-3]}"
    for f in os.listdir(QUERIES_DIR)
    if f.endswith('.py')
    if not f.endswith('__.py')
]

# No timing route class here: it would hold on to the (endless) event stream.
router = APIRouter(
    prefix="/api/v1",
    tags=["Realtime"],
    responses={404: {"description": "Not found"}},
)

for Q in QUERIES:
    module = importlib.import_module(Q)
    router.include_router(module.router)

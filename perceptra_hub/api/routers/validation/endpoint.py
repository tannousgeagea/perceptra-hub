import os
import importlib
from glob import glob
from fastapi import APIRouter
from fastapi import HTTPException

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.validation.queries.{f.replace('/', '.')[:-3]}" 
    for f in os.listdir(QUERIES_DIR) 
    if f.endswith('.py') 
    if not f.endswith('__.py')
    ]

router = APIRouter(
    prefix="/api/v1",
    tags=["Validation"],
    responses={404: {"description": "Not found"}},
)


for Q in QUERIES:
    module = importlib.import_module(Q)
    router.include_router(module.router)
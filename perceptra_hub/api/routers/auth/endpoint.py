import os
import importlib
from glob import glob
from fastapi import APIRouter
from fastapi import HTTPException

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.auth.queries.{f.replace('/', '.')[:-3]}" 
    for f in os.listdir(QUERIES_DIR) 
    if f.endswith('.py') 
    if not f.endswith('__.py')
    ]

router = APIRouter(
    prefix="/api/v1",
    responses={404: {"description": "Not found"}},
)


# from .queries.auth import router as auth_router
from .queries.auth_v2 import router as auth_router
from .queries.oauth import router as oauth_router
router.include_router(auth_router, tags=["Authentication"])
router.include_router(oauth_router)

# for Q in QUERIES:
#     module = importlib.import_module(Q)
#     router.include_router(module.router)
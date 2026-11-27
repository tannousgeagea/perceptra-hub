import os
import importlib
from glob import glob
from fastapi import APIRouter
from fastapi import HTTPException

QUERIES_DIR = os.path.dirname(__file__) + "/queries"
QUERIES = [
    f"api.routers.agent.queries.{f.replace('/', '.')[:-3]}" 
    for f in os.listdir(QUERIES_DIR) 
    if f.endswith('.py') 
    if not f.endswith('__.py')
    ]
# The installer's fixed path (/agents/install-agent.sh) must be registered before
# agent.py's GET /agents/{agent_id}, which would otherwise catch it.
QUERIES.sort(key=lambda q: (not q.endswith('.agent_install'), q))

router = APIRouter(
    prefix="/api/v1",
    tags=["Agent"],
    responses={404: {"description": "Not found"}},
)


for Q in QUERIES:
    module = importlib.import_module(Q)
    router.include_router(module.router)
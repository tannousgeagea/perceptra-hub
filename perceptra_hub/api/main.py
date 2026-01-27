import os
import uvicorn
import logging
import inspect
import importlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from api.config.celery_utils import celery_app
from api.error_handling import setup_exception_handlers


ROUTERS_DIR = os.path.dirname(__file__) + "/routers"
ROUTERS = [
    f"api.routers.{f.replace('/', '.')}" 
    for f in os.listdir(ROUTERS_DIR)
    if not f.endswith('__pycache__')
    if not f.endswith('__.py')
    ]

def create_app() -> FastAPI:
    
    import django
    django.setup()
    
    tags_meta = [
        {
            "name": "Data Upload DATA API",
            "description": "Image Upload data entrypoint API"
        }
    ]

    app = FastAPI(
        openapi_tags = tags_meta,
        debug=True,
        title="Data Upload data entrypoint API",
        summary="",
        version="0.0.1",
        contact={
            "name": "Tannous Geagea",
            "url": "https://wasteant.com",
            "email": "tannous.geagea@wasteant.com",            
        },
        openapi_url="/openapi.json"
    )

    origins = ["http://localhost:3002", "http://localhost:29083", "http://localhost:5173", os.getenv("FRONTEND_ENDPOINT")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["X-Requested-With", "X-Request-ID", "X-Organization-ID", "Authorization"],
        expose_headers=["X-Request-ID", "X-Progress-ID", "x-response-time"],
    )
    
    app.celery_app = celery_app

    for R in ROUTERS:
        try:
            module = importlib.import_module(R)
            attr = getattr(module, 'endpoint')
            if inspect.ismodule(attr):
                app.include_router(module.endpoint.router)
        except ImportError as err:
            logging.error(f'Failed to import {R}: {err}')
    
    return app

app = create_app()
setup_exception_handlers(app)
celery = app.celery_app
celery.autodiscover_tasks(['api.tasks'])

if __name__ == "__main__":
    import os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv('DATA_API_PORT')), log_level="debug", reload=True)
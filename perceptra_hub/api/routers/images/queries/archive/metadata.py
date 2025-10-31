import os
import math
import uuid
import time
import django
import shutil
django.setup()
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

from tenants.models import (
    Tenant,
    EdgeBox,
    Plant,
    Domain,
)
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
)

@router.api_route(
    "/images/metadata", methods=["GET"], tags=["Images"]
)
def get_images_metadata(
    response: Response,
    plant:str=None,
    ):
    results = {}
    try:
        if not plant:
            plants = Plant.objects.all()
        else:
            plants = Plant.objects.filter(plant_name=plant)
            
        filters = []
        filters.append(
            {
                "key": 'plant',
                "title": 'Plant',
                "description": "",
                "placeholder": "Plant",
                "type": "select",
                "items": [
                    {
                        "key": plant.plant_name,
                        "value": plant.plant_name,
                    } for plant in plants
                ]
            }
        )
        
        filters.append(
            {
                "key": 'location',
                "title": 'Location',
                "description": "",
                "placeholder": "Location",
                "type": "select",
                "items": [
                    {
                        "key": "Gate 3",
                        "value": "Gate 3",
                    },
                    {
                        "key": "Gate 4",
                        "value": "Gate 4",
                    }
                ]
            }
        )
        results = {
            'data': {
                'columns': [],
                'filters': filters,
            }
        }
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    


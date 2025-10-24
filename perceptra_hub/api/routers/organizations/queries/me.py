import time
from fastapi import APIRouter, Depends
from users.models import CustomUser as User
from projects.models import Project
from organizations.models import Organization
from memberships.models import OrganizationMembership
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Optional, Callable
from fastapi import Request, Response
from fastapi import Depends, HTTPException
from api.routers.auth.queries.dependencies import get_current_user

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

router = APIRouter()

class OrganizationOut(BaseModel):
    id: int
    name: str
    userCount:int
    projectCount:int

@router.get("/organizations/me", response_model=OrganizationOut)
def get_my_organization(user: User = Depends(get_current_user)):
    org_membership = OrganizationMembership.objects.select_related("organization").filter(user=user).first()
    if not org_membership:
        raise HTTPException(status_code=401, detail="Invalid token")

    org = org_membership.organization
    return OrganizationOut(
        id=org.id,
        name=org.name,
        userCount=OrganizationMembership.objects.filter(organization=org).count(),
        projectCount=Project.objects.filter(organization=org).count()
    )
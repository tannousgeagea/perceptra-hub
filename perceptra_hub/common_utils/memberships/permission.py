import os
from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer
from memberships.models import ProjectMembership, OrganizationMembership
from django.contrib.auth import get_user_model
from typing import Callable
from jose import jwt

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
ALGORITHM = "HS256"
User = get_user_model()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=401, detail="User not found")

    return user

def check_project_access(project_id: int, role_required: list[str] = None) -> Callable:
    def dependency(user: User = Depends(get_current_user)):
        membership = ProjectMembership.objects.filter(user=user, project_id=project_id).select_related("role").first()
        if not membership:
            raise HTTPException(status_code=403, detail="Access denied to this project")
        if role_required and membership.role.name not in role_required:
            raise HTTPException(status_code=403, detail="Insufficient project role permissions")
        return membership
    return dependency

def check_organization_access(org_id: int, role_required: list[str] = None) -> Callable:
    def dependency(user: User = Depends(get_current_user)):
        membership = OrganizationMembership.objects.filter(user=user, organization_id=org_id).select_related("role").first()
        if not membership:
            raise HTTPException(status_code=403, detail="Access denied to this organization")
        if role_required and membership.role.name not in role_required:
            raise HTTPException(status_code=403, detail="Insufficient organization role permissions")
        return membership
    return dependency
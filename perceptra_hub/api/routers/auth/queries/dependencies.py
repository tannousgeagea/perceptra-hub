from fastapi import Depends, HTTPException, Path
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from users.models import CustomUser as User
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
import os
from typing import Callable
from projects.models import Project
from jobs.models import Job
from memberships.models import ProjectMembership, OrganizationMembership

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
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

def organization_access_dependency(org_id: int = Path(...), user: User = Depends(get_current_user)):
    membership = OrganizationMembership.objects.filter(user=user, organization_id=org_id).select_related("role").first()
    if not membership or membership.role.name.lower() != "admin":
        raise HTTPException(status_code=403, detail="Access denied to this organization")
    return membership

def user_project_access_dependency(project_id: str = Path(...), user: User = Depends(get_current_user)):
    membership = ProjectMembership.objects.filter(user=user, project__name=project_id).select_related("role").first()
    if not membership or membership.role.name.lower() != "admin":
        raise HTTPException(status_code=403, detail="Access denied to this project")
    return membership

def project_admin_or_org_admin_dependency(
    project_id: str = Path(...), 
    user: User = Depends(get_current_user)
):
    project = Project.objects.select_related("organization").filter(name=project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if user is project admin
    project_membership = ProjectMembership.objects.filter(
        user=user, project=project
    ).select_related("role").first()
    if project_membership and project_membership.role.name.lower() == "admin":
        return {"membership": project_membership, "source": "project"}

    # Else check if user is organization admin
    org_membership = OrganizationMembership.objects.filter(
        user=user, organization=project.organization
    ).select_related("role").first()
    if org_membership and org_membership.role.name.lower() == "admin":
        return {"membership": org_membership, "source": "organization"}

    raise HTTPException(status_code=403, detail="Requires admin access to project or organization")

def project_edit_admin_or_org_admin_dependency(
    project_id: str = Path(...), 
    user: User = Depends(get_current_user)
):
    project = Project.objects.select_related("organization").filter(name=project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if user is project admin
    project_membership = ProjectMembership.objects.filter(
        user=user, project=project
    ).select_related("role").first()
    if project_membership and project_membership.role.name.lower() in ["admin", "editor"]:
        return {"membership": project_membership, "source": "project"}

    # Else check if user is organization admin
    org_membership = OrganizationMembership.objects.filter(
        user=user, organization=project.organization
    ).select_related("role").first()
    if org_membership and org_membership.role.name.lower() == "admin":
        return {"membership": org_membership, "source": "organization"}

    raise HTTPException(status_code=403, detail="Requires admin or edit access to project or organization")

def job_project_admin_or_org_admin_dependency(
    job_id: int = Path(...),
    user: User = Depends(get_current_user)
):
    job = get_object_or_404(Job.objects.select_related("project__organization"), id=job_id)
    project = job.project

    # Project-level check
    project_membership = ProjectMembership.objects.filter(
        user=user, project=project
    ).select_related("role").first()
    if project_membership and project_membership.role.name.lower() == "admin":
        return {"membership": project_membership, "source": "project"}

    # Org-level check
    org_membership = OrganizationMembership.objects.filter(
        user=user, organization=project.organization
    ).select_related("role").first()
    if org_membership and org_membership.role.name.lower() == "admin":
        return {"membership": org_membership, "source": "organization"}

    raise HTTPException(status_code=403, detail="Requires admin access to project or organization")

def job_project_editor_or_admin_dependency(
    job_id: int = Path(...),
    user: User = Depends(get_current_user)
):
    job = get_object_or_404(Job.objects.select_related("project__organization"), id=job_id)
    project = job.project

    # Project-level check (admin or editor)
    project_membership = ProjectMembership.objects.filter(
        user=user, project=project
    ).select_related("role").first()

    if project_membership and project_membership.role.name.lower() in ["admin", "editor"]:
        return {"membership": project_membership, "source": "project"}

    # Org-level check (admin only still makes sense here)
    org_membership = OrganizationMembership.objects.filter(
        user=user, organization=project.organization
    ).select_related("role").first()

    if org_membership and org_membership.role.name.lower() == "admin":
        return {"membership": org_membership, "source": "organization"}

    raise HTTPException(status_code=403, detail="Requires editor or admin access to project")

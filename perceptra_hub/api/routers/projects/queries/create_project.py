from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Callable
import django
import time
from asgiref.sync import sync_to_async
from fastapi import Request, Response
from fastapi.routing import APIRoute, APIRouter
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import transaction
import os


# Import your Django models after setup
from projects.models import Project, ProjectType, Visibility 
from annotations.models import AnnotationGroup, AnnotationClass
from organizations.models import Organization

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
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request/response
class AnnotationClassCreate(BaseModel):
    class_id: int = Field(..., ge=0, description="Unique class ID within the annotation group")
    name: str = Field(..., min_length=1, max_length=255)
    color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$', description="Hex color code")
    description: Optional[str] = None

class AnnotationClassResponse(BaseModel):
    id: int
    class_id: int
    name: str
    color: Optional[str]
    description: Optional[str]
    created_at: str

    class Config:
        from_attributes = True

class AnnotationGroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    classes: List[AnnotationClassCreate] = Field(default_factory=list)

class AnnotationGroupResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: str
    classes: List[AnnotationClassResponse]

    class Config:
        from_attributes = True

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    project_type_name: str = Field(..., description="Must be one of: object-detection, classification, segmentation")
    visibility_name: str = Field(..., description="Must be one of: private, public")
    organization_id: Optional[int] = None
    annotation_groups: List[AnnotationGroupCreate] = Field(default_factory=list)

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    project_type_name: str
    visibility_name: str
    organization_id: Optional[int]
    created_at: str
    last_edited: str
    is_active: bool
    annotation_groups: List[AnnotationGroupResponse]

    class Config:
        from_attributes = True

# Async wrapper functions for Django ORM calls
@sync_to_async
def get_project_type(name: str):
    return ProjectType.objects.get(name=name)

@sync_to_async
def get_visibility(name: str):
    return Visibility.objects.get(name=name)

@sync_to_async
def get_organization(org_id: int):
    return Organization.objects.get(id=org_id)

@sync_to_async
def create_project_with_groups(project_data, project_type, visibility, organization):
    with transaction.atomic():
        # Create the project
        project = Project.objects.create(
            name=project_data.name,
            description=project_data.description,
            thumbnail_url=project_data.thumbnail_url,
            project_type=project_type,
            visibility=visibility,
            organization=organization
        )
        
        # Create annotation groups and their classes
        created_groups = []
        for group_data in project_data.annotation_groups:
            # Create annotation group
            annotation_group = AnnotationGroup.objects.create(
                project=project,
                name=group_data.name,
                description=group_data.description
            )
            
            # Create annotation classes for this group
            created_classes = []
            for class_data in group_data.classes:
                annotation_class = AnnotationClass.objects.create(
                    annotation_group=annotation_group,
                    class_id=class_data.class_id,
                    name=class_data.name,
                    color=class_data.color,
                    description=class_data.description
                )
                created_classes.append(annotation_class)
            
            # Store created classes with the group (temporary attribute for response building)
            annotation_group.created_classes = created_classes
            created_groups.append(annotation_group)
        
        return project, created_groups

@sync_to_async
def get_project_by_id(project_id: int):
    return Project.objects.select_related('project_type', 'visibility', 'organization').get(id=project_id)

@sync_to_async
def get_annotation_groups_for_project(project):
    return list(AnnotationGroup.objects.filter(project=project).prefetch_related('classes'))

@sync_to_async
def get_all_project_types():
    return list(ProjectType.objects.all().values('id', 'name', 'description'))

@sync_to_async
def get_all_visibility_options():
    return list(Visibility.objects.all().values('id', 'name', 'description'))

@sync_to_async
def get_projects_list(skip: int, limit: int):
    return list(Project.objects.select_related('project_type', 'visibility', 'organization')[skip:skip+limit])

# Dependency to validate project types
async def validate_project_type(project_type_name: str):
    try:
        return await get_project_type(project_type_name)
    except ObjectDoesNotExist:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid project_type_name: {project_type_name}. Must be one of: object-detection, classification, segmentation"
        )

# Dependency to validate visibility
async def validate_visibility(visibility_name: str):
    try:
        return await get_visibility(visibility_name)
    except ObjectDoesNotExist:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid visibility_name: {visibility_name}. Must be one of: private, public"
        )

# Dependency to validate organization (if provided)
async def validate_organization(organization_id: Optional[int]):
    if organization_id is None:
        return None
    try:
        return await get_organization(organization_id)
    except ObjectDoesNotExist:
        raise HTTPException(
            status_code=400, 
            detail=f"Organization with id {organization_id} does not exist"
        )

@router.post("/projects/add", response_model=ProjectResponse, status_code=201)
async def create_project(project_data: ProjectCreate):
    """
    Create a new project with annotation groups and classes.
    
    - **name**: Unique project name
    - **project_type_name**: Must be one of the existing project types
    - **visibility_name**: Must be one of the existing visibility options
    - **annotation_groups**: List of annotation groups with their classes
    """
    try:
        # Validate dependencies
        project_type = await validate_project_type(project_data.project_type_name)
        visibility = await validate_visibility(project_data.visibility_name)
        organization = await validate_organization(project_data.organization_id)
        
        # Create project with annotation groups
        project, created_groups = await create_project_with_groups(
            project_data, project_type, visibility, organization
        )
        
        # Prepare response
        response_groups = []
        for group in created_groups:
            group_classes = [
                AnnotationClassResponse(
                    id=cls.id,
                    class_id=cls.class_id,
                    name=cls.name,
                    color=cls.color,
                    description=cls.description,
                    created_at=cls.created_at.isoformat()
                )
                for cls in group.created_classes
            ]
            
            response_groups.append(
                AnnotationGroupResponse(
                    id=group.id,
                    name=group.name,
                    description=group.description,
                    created_at=group.created_at.isoformat(),
                    classes=group_classes
                )
            )
        
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            thumbnail_url=project.thumbnail_url,
            project_type_name=project.project_type.name,
            visibility_name=project.visibility.name,
            organization_id=project.organization.id if project.organization else None,
            created_at=project.created_at.isoformat(),
            last_edited=project.last_edited.isoformat(),
            is_active=project.is_active,
            annotation_groups=response_groups
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    """Get a project by ID with all its annotation groups and classes."""
    try:
        project = await get_project_by_id(project_id)
        
        # Get annotation groups with their classes
        annotation_groups = await get_annotation_groups_for_project(project)
        
        response_groups = []
        for group in annotation_groups:
            group_classes = [
                AnnotationClassResponse(
                    id=cls.id,
                    class_id=cls.class_id,
                    name=cls.name,
                    color=cls.color,
                    description=cls.description,
                    created_at=cls.created_at.isoformat()
                )
                for cls in group.classes.all()
            ]
            
            response_groups.append(
                AnnotationGroupResponse(
                    id=group.id,
                    name=group.name,
                    description=group.description,
                    created_at=group.created_at.isoformat(),
                    classes=group_classes
                )
            )
        
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            thumbnail_url=project.thumbnail_url,
            project_type_name=project.project_type.name,
            visibility_name=project.visibility.name,
            organization_id=project.organization.id if project.organization else None,
            created_at=project.created_at.isoformat(),
            last_edited=project.last_edited.isoformat(),
            is_active=project.is_active,
            annotation_groups=response_groups
        )
        
    except ObjectDoesNotExist:
        raise HTTPException(status_code=404, detail=f"Project with id {project_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/project-types/")
async def get_project_types():
    """Get all available project types."""
    try:
        project_types = await get_all_project_types()
        return {"project_types": project_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/visibility-options/")
async def get_visibility_options():
    """Get all available visibility options."""
    try:
        visibility_options = await get_all_visibility_options()
        return {"visibility_options": visibility_options}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/projects/list")
async def list_projects(skip: int = 0, limit: int = 100):
    """List all projects with pagination."""
    try:
        projects = await get_projects_list(skip, limit)
        
        result = []
        for project in projects:
            result.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "project_type_name": project.project_type.name,
                "visibility_name": project.visibility.name,
                "organization_id": project.organization.id if project.organization else None,
                "created_at": project.created_at.isoformat(),
                "last_edited": project.last_edited.isoformat(),
                "is_active": project.is_active
            })
        
        return {"projects": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
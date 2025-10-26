"""
FastAPI dependencies for authentication, database, and organization management.
"""
from fastapi import Depends, HTTPException, status, Header, Request
from typing import Optional, Generator
import logging
import jwt
from uuid import UUID
from datetime import datetime

# Django imports
from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from django.conf import settings

from projects.models import Project
from organizations.models import Organization
from memberships.models import OrganizationMembership, Role, ProjectMembership
from asgiref.sync import sync_to_async
User = get_user_model()
logger = logging.getLogger(__name__)


# ============= Database Dependency =============

def get_db() -> Generator:
    """
    Database session dependency for Django ORM.
    Django manages its own connections, so this is a pass-through.
    """
    try:
        yield None
    finally:
        pass


# ============= Authentication Dependencies =============
@sync_to_async
def fetch_user_from_db(user_id: str) -> User:
    """Synchronous function to fetch user from database."""
    try:
        return User.objects.get(id=user_id, is_active=True)
    except User.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        authorization: Authorization header with Bearer token
    
    Returns:
        Django User instance
    
    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Extract token
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Use 'Bearer <token>'",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Decode JWT token
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=['HS256']
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from token payload
        user_id = payload.get('user_id')
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Fetch user from database
        user = await fetch_user_from_db(user_id)
        
        return user
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============= Organization Dependencies =============
@sync_to_async
def fetch_organization_and_verify_membership(user: User, org_id: Optional[str], org_slug: Optional[str]) -> Organization:
    """
    Synchronous function to fetch organization and verify user membership.
    """
    try:
        # Get organization
        if org_id:
            organization = Organization.objects.get(org_id=org_id)
        else:
            organization = Organization.objects.get(slug=org_slug)
        
        # Verify user is a member of this organization
        try:
            membership = OrganizationMembership.objects.get(
                user=user,
                organization=organization
            )
        except OrganizationMembership.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User is not a member of organization '{organization.name}'"
            )
        
        return organization
        
    except Organization.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization not found"
        )

async def get_current_organization(
    request: Request,
    user: User = Depends(get_current_user)
) -> Organization:
    """
    Get current organization from request context.
    
    The organization can be provided via:
    1. X-Organization-ID header
    2. X-Organization-Slug header
    3. Query parameter: organization_id
    4. Query parameter: organization_slug
    
    Args:
        request: FastAPI request object
        user: Current authenticated user
    
    Returns:
        Organization instance
    
    Raises:
        HTTPException: If organization not found or user not a member
    """
    # Try to get organization from headers
    org_id = request.headers.get('X-Organization-ID')
    org_slug = request.headers.get('X-Organization-Slug')
    
    # Try query parameters if headers not present
    if not org_id:
        org_id = request.query_params.get('organization_id')
    if not org_slug:
        org_slug = request.query_params.get('organization_slug')
    
    if not org_id and not org_slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization ID or slug required. Provide via X-Organization-ID header, X-Organization-Slug header, or query parameter"
        )
    
    try:
        # Get organization
        organization = await fetch_organization_and_verify_membership(
            user, org_id, org_slug
        )
        return organization
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve organization"
        )

@sync_to_async
def fetch_user_organization_role(user: User, organization: Organization) -> Role:
    """
    Synchronous function to get user's role in organization.
    """
    try:
        membership = OrganizationMembership.objects.get(
            user=user,
            organization=organization
        )
        return membership.role
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a member of this organization"
        )


async def get_user_organization_role(
    user: User = Depends(get_current_user),
    organization: Organization = Depends(get_current_organization)
) -> Role:
    """
    Get user's role in the current organization.
    
    Args:
        user: Current authenticated user
        organization: Current organization
    
    Returns:
        Role instance
    """
    return await fetch_user_organization_role(user, organization)



# ============= Permission Dependencies =============

def require_organization_role(*required_roles: str):
    """
    Dependency to require specific organization roles.
    
    Usage:
        @router.post("/profiles")
        async def create_profile(
            user=Depends(require_organization_role("admin", "owner"))
        ):
            ...
    
    Args:
        required_roles: Role names that are allowed
    
    Returns:
        Dependency function
    """
    async def role_checker(
        user: User = Depends(get_current_user),
        organization: Organization = Depends(get_current_organization)
    ) -> tuple[User, Organization, Role]:
        """Check if user has required role in organization."""
        try:
            membership = OrganizationMembership.objects.get(
                user=user,
                organization=organization
            )
            
            if membership.role.name not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {', '.join(required_roles)}. Your role: {membership.role.name}"
                )
            
            return user, organization, membership.role
            
        except OrganizationMembership.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not a member of this organization"
            )
    
    return role_checker


def require_organization_admin():
    """
    Dependency to require organization admin role.
    
    Usage:
        @router.post("/profiles")
        async def create_profile(
            deps=Depends(require_organization_admin())
        ):
            user, organization, role = deps
            ...
    """
    return require_organization_role('admin', 'owner')


# ============= Pagination Dependency =============

class PaginationParams:
    """Pagination parameters."""
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        max_limit: int = 1000
    ):
        self.skip = max(0, skip)
        self.limit = min(max(1, limit), max_limit)
        self.page = (self.skip // self.limit) + 1 if self.limit > 0 else 1


def get_pagination(
    skip: int = 0,
    limit: int = 100
) -> PaginationParams:
    """
    Pagination dependency.
    
    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return
    
    Returns:
        PaginationParams instance
    """
    return PaginationParams(skip=skip, limit=limit)


# ============= Optional Authentication =============

async def get_optional_user(
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Useful for endpoints that work differently for authenticated/anonymous users.
    """
    if not authorization:
        return None
    
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


# ============= Helper Functions =============

def user_has_organization_permission(
    user: User,
    organization: Organization,
    required_roles: list[str]
) -> bool:
    """
    Check if user has any of the required roles in organization.
    
    Args:
        user: User instance
        organization: Organization instance
        required_roles: List of role names
    
    Returns:
        True if user has permission, False otherwise
    """
    try:
        membership = OrganizationMembership.objects.get(
            user=user,
            organization=organization
        )
        return membership.role.name in required_roles
    except OrganizationMembership.DoesNotExist:
        return False


def get_user_organizations(user: User) -> list[Organization]:
    """
    Get all organizations a user belongs to.
    
    Args:
        user: User instance
    
    Returns:
        List of Organization instances
    """
    memberships = OrganizationMembership.objects.filter(
        user=user
    ).select_related('organization')
    
    return [m.organization for m in memberships]


# ============= Development Mode Dependencies (REMOVE IN PRODUCTION) =============

async def bypass_auth_dev(request: Request) -> tuple[User, Organization]:
    """
    Bypass authentication for development/testing.
    
    WARNING: Only use this in development mode with DEBUG=True!
    REMOVE THIS DEPENDENCY IN PRODUCTION!
    """
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development bypass only available in DEBUG mode"
        )
    
    logger.warning("⚠️  DEVELOPMENT MODE: Authentication bypassed")
    
    # Get or create test user
    user, created = User.objects.get_or_create(
        username='dev-user',
        defaults={
            'email': 'dev@example.com',
            'first_name': 'Dev',
            'last_name': 'User',
            'is_active': True
        }
    )
    
    if created:
        user.set_password('devpassword')
        user.save()
        logger.warning("Created dev user: dev-user / devpassword")
    
    # Get or create test organization
    organization, created = Organization.objects.get_or_create(
        slug='dev-org',
        defaults={'name': 'Development Organization'}
    )
    
    if created:
        logger.warning("Created dev organization: dev-org")
    
    # Get or create admin role
    admin_role, _ = Role.objects.get_or_create(
        name='admin',
        defaults={'description': 'Administrator'}
    )
    
    # Ensure user is member of organization
    OrganizationMembership.objects.get_or_create(
        user=user,
        organization=organization,
        defaults={'role': admin_role}
    )
    
    return user, organization


# ============= Context Classes for Better Type Hints =============

class RequestContext:
    """Request context with user and organization."""
    
    def __init__(
        self,
        user: User,
        organization: Organization,
        role: Optional[Role] = None
    ):
        self.user = user
        self.organization = organization
        self.role = role
    
    def has_role(self, *role_names: str) -> bool:
        """Check if user has any of the specified roles."""
        return self.role and self.role.name in role_names
    
    def require_role(self, *role_names: str):
        """Raise exception if user doesn't have required role."""
        if not self.has_role(*role_names):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(role_names)}"
            )


async def get_request_context(
    user: User = Depends(get_current_user),
    organization: Organization = Depends(get_current_organization),
    role: Role = Depends(get_user_organization_role)
) -> RequestContext:
    """
    Get complete request context with user, organization, and role.
    
    Usage:
        @router.get("/profiles")
        async def list_profiles(ctx: RequestContext = Depends(get_request_context)):
            # ctx.user, ctx.organization, ctx.role are all available
            ...
    """
    return RequestContext(user, organization, role)


# ============= Project Context =============

class ProjectContext:
    """Project context with user, organization, role, and project."""
    
    def __init__(
        self,
        user: User,
        organization: Organization,
        org_role: Role,
        project: Project,
        project_role: Optional[Role] = None
    ):
        self.user = user
        self.organization = organization
        self.org_role = org_role
        self.project = project
        self.project_role = project_role
    
    def has_org_role(self, *role_names: str) -> bool:
        """Check if user has any of the specified organization roles."""
        return self.org_role and self.org_role.name in role_names
    
    def has_project_role(self, *role_names: str) -> bool:
        """Check if user has any of the specified project roles."""
        return self.project_role and self.project_role.name in role_names
    
    def is_org_admin(self) -> bool:
        """Check if user is organization admin/owner."""
        return self.has_org_role('admin', 'owner')
    
    def is_project_member(self) -> bool:
        """Check if user is a project member."""
        return self.project_role is not None
    
    def require_org_role(self, *role_names: str):
        """Raise exception if user doesn't have required organization role."""
        if not self.has_org_role(*role_names):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required organization roles: {', '.join(role_names)}"
            )
    
    def require_project_role(self, *role_names: str):
        """Raise exception if user doesn't have required project role."""
        if not self.has_project_role(*role_names):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required project roles: {', '.join(role_names)}"
            )
    
    def require_project_access(self):
        """Require user to be org admin or project member."""
        if not self.is_org_admin() and not self.is_project_member():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You must be an organization admin or project member."
            )


@sync_to_async
def fetch_project_with_access_check(
    user: User,
    organization: Organization,
    org_role: Role,
    project_id: UUID
) -> tuple[Project, Optional[Role]]:
    """
    Fetch project and check user access.
    Returns project and user's project role (None if only org admin).
    """
    from projects.models import Project
    
    # Get project
    try:
        project = Project.objects.select_related('organization').get(
            project_id=project_id,
            organization=organization,
            is_deleted=False
        )
    except Project.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    # Check if user is org admin/owner (automatic access)
    if org_role.name in ['admin', 'owner']:
        return project, None  # Org admins don't need project role
    
    # Check if user is project member
    try:
        membership = ProjectMembership.objects.select_related('role').get(
            user=user,
            project=project
        )
        return project, membership.role
    except ProjectMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You must be an organization admin or project member."
        )


async def get_project_context(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
) -> ProjectContext:
    """
    Get complete project context with user, organization, and project.
    
    Automatically checks if user has access to the project:
    - Organization admin/owner: automatic access
    - Project member: access granted
    - Others: 403 Forbidden
    
    Usage:
        @router.get("/projects/{project_id}/members")
        async def get_members(
            project_ctx: ProjectContext = Depends(get_project_context)
        ):
            # project_ctx.project, project_ctx.user, etc. available
            project_ctx.require_project_access()
            ...
    """
    project, project_role = await fetch_project_with_access_check(
        ctx.user,
        ctx.organization,
        ctx.role,
        project_id
    )
    
    return ProjectContext(
        user=ctx.user,
        organization=ctx.organization,
        org_role=ctx.role,
        project=project,
        project_role=project_role
    )

# ============= Usage Examples in Docstring =============

"""
Usage Examples:

1. Basic authenticated endpoint:
    @router.get("/profiles")
    async def list_profiles(
        user: User = Depends(get_current_user),
        organization: Organization = Depends(get_current_organization)
    ):
        # User and organization automatically injected
        profiles = StorageProfile.objects.filter(tenant=organization)
        return profiles

2. Require specific role:
    @router.post("/profiles")
    async def create_profile(
        profile_data: dict,
        deps = Depends(require_organization_admin())
    ):
        user, organization, role = deps
        # Only admins/owners can access
        ...

3. Using RequestContext:
    @router.get("/profiles")
    async def list_profiles(
        ctx: RequestContext = Depends(get_request_context)
    ):
        # Check role inline
        if ctx.has_role('admin', 'owner'):
            # Show all profiles
            ...
        else:
            # Show limited profiles
            ...

4. Development bypass (REMOVE IN PRODUCTION):
    @router.get("/dev-test")
    async def dev_test(
        deps = Depends(bypass_auth_dev)
    ):
        user, organization = deps
        # No auth required in development
        ...
"""
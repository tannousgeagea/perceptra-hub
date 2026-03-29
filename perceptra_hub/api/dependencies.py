"""
FastAPI dependencies for authentication, database, and organization management.
"""
from fastapi import Depends, HTTPException, status, Header, Request
from typing import Optional, Generator, Tuple, List, Annotated
import logging
import jwt
from uuid import UUID
from datetime import datetime

# Django imports
from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from django.conf import settings
from django.utils import timezone
from projects.models import Project
from api_keys.models import APIKey
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
    """
    Request context with user, organization, role, and optional API key.

    Works with both JWT and API key authentication. The `role` is always
    a Role model instance so existing has_role()/require_role() calls
    continue to work unchanged.
    """

    def __init__(
        self,
        user,
        organization: Organization,
        role: Optional[Role] = None,
        membership: Optional[OrganizationMembership] = None,
        api_key: Optional[APIKey] = None,
        auth_method: str = 'jwt',
    ):
        self.user = user
        self.organization = organization
        self.role = role
        self.membership = membership
        self.api_key = api_key
        self.auth_method = auth_method

    @property
    def is_api_key_auth(self) -> bool:
        return self.auth_method == 'api_key'

    def has_role(self, *role_names: str) -> bool:
        """Check if user has any of the specified roles."""
        return self.role is not None and self.role.name in role_names

    def require_role(self, *role_names: str):
        """Raise exception if user doesn't have required role."""
        if not self.has_role(*role_names):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(role_names)}"
            )


@sync_to_async
def _fetch_organization_membership(user, organization) -> OrganizationMembership:
    """Fetch user's active membership in organization."""
    try:
        return OrganizationMembership.objects.select_related('role').get(
            user=user,
            organization=organization,
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User is not a member of organization '{organization.name}'"
        )


async def get_request_context(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    x_organization_id: Annotated[Optional[str], Header(alias="X-Organization-ID")] = None,
    x_organization_slug: Annotated[Optional[str], Header(alias="X-Organization-Slug")] = None,
) -> RequestContext:
    """
    Get complete request context with user, organization, and role.

    Supports dual authentication:
      - **API Key** (X-API-Key header) — organization is implicit from the key.
      - **JWT** (Authorization: Bearer) — organization via X-Organization-ID/Slug header.

    API Key takes priority if both are provided.

    Usage:
        @router.get("/profiles")
        async def list_profiles(ctx: RequestContext = Depends(get_request_context)):
            # ctx.user, ctx.organization, ctx.role are all available
            # ctx.is_api_key_auth, ctx.api_key for API key specific logic
            ...
    """
    # ── API Key authentication (priority) ──
    if x_api_key:
        from api_keys.auth import authenticate_with_api_key
        return await authenticate_with_api_key(request, x_api_key)

    # ── JWT authentication ──
    if authorization:
        user = await get_current_user(authorization)

        # Resolve organization from headers or query params
        org_id = x_organization_id or request.query_params.get('organization_id')
        org_slug = x_organization_slug or request.query_params.get('organization_slug')

        if not org_id and not org_slug:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization ID or slug required. Provide via X-Organization-ID header, X-Organization-Slug header, or query parameter"
            )

        organization = await fetch_organization_and_verify_membership(user, org_id, org_slug)
        membership = await _fetch_organization_membership(user, organization)

        return RequestContext(
            user=user,
            organization=organization,
            role=membership.role,
            membership=membership,
            auth_method='jwt',
        )

    # ── No authentication ──
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key or Authorization header.",
        headers={"WWW-Authenticate": "ApiKey, Bearer"},
    )


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
    
    # NEW: Convenience method
    def can_edit(self) -> bool:
        """Check if user can edit project (admin or project editor/owner)."""
        return self.is_org_admin() or self.has_project_role('owner', 'editor', 'admin')
    
    # NEW: Convenience method
    def can_annotate(self) -> bool:
        """Check if user can annotate (any project member or org admin)."""
        return self.is_org_admin() or self.is_project_member()
    
    def require_org_role(self, *role_names: str):
        """Raise exception if user doesn't have required organization role."""
        if not self.has_org_role(*role_names):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required organization roles: {', '.join(role_names)}"
            )
    
    def require_project_role(self, *role_names: str):
        """Raise exception if user doesn't have required project role."""
        if self.is_org_admin():
            return True
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
    
    # NEW: More specific permission check
    def require_edit_permission(self):
        """Require user to have edit permissions."""
        if not self.can_edit():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Edit permission required. Must be organization admin or project owner/editor."
            )
    
    # NEW: For annotation endpoints
    def require_annotate_permission(self):
        """Require user to have annotation permissions."""
        if not self.can_annotate():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Annotation permission required. Must be project member or organization admin."
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
            is_deleted=False,
            is_active=True,
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

# ============= Permission & Scope Dependencies =============

def require_permission(permission: str):
    """
    Dependency factory to require a specific permission level.
    Works with both JWT (role-based) and API Key (permission-based) auth.

    Usage:
        @router.post("/endpoint")
        async def endpoint(ctx = Depends(require_permission("write"))):
            ...
    """
    async def _check_permission(ctx: RequestContext = Depends(get_request_context)):
        if ctx.api_key:
            from api_keys.auth import APIKeyAuth
            APIKeyAuth.check_permission(ctx.api_key, permission)
        else:
            role_permission_map = {
                'read': ['admin', 'owner', 'editor', 'viewer'],
                'write': ['admin', 'owner', 'editor'],
                'admin': ['admin', 'owner'],
            }
            allowed_roles = role_permission_map.get(permission, [])
            if not ctx.role or ctx.role.name not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires '{permission}' permission"
                )
        return ctx

    return _check_permission


def require_scope(scope: str):
    """
    Dependency factory to require a specific API key scope.
    Only enforced for API key auth; JWT auth bypasses scopes.

    Usage:
        @router.get("/endpoint")
        async def endpoint(ctx = Depends(require_scope("projects:read"))):
            ...
    """
    async def _check_scope(ctx: RequestContext = Depends(get_request_context)):
        if ctx.api_key:
            from api_keys.auth import APIKeyAuth
            APIKeyAuth.check_scope(ctx.api_key, scope)
        return ctx

    return _check_scope


# ============= Usage Examples =============

"""
Usage Examples:

1. Basic context (supports both JWT and API key automatically):
    @router.get("/profiles")
    async def list_profiles(ctx: RequestContext = Depends(get_request_context)):
        ctx.require_role('admin', 'owner')
        # ctx.user, ctx.organization, ctx.role available
        # ctx.is_api_key_auth, ctx.api_key for API key specific logic

2. Require write permission (works with both JWT roles and API key permissions):
    @router.post("/profiles")
    async def create_profile(ctx = Depends(require_permission("write"))):
        ...

3. Require specific scope (API key only, JWT bypasses):
    @router.get("/projects")
    async def list_projects(ctx = Depends(require_scope("projects:read"))):
        ...
"""
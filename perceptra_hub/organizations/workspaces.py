"""
Personal workspaces: the organization a new account gets when it doesn't
join an existing one.
"""
import logging

from django.utils.text import slugify

from memberships.models import OrganizationMembership, Role
from organizations.models import Organization

logger = logging.getLogger(__name__)


def _free_slug(base: str) -> str:
    """`base`, or `base-2`, `base-3`, … — the first slug no organization uses."""
    slug, n = base, 1
    while Organization.objects.filter(slug=slug).exists():
        n += 1
        slug = f"{base}-{n}"
    return slug


def create_personal_workspace(user) -> Organization:
    """Create "<first name>'s Workspace" owned by this user (names may repeat; slugs don't)."""
    name = f"{user.first_name or user.username}'s Workspace"
    slug = _free_slug(f"{slugify(user.username) or 'user'}-workspace")
    workspace = Organization.objects.create(name=name, slug=slug, owner=user)

    owner_role, _ = Role.objects.get_or_create(name='owner', defaults={'description': 'Organization owner'})
    OrganizationMembership.objects.create(user=user, organization=workspace, role=owner_role)

    logger.info(f"Created personal organization for {user.username}")
    return workspace

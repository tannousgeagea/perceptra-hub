"""
Shared ProjectListItem serialization for the project list / detail endpoints.

Statistics are computed with correlated COUNT subqueries (one SQL query for the
whole list) instead of prefetching every image row of every project.
Lives outside queries/ because every module there is mounted as a router.
"""
import logging

from django.db.models import Count, IntegerField, OuterRef, Prefetch, QuerySet, Subquery
from django.db.models.functions import Coalesce

from annotations.models import Annotation, AnnotationGroup
from api.routers.projects.schemas import ProjectListItem, ProjectStatistics, UserBasicInfo
from memberships.models import ProjectMembership
from projects.models import Project, ProjectImage

logger = logging.getLogger(__name__)


def _count(queryset: QuerySet, group_by: str):
    """COUNT(*) of `queryset` per outer project, 0 when there are no rows."""
    counted = (
        queryset.order_by()
        .values(group_by)
        .annotate(c=Count("pk"))
        .values("c")
    )
    return Coalesce(Subquery(counted, output_field=IntegerField()), 0)


def with_list_data(queryset: QuerySet, user) -> QuerySet:
    """Annotate statistics and attach the user's own membership."""
    return queryset.select_related(
        "project_type", "visibility", "created_by", "updated_by",
    ).annotate(
        stat_images=_count(
            ProjectImage.objects.filter(project=OuterRef("pk"), is_active=True),
            "project",
        ),
        stat_annotations=_count(
            Annotation.objects.filter(
                project_image__project=OuterRef("pk"),
                project_image__is_active=True,
                is_active=True,
            ),
            "project_image__project",
        ),
        stat_groups=_count(AnnotationGroup.objects.filter(project=OuterRef("pk")), "project"),
    ).prefetch_related(
        Prefetch(
            "memberships",
            queryset=ProjectMembership.objects.filter(user=user).select_related("role"),
            to_attr="user_memberships",
        )
    )


def _user_info(user) -> UserBasicInfo | None:
    if not user:
        return None
    return UserBasicInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
    )


def _thumbnail(project: Project) -> str | None:
    """Explicit thumbnail, else a short-lived signed URL of the first image (never persisted)."""
    if project.thumbnail_url:
        return project.thumbnail_url
    first = (
        ProjectImage.objects.filter(project=project, is_active=True)
        .select_related("image", "image__storage_profile")
        .order_by("id")
        .first()
    )
    if not first or not first.image:
        return None
    try:
        return first.image.get_download_url(expiration=3600)
    except Exception as e:
        logger.warning(f"Failed to get thumbnail for project {project.id}: {e}")
        return None


def serialize_project(project: Project, user=None, fallback_role: str = "member") -> ProjectListItem:
    """
    Build a ProjectListItem. Uses the annotations from with_list_data() when
    present; otherwise (a single project fetched elsewhere) queries them.

    fallback_role is shown when the user isn't a project member, e.g. "admin"
    for an org admin, who has access to every project anyway.
    """
    if not hasattr(project, "stat_images"):
        project = with_list_data(Project.objects.filter(pk=project.pk), user).get()

    memberships = getattr(project, "user_memberships", None) or []
    user_role = memberships[0].role.name if memberships else fallback_role

    return ProjectListItem(
        id=project.id,
        project_id=str(project.project_id),
        name=project.name,
        description=project.description,
        thumbnail_url=_thumbnail(project),
        project_type_name=project.project_type.name,
        visibility_name=project.visibility.name,
        is_active=project.is_active,
        statistics=ProjectStatistics(
            total_images=project.stat_images,
            total_annotations=project.stat_annotations,
            annotation_groups=project.stat_groups,
        ),
        created_by=_user_info(project.created_by),
        updated_by=_user_info(project.updated_by),
        created_at=project.created_at.isoformat(),
        last_edited=project.last_edited.isoformat(),
        user_role=user_role,
    )

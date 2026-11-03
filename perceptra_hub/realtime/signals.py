"""
Model-level events: precise, per image. Coarser "something changed in this
project" events for bulk writes (queryset.update) come from
api.middleware.realtime_middleware.
"""
from functools import lru_cache
from typing import Optional

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from annotations.models import Annotation
from projects.models import Project, ProjectImage
from realtime.bus import publish_on_commit


@lru_cache(maxsize=1024)
def _project_uuid(project_pk: int) -> Optional[str]:
    # A project's uuid never changes, so this is safe to cache per process.
    value = Project.objects.filter(pk=project_pk).values_list("project_id", flat=True).first()
    return str(value) if value else None


@lru_cache(maxsize=8192)
def _project_pk_of_image(project_image_pk: int) -> Optional[int]:
    # Neither does the project a project-image belongs to.
    return ProjectImage.objects.filter(pk=project_image_pk).values_list("project_id", flat=True).first()


def _annotation_event(instance: Annotation, action: str) -> None:
    project_pk = _project_pk_of_image(instance.project_image_id)
    if project_pk is None:
        return
    publish_on_commit(
        _project_uuid(project_pk),
        "annotation.changed",
        {
            "project_image_id": instance.project_image_id,
            "annotation_uid": instance.annotation_uid,
            "action": action,
        },
        actor=instance.updated_by_id or instance.created_by_id,
    )


@receiver(post_save, sender=Annotation, dispatch_uid="realtime_annotation_saved")
def annotation_saved(sender, instance, created, **kwargs):
    if kwargs.get("raw"):
        return
    action = "deleted" if instance.is_deleted else ("created" if created else "updated")
    _annotation_event(instance, action)


@receiver(post_delete, sender=Annotation, dispatch_uid="realtime_annotation_deleted")
def annotation_deleted(sender, instance, **kwargs):
    _annotation_event(instance, "deleted")


@receiver(post_save, sender=ProjectImage, dispatch_uid="realtime_project_image_saved")
def project_image_saved(sender, instance, created, **kwargs):
    if kwargs.get("raw") or created:
        return  # new images arrive in bulk; the middleware event covers them
    publish_on_commit(
        _project_uuid(instance.project_id),
        "image.changed",
        {
            "project_image_id": instance.pk,
            "status": instance.status,
            "marked_as_null": getattr(instance, "marked_as_null", None),
        },
        actor=getattr(instance, "reviewed_by_id", None),
    )

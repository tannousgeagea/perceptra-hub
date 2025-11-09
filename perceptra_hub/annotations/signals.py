from django.db.models.signals import post_save
from django.dispatch import receiver
from annotations.models import Annotation, AnnotationAudit
from api.tasks import annotation_audit

@receiver(post_save, sender=Annotation)
def update_audit_on_save(sender, instance, created, **kwargs):
    """Auto-update audit with edit tolerance."""
    annotation_audit.compute_annotation_audit.apply_async(
        args=(
            instance.id,
            created,
        ),
    )
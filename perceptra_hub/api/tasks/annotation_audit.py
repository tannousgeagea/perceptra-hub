# tasks.py
from celery import shared_task
import numpy as np
from datetime import datetime
from django.db import transaction

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

@shared_task(
    bind=True, 
    name='annotation_audit:compute_annotation_audit'
)
def compute_annotation_audit(self, instance_id: int, created: bool, **kwargs):
    """
    Compute audit based on annotation changes.
    
    Logic:
    - TP: Prediction still active, not edited
    - FP: Prediction deleted OR edited (class/bbox changed)
    - FN: Manual annotation (human added, model missed)
    """
    from annotations.models import Annotation, AnnotationAudit
    
    status = 'N/A'
    try:
        instance = Annotation.objects.get(id=instance_id)
    except Annotation.DoesNotExist as e:
        return {
            "status": "failed",
            "detail": f"Invalid Reference: {str(e)}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
    if instance.annotation_source == 'prediction':
        
        edit_mag = instance.edit_magnitude or 0
        edit_type = instance.edit_type or Annotation.EditType.NONE
        
        # Determine audit status
        if not instance.is_active:
            # Deleted
            status = 'FP'
            edit_type = 'deleted'
        
        elif instance.version > 1:
            # Check if class changed
            if instance.edit_type == Annotation.EditType.CLASS_CHANGE:
                status = 'FP'
                edit_type = 'class_change'
            
            # Minor adjustment (< 10% change)
            elif edit_type == Annotation.EditType.MINOR:
                status = 'TP'  # Still True Positive
                edit_type = 'minor'
            
            # Major change (> 30% change)
            elif edit_type == Annotation.EditType.MAJOR:
                status = 'FP'
                edit_type = 'major'
            
            # Moderate change (10-30%)
            else:
                status = 'TP'  # Accept as refinement
                edit_type = 'minor'
        
        else:
            # Not edited at all
            status = 'TP'
            edit_type = 'none'
        
        AnnotationAudit.objects.update_or_create(
            annotation=instance,
            defaults={
                'evaluation_status': status,
                'was_edited': instance.version > 1,
                'edit_magnitude': edit_mag,
                'edit_type': edit_type
            }
        )
    
    elif created and instance.annotation_source == 'manual':
        # New manual annotation = model missed it
        status = "FN"
        AnnotationAudit.objects.update_or_create(
            annotation=instance,
            defaults={
                'evaluation_status': 'FN',
                'was_edited': False
            }
        )
    
    return {
        'id': instance.id,
        'status': 'completed',
        'evaluation_status': status,
        "annotation_source": instance.annotation_source,
    }


@shared_task(name='annotations.compute_project_audit')
def compute_project_audit(project_id: int, iou_threshold: float = 0.5):
    """Compute audit for entire project in batches."""
    from projects.models import Project, ProjectImage
    
    project = Project.objects.get(id=project_id)
    project_images = ProjectImage.objects.filter(
        project=project,
        is_active=True,
        annotated=True
    ).values_list('id', flat=True)
    
    total = len(project_images)
    
    # Process in batches
    for project_image_id in project_images:
        compute_annotation_audit.delay(project_image_id, iou_threshold)
    
    return {'total_images': total, 'status': 'queued'}
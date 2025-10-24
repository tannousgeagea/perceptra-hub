import django
django.setup()
import uuid
from typing import List, Dict
from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationGroup,
)

def save_annotations(
    data:List[Dict],
    annotation_type,
    project_image,
):
    success = False
    results = []
    try:
        for i, coords in enumerate(data):
            annotation_class = AnnotationClass.objects.filter(
                class_id=coords[0],
                annotation_group__project=project_image.project
            )
            
            if annotation_class:
                annotation_class = annotation_class.first()
            else:
                annotation_group = AnnotationGroup.objects.get(
                    project=project_image.project,
                )
                annotation_class = AnnotationClass.objects.create(
                    class_id=coords[0],
                    annotation_group=annotation_group
                )
            
            annotation, created = Annotation.objects.get_or_create(
                project_image=project_image,
                annotation_type=annotation_type,
                annotation_class=annotation_class,
                data=coords[1:5],
                annotation_uid=f"{str(uuid.uuid4())}",
                annotation_source="prediction",
                confidence=coords[-1] if len(coords) == 6 else None,
            )
            
            if created:
                results.append(
                    {
                        "id": project_image.image.image_id,
                        "status": "created",
                        "status_description": f"New annotation ({annotation}: {annotation.data}) has been created !"
                    }
                )
            else:
                results.append(
                    {                 
                        "id": project_image.image.image_id,   
                        "status": "failed",
                        "status_description": f"Annotation ({annotation}: {annotation.data}) already exists"
                    }
                )
                
        success = True
        
    except Exception as err:
        results = {
            'status': 'failed',
            'reason': str(err)
        }
        
    return success, results
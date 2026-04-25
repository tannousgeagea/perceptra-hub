"""
Auto-annotation Celery task.

Runs a deployed model against a set of project images and writes the
predictions as Annotation records (annotation_source='prediction').
These appear immediately in the annotation tool for human review.
"""

import logging
import traceback

from celery import shared_task
from django.db import transaction

from common_utils.inference.utils import run_inference
from common_utils.progress.core import track_progress

logger = logging.getLogger(__name__)


@shared_task(bind=True, queue='auto_annotate', max_retries=2, default_retry_delay=30)
def auto_annotate_images(
    self,
    project_id: int,
    image_ids: list,
    model_version_id: str,
    confidence_threshold: float = 0.25,
    task_id: str = None,
):
    """
    Run model inference on project images and persist predictions as annotations.

    For each image_id:
    1. Fetch Image + ProjectImage
    2. Call run_inference() → list of prediction dicts
    3. Look up AnnotationClass by class_label in the project's AnnotationGroup
       (skip unknown classes — never auto-create classes)
    4. Bulk-create Annotation records with annotation_source='prediction'
    5. Mark ProjectImage.annotated = True if at least one annotation created
    6. Report progress via track_progress
    """
    from annotations.models import Annotation, AnnotationClass, AnnotationGroup, AnnotationType
    from images.models import Image
    from ml_models.models import ModelVersion
    from projects.models import Project, ProjectImage

    if task_id:
        track_progress(task_id=task_id, percentage=0, message="Starting auto-annotation...", status="running")

    try:
        project = Project.objects.get(id=project_id)
        model_version = ModelVersion.objects.select_related(
            "model__task", "storage_profile"
        ).get(version_id=model_version_id)

        # Build class name → AnnotationClass lookup for this project
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        if annotation_group is None:
            logger.warning("Project %s has no AnnotationGroup — auto-annotation skipped", project_id)
            if task_id:
                track_progress(task_id=task_id, percentage=100, message="No annotation classes found", status="completed")
            return {"annotated": 0, "skipped": len(image_ids), "error": "no_annotation_group"}

        class_lookup: dict[str, AnnotationClass] = {
            cls.name.lower(): cls
            for cls in AnnotationClass.objects.filter(annotation_group=annotation_group)
        }

        bbox_type = AnnotationType.objects.filter(name__iexact="Bounding Box").first()
        if bbox_type is None:
            bbox_type = AnnotationType.objects.first()

        total = len(image_ids)
        annotated_count = 0
        skipped_count = 0

        for idx, image_id in enumerate(image_ids):
            try:
                image = Image.objects.get(id=image_id)
                project_image = ProjectImage.objects.get(project=project, image=image)
            except (Image.DoesNotExist, ProjectImage.DoesNotExist):
                logger.warning("Image %s not found in project %s — skipping", image_id, project_id)
                skipped_count += 1
                continue

            try:
                predictions = run_inference(
                    image=image.storage_key,
                    model_version_id=model_version_id,
                    confidence_threshold=confidence_threshold,
                )
            except Exception as e:
                logger.warning("Inference failed for image %s: %s", image_id, e)
                skipped_count += 1
                continue

            if not predictions:
                _progress(task_id, idx + 1, total)
                continue

            new_annotations = []
            for pred in predictions:
                class_label = (pred.get("class_label") or "").lower()
                ann_class = class_lookup.get(class_label)
                if ann_class is None:
                    logger.debug(
                        "Unknown class '%s' from model — skipping prediction", class_label
                    )
                    continue

                bbox = pred.get("bbox", {})
                # Normalise bbox to list [xmin, ymin, xmax, ymax]
                if isinstance(bbox, dict):
                    data = [
                        bbox.get("x_min", 0.0),
                        bbox.get("y_min", 0.0),
                        bbox.get("x_max", 0.0),
                        bbox.get("y_max", 0.0),
                    ]
                elif isinstance(bbox, list) and len(bbox) == 4:
                    data = bbox
                else:
                    continue

                new_annotations.append(
                    Annotation(
                        project_image=project_image,
                        annotation_type=bbox_type,
                        annotation_class=ann_class,
                        annotation_source="prediction",
                        model_version=str(model_version_id),
                        confidence=pred.get("confidence", 0.0),
                        data=data,
                        reviewed=False,
                        is_active=True,
                    )
                )

            if new_annotations:
                with transaction.atomic():
                    Annotation.objects.bulk_create(new_annotations)
                    if not project_image.annotated:
                        project_image.annotated = True
                        project_image.save(update_fields=["annotated"])
                annotated_count += 1

            _progress(task_id, idx + 1, total)

        logger.info(
            "Auto-annotation complete: project=%s model=%s annotated=%d skipped=%d",
            project_id, model_version_id, annotated_count, skipped_count,
        )

        if task_id:
            track_progress(
                task_id=task_id,
                percentage=100,
                message=f"Done: {annotated_count} images annotated",
                status="completed",
            )

        return {"annotated": annotated_count, "skipped": skipped_count, "total": total}

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("auto_annotate_images failed: %s\n%s", e, error_trace)
        if task_id:
            track_progress(task_id=task_id, percentage=0, message=f"Failed: {e}", status="failed")
        raise self.retry(exc=e)


def _progress(task_id: str | None, done: int, total: int) -> None:
    if not task_id or total == 0:
        return
    pct = round((done / total) * 100)
    track_progress(task_id=task_id, percentage=pct, message=f"Processed {done}/{total} images", status="running")

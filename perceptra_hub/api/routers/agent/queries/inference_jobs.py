"""
Agent-facing endpoints for on-premise inference jobs.

GET  /api/v1/agents/poll/inference-job            → next pending job or empty
POST /api/v1/agents/inference-jobs/{job_id}/results → agent posts predictions
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from asgiref.sync import sync_to_async
from django.utils import timezone

from compute.models import Agent, InferenceJob
from api.routers.agent.utils import authenticate_agent_from_header

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents")


class InferenceJobPayload(BaseModel):
    job_id: Optional[str] = None
    model_version_id: Optional[str] = None
    project_id: Optional[str] = None
    image_ids: List[int] = []
    confidence_threshold: float = 0.25
    onnx_presigned_url: Optional[str] = None


class PredictionResult(BaseModel):
    image_id: int
    predictions: List[dict]


class InferenceResultsRequest(BaseModel):
    predictions: List[PredictionResult] = []
    annotations_created: int = 0
    error: Optional[str] = None


@router.get("/poll/inference-job", response_model=InferenceJobPayload)
async def poll_for_inference_job(
    agent: Agent = Depends(authenticate_agent_from_header),
):
    """Agent polls for the next pending inference job for its organization."""

    @sync_to_async
    def _poll():
        job = (
            InferenceJob.objects
            .filter(organization=agent.organization, status='pending')
            .select_related('model_version', 'project')
            .order_by('created_at')
            .first()
        )
        if not job:
            return None

        _maybe_refresh_onnx_url(job)

        job.agent = agent
        job.status = 'running'
        job.started_at = timezone.now()
        job.save(update_fields=['agent', 'status', 'started_at'])

        return InferenceJobPayload(
            job_id=job.job_id,
            model_version_id=str(job.model_version.version_id),
            project_id=str(job.project.project_id),
            image_ids=job.image_ids,
            confidence_threshold=job.confidence_threshold,
            onnx_presigned_url=job.onnx_presigned_url,
        )

    result = await _poll()
    return result or InferenceJobPayload()


@router.post("/inference-jobs/{job_id}/results")
async def submit_inference_results(
    job_id: str,
    body: InferenceResultsRequest,
    agent: Agent = Depends(authenticate_agent_from_header),
):
    """Agent posts inference results; platform bulk-creates Annotation records."""

    @sync_to_async
    def _handle():
        try:
            job = InferenceJob.objects.select_related(
                'model_version', 'project', 'organization'
            ).get(job_id=job_id, agent=agent)
        except InferenceJob.DoesNotExist:
            raise HTTPException(status_code=404, detail="Inference job not found")

        if job.status not in ('running', 'pending'):
            raise HTTPException(status_code=400, detail=f"Job is already {job.status}")

        if body.error:
            job.status = 'failed'
            job.error_message = body.error
            job.completed_at = timezone.now()
            job.save(update_fields=['status', 'error_message', 'completed_at'])
            logger.error("Inference job %s failed: %s", job_id, body.error)
            return {"status": "failed"}

        from annotations.models import Annotation
        from projects.models import ProjectImage
        from images.models import Image
        from django.db import transaction

        # Build class_id → AnnotationClass lookup for the project
        class_id_lookup: dict = {}
        try:
            from annotations.models import AnnotationClass, AnnotationGroup
            group = AnnotationGroup.objects.filter(project=job.project).first()
            if group:
                for cls in AnnotationClass.objects.filter(annotation_group=group):
                    class_id_lookup[cls.class_id] = cls
                    class_id_lookup[cls.name.lower()] = cls
        except Exception as e:
            logger.warning("Could not build class lookup: %s", e)

        annotations_created = 0
        with transaction.atomic():
            for pred_result in body.predictions:
                try:
                    image = Image.objects.get(pk=pred_result.image_id)
                    project_image = ProjectImage.objects.filter(
                        project=job.project, image=image
                    ).first()
                    if not project_image:
                        continue

                    new_annotations = []
                    for pred in pred_result.predictions:
                        label = pred.get('class_label', '')
                        class_id = pred.get('class_id')
                        ann_class = class_id_lookup.get(class_id) or class_id_lookup.get(
                            label.lower() if label else ''
                        )
                        if not ann_class:
                            continue
                        bbox = pred.get('bbox', [])
                        new_annotations.append(Annotation(
                            project_image=project_image,
                            annotation_class=ann_class,
                            annotation_source='prediction',
                            model_version=job.model_version,
                            confidence=pred.get('confidence', 0.0),
                            annotation_type='BoundingBox',
                            data=bbox,
                            is_active=True,
                            is_deleted=False,
                        ))
                    if new_annotations:
                        Annotation.objects.bulk_create(new_annotations, ignore_conflicts=True)
                        annotations_created += len(new_annotations)
                        project_image.annotated = True
                        project_image.save(update_fields=['annotated'])
                except Exception as e:
                    logger.warning("Failed to create annotations for image %s: %s",
                                   pred_result.image_id, e)

        job.status = 'completed'
        job.result_summary = {'annotations_created': annotations_created}
        job.completed_at = timezone.now()
        job.save(update_fields=['status', 'result_summary', 'completed_at'])

        logger.info("Inference job %s completed: %d annotations created", job_id, annotations_created)
        return {"status": "completed", "annotations_created": annotations_created}

    return await _handle()


def _maybe_refresh_onnx_url(job: InferenceJob) -> None:
    if job.onnx_url_expires_at and job.onnx_url_expires_at > timezone.now():
        return
    try:
        mv = job.model_version
        if not getattr(mv, 'onnx_model_key', None):
            return
        storage_profile = getattr(mv, 'storage_profile', None)
        if not storage_profile:
            return
        from storage.services import get_storage_adapter_for_profile
        from datetime import timedelta
        adapter = get_storage_adapter_for_profile(storage_profile)
        result = adapter.generate_presigned_url(mv.onnx_model_key, expiration=3600, method='GET')
        job.onnx_presigned_url = result.url
        job.onnx_url_expires_at = timezone.now() + timedelta(seconds=3500)
        job.save(update_fields=['onnx_presigned_url', 'onnx_url_expires_at'])
    except Exception as e:
        logger.warning("Failed to refresh ONNX presigned URL: %s", e)

"""
Champion / challenger evaluation Celery task.

Runs automatically after every training job completes:
  1. Deploys the challenger (new version) to 'staging'
  2. Validates it against the val split of its dataset version
  3. Computes EvaluationReport metrics (reuses evaluation/pipeline.py dataclasses)
  4. Compares against the champion (current production version)
  5. Emits a recommendation and optionally auto-promotes
"""

import logging
import traceback

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    queue='evaluation',
    max_retries=2,
    default_retry_delay=60,
    name='api.tasks.champion_challenger.run_champion_challenger_evaluation',
)
def run_champion_challenger_evaluation(self, evaluation_id: str, task_id: str = None):
    """
    Execute champion/challenger evaluation for a ModelEvaluation record.

    Called ~30 seconds after training completes (see train_model.py step 10).
    """
    from inferences.models import ModelEvaluation
    from common_utils.progress.core import track_progress

    if task_id:
        track_progress(task_id=task_id, percentage=0, message="Starting evaluation...", status="running")

    try:
        evaluation = ModelEvaluation.objects.select_related(
            "challenger__model__organization",
            "champion__model__organization",
            "dataset_version__project",
        ).get(evaluation_id=evaluation_id)
    except ModelEvaluation.DoesNotExist:
        logger.error("ModelEvaluation %s not found", evaluation_id)
        return

    evaluation.status = "running"
    evaluation.save(update_fields=["status"])

    try:
        challenger_metrics = _evaluate_model_version(evaluation.challenger, evaluation.dataset_version)

        champion_metrics = {}
        if evaluation.champion is not None:
            champion_metrics = _get_or_compute_champion_metrics(
                evaluation.champion, evaluation.dataset_version
            )

        evaluation.challenger_metrics = challenger_metrics
        evaluation.champion_metrics = champion_metrics

        # Compute delta on primary metric
        primary = evaluation.primary_metric
        challenger_score = challenger_metrics.get(primary, 0.0) or 0.0
        champion_score = champion_metrics.get(primary, 0.0) or 0.0

        delta = challenger_score - champion_score
        evaluation.improvement_delta = delta

        # Recommendation
        if evaluation.champion is None:
            # No previous production version — always promote
            recommendation = "promote"
        elif delta >= evaluation.auto_promote_threshold:
            recommendation = "promote"
        elif delta < -0.01:
            recommendation = "keep_champion"
        else:
            recommendation = "inconclusive"

        evaluation.recommendation = recommendation

        if task_id:
            track_progress(
                task_id=task_id, percentage=80,
                message=f"Recommendation: {recommendation} (delta={delta:+.4f})",
                status="running",
            )

        # Auto-promote if recommended
        if recommendation == "promote":
            _promote_challenger(evaluation)

        evaluation.status = "completed"
        evaluation.completed_at = timezone.now()
        evaluation.save()

        logger.info(
            "Evaluation %s complete: recommendation=%s delta=%+.4f auto_promoted=%s",
            evaluation_id, recommendation, delta, evaluation.auto_promoted,
        )

        if task_id:
            track_progress(task_id=task_id, percentage=100, message="Evaluation complete", status="completed")

        return {
            "evaluation_id": evaluation_id,
            "recommendation": recommendation,
            "improvement_delta": delta,
            "auto_promoted": evaluation.auto_promoted,
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("Evaluation %s failed: %s\n%s", evaluation_id, e, error_trace)
        evaluation.status = "failed"
        evaluation.error_message = str(e)
        evaluation.save(update_fields=["status", "error_message"])

        if task_id:
            track_progress(task_id=task_id, percentage=0, message=f"Failed: {e}", status="failed")

        raise self.retry(exc=e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_model_version(model_version, dataset_version) -> dict:
    """
    Run inference on val-split images and compute metrics.

    Reuses the stored PredictionImageResult records if they exist,
    otherwise calls run_inference() to generate them.
    """
    from inferences.models import PredictionImageResult, PredictionOverlay
    from projects.models import VersionImage
    from common_utils.inference.utils import run_inference
    from django.db import transaction

    val_images = VersionImage.objects.filter(
        version=dataset_version,
        split__in=["val", "valid"],
    ).select_related("project_image__image")

    tp = fp = fn = 0
    all_predictions = []

    for vi in val_images:
        image = vi.project_image.image
        result, created = PredictionImageResult.objects.get_or_create(
            model_version=model_version,
            dataset_version=dataset_version,
            image=image,
        )

        if created:
            # Need to run inference
            try:
                preds = run_inference(
                    image=image.storage_key,
                    model_version_id=str(model_version.version_id),
                )
            except Exception as e:
                logger.warning("Inference failed for image %s: %s", image.id, e)
                result.delete()
                continue

            overlays = []
            for pred in preds:
                bbox = pred.get("bbox", {})
                if isinstance(bbox, dict):
                    bbox_list = [
                        bbox.get("x_min", 0.0), bbox.get("y_min", 0.0),
                        bbox.get("x_max", 0.0), bbox.get("y_max", 0.0),
                    ]
                elif isinstance(bbox, list):
                    bbox_list = bbox
                else:
                    continue
                overlays.append(PredictionOverlay(
                    prediction_result=result,
                    class_label=pred.get("class_label", ""),
                    confidence=pred.get("confidence", 0.0),
                    bbox=bbox_list,
                ))
            with transaction.atomic():
                PredictionOverlay.objects.bulk_create(overlays)

        # Compare predictions vs. ground truth annotations
        from annotations.models import Annotation
        gt_annotations = Annotation.objects.filter(
            project_image=vi.project_image,
            annotation_source="manual",
            is_active=True,
            is_deleted=False,
        )
        overlays_qs = PredictionOverlay.objects.filter(prediction_result=result)

        image_tp, image_fp, image_fn = _match_predictions_to_gt(
            list(overlays_qs), list(gt_annotations)
        )
        tp += image_tp
        fp += image_fp
        fn += image_fn

    total_gt = tp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "total_val_images": val_images.count(),
    }


def _get_or_compute_champion_metrics(champion, dataset_version) -> dict:
    """Return stored metrics if the champion was already validated; otherwise compute."""
    if champion.metrics:
        # Prefer stored training metrics as a fast approximation
        stored = champion.metrics
        if "f1_score" in stored or "precision" in stored:
            return stored
    return _evaluate_model_version(champion, dataset_version)


def _match_predictions_to_gt(overlays, gt_annotations, iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """
    Greedy IoU matching between predictions and ground-truth boxes.
    Returns (tp, fp, fn).
    """
    if not gt_annotations:
        return 0, len(overlays), 0
    if not overlays:
        return 0, 0, len(gt_annotations)

    matched_gt = set()
    tp = 0
    for overlay in overlays:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_annotations):
            if idx in matched_gt:
                continue
            iou = _bbox_iou(overlay.bbox, gt.data)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(overlays) - tp
    fn = len(gt_annotations) - len(matched_gt)
    return tp, fp, fn


def _bbox_iou(pred_bbox, gt_bbox) -> float:
    """
    Compute IoU between two normalized [xmin, ymin, xmax, ymax] bboxes.
    Handles both list and dict formats.
    """
    def _to_list(bbox):
        if isinstance(bbox, dict):
            return [bbox.get("x_min", 0), bbox.get("y_min", 0),
                    bbox.get("x_max", 0), bbox.get("y_max", 0)]
        return bbox

    try:
        p = _to_list(pred_bbox)
        g = _to_list(gt_bbox)
        xi1 = max(p[0], g[0]); yi1 = max(p[1], g[1])
        xi2 = min(p[2], g[2]); yi2 = min(p[3], g[3])
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        area_p = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
        area_g = max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1])
        union = area_p + area_g - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _promote_challenger(evaluation) -> None:
    """Deploy challenger to production and retire champion."""
    from inferences.orchestrator import InferenceOrchestrator

    try:
        orchestrator = InferenceOrchestrator(evaluation.challenger)
        orchestrator.deploy(target_env="production")
        evaluation.auto_promoted = True
        logger.info("Auto-promoted challenger %s to production", evaluation.challenger.version_id)
    except Exception as e:
        logger.warning("Auto-promotion failed for %s: %s", evaluation.challenger.version_id, e)

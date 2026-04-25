"""
Retraining policy evaluator.

Called hourly by the evaluate_retraining_policies Celery Beat task.
For each active RetrainingPolicy, checks whether the trigger condition
is met and, if so, creates a new DatasetVersion snapshot and submits
a training job via the existing TrainingOrchestrator.
"""

import logging
import uuid
from datetime import timedelta
from typing import Optional

from django.utils import timezone

from training.models import RetrainingPolicy

logger = logging.getLogger(__name__)


class RetrainingService:

    def evaluate_all_policies(self) -> list[dict]:
        """
        Evaluate all active retraining policies.

        Returns a list of summary dicts for each policy that was triggered.
        """
        policies = RetrainingPolicy.objects.filter(is_active=True).select_related(
            "model__project", "compute_profile"
        )

        triggered = []
        for policy in policies:
            try:
                result = self._evaluate_policy(policy)
                if result:
                    triggered.append(result)
            except Exception as e:
                logger.error("Error evaluating policy %s: %s", policy.policy_id, e)

        logger.info("Retraining evaluation complete: %d/%d policies triggered", len(triggered), len(policies))
        return triggered

    def _evaluate_policy(self, policy: RetrainingPolicy) -> Optional[dict]:
        # Cooldown check
        if policy.last_triggered_at:
            elapsed_hours = (timezone.now() - policy.last_triggered_at).total_seconds() / 3600
            if elapsed_hours < policy.min_hours_between_runs:
                return None

        triggered = False
        reason = ""

        if policy.trigger_type in ("annotation_count", "combined"):
            if self._check_annotation_count(policy):
                triggered = True
                reason = "annotation_count_threshold_reached"

        if not triggered and policy.trigger_type in ("correction_rate", "combined"):
            if self._check_correction_rate(policy):
                triggered = True
                reason = "correction_rate_threshold_reached"

        if not triggered and policy.trigger_type in ("time_elapsed", "combined"):
            if self._check_time_elapsed(policy):
                triggered = True
                reason = "max_days_since_training_exceeded"

        if not triggered:
            return None

        logger.info("Policy %s triggered: reason=%s", policy.policy_id, reason)
        return self._trigger_retraining(policy, reason)

    # ------------------------------------------------------------------
    # Trigger condition checks
    # ------------------------------------------------------------------

    def _check_annotation_count(self, policy: RetrainingPolicy) -> bool:
        from annotations.models import Annotation

        cutoff = (
            policy.last_triggered_at
            or (timezone.now() - timedelta(days=policy.lookback_days))
        )

        count = Annotation.objects.filter(
            project_image__project=policy.model.project,
            reviewed=True,
            is_deleted=False,
            reviewed_at__gte=cutoff,
        ).count()

        logger.debug("Policy %s: new_reviewed_annotations=%d threshold=%d",
                     policy.policy_id, count, policy.min_new_annotations)
        return count >= policy.min_new_annotations

    def _check_correction_rate(self, policy: RetrainingPolicy) -> bool:
        if policy.min_correction_rate is None:
            return False

        from annotations.models import AnnotationAudit

        latest_version = policy.model.get_latest_version()
        if latest_version is None:
            return False

        cutoff = timezone.now() - timedelta(days=policy.lookback_days)
        audits = AnnotationAudit.objects.filter(
            annotation__project_image__project=policy.model.project,
            annotation__annotation_source="prediction",
            annotation__reviewed_at__gte=cutoff,
        )
        total = audits.count()
        if total == 0:
            return False

        errors = audits.filter(evaluation_status__in=["FP", "FN"]).count()
        rate = errors / total
        logger.debug("Policy %s: correction_rate=%.3f threshold=%.3f",
                     policy.policy_id, rate, policy.min_correction_rate)
        return rate >= policy.min_correction_rate

    def _check_time_elapsed(self, policy: RetrainingPolicy) -> bool:
        if policy.max_days_since_training is None:
            return False

        from ml_models.models import ModelVersion
        latest = ModelVersion.objects.filter(
            model=policy.model,
            status="trained",
        ).order_by("-created_at").first()

        if latest is None:
            return True  # Never trained — retrain now

        age_days = (timezone.now() - latest.created_at).days
        logger.debug("Policy %s: age_days=%d max_days=%d",
                     policy.policy_id, age_days, policy.max_days_since_training)
        return age_days >= policy.max_days_since_training

    # ------------------------------------------------------------------
    # Trigger action
    # ------------------------------------------------------------------

    def _trigger_retraining(self, policy: RetrainingPolicy, reason: str) -> dict:
        from django.db import transaction
        from ml_models.models import ModelVersion
        from projects.models import ProjectImage, Version, VersionImage
        from training.models import TrainingSession
        from training.orchestrator import TrainingOrchestrator

        model = policy.model
        project = model.project
        now = timezone.now()

        with transaction.atomic():
            # 1. Create a new dataset version snapshot
            dataset_version = None
            if policy.auto_create_dataset_version:
                # Count splits from latest existing version or use 80/10/10
                dataset_version = Version.objects.create(
                    version_id=str(uuid.uuid4()),
                    project=project,
                    version_number=self._next_version_number(project),
                    version_name=f"auto-{now.strftime('%Y%m%d-%H%M')}",
                    description=f"Auto-generated by retraining policy ({reason})",
                    export_status="PENDING",
                )

                # Add all finalized project images with a simple 80/10/10 split
                images = list(
                    ProjectImage.objects.filter(
                        project=project, is_active=True, annotated=True
                    ).order_by("id")
                )
                n = len(images)
                val_start = int(n * 0.8)
                test_start = int(n * 0.9)

                version_images = []
                for i, pi in enumerate(images):
                    if i < val_start:
                        split = "train"
                    elif i < test_start:
                        split = "val"
                    else:
                        split = "test"
                    version_images.append(
                        VersionImage(version=dataset_version, project_image=pi, split=split)
                    )
                VersionImage.objects.bulk_create(version_images, ignore_conflicts=True)
                dataset_version.total_images = n
                dataset_version.save(update_fields=["total_images"])

            # 2. Create a new ModelVersion (child of current production)
            current_production = model.get_production_version()
            latest_trained = (
                ModelVersion.objects.filter(model=model, status="trained")
                .order_by("-version_number")
                .first()
            )
            parent = current_production or latest_trained

            new_version_number = (
                ModelVersion.objects.filter(model=model)
                .order_by("-version_number")
                .values_list("version_number", flat=True)
                .first()
                or 0
            ) + 1

            storage_profile = model.project.organization.storage_profiles.filter(
                is_default=True
            ).first()

            model_version = ModelVersion.objects.create(
                version_id=str(uuid.uuid4()),
                model=model,
                dataset_version=dataset_version,
                parent_version=parent,
                version_number=new_version_number,
                version_name=f"auto-{now.strftime('%Y%m%d-%H%M')}",
                status="draft",
                storage_profile=storage_profile,
                config=model.default_config or {},
            )

            # 3. Create TrainingSession
            training_session = TrainingSession.objects.create(
                session_id=str(uuid.uuid4()),
                model_version=model_version,
                storage_profile=storage_profile,
                config=model.default_config or {},
                status="pending",
            )

            # 4. Submit via TrainingOrchestrator
            if policy.auto_submit_training:
                compute_profile_id = (
                    policy.compute_profile.profile_id if policy.compute_profile else None
                )
                try:
                    orchestrator = TrainingOrchestrator(model_version)
                    orchestrator.submit_training(
                        training_session=training_session,
                        compute_profile_id=compute_profile_id,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to submit training for policy %s: %s",
                        policy.policy_id, e,
                    )

            # 5. Update policy cooldown
            policy.last_triggered_at = now
            policy.save(update_fields=["last_triggered_at"])

        logger.info(
            "Retraining triggered: policy=%s model=%s new_version=%s reason=%s",
            policy.policy_id, model.name, model_version.version_id, reason,
        )
        return {
            "policy_id": policy.policy_id,
            "model_id": model.model_id,
            "model_version_id": str(model_version.version_id),
            "training_session_id": training_session.session_id,
            "reason": reason,
        }

    @staticmethod
    def _next_version_number(project) -> int:
        from projects.models import Version
        last = Version.objects.filter(project=project).order_by("-version_number").first()
        return (last.version_number + 1) if last else 1

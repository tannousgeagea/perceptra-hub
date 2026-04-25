"""
Inference deployment orchestrator.

Mirrors training/orchestrator.py: manages the lifecycle of deploying
a trained ModelVersion to the perceptra-inference service so it can
serve predictions.
"""

import logging
import os
from typing import Optional

import requests
from django.db import transaction
from django.utils import timezone

from ml_models.models import ModelVersion

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """
    Orchestrates deployment of a ModelVersion to the inference service.

    Responsibilities:
    - Validate the model artifact exists in storage
    - Generate a presigned download URL for the ONNX artifact
    - Call the inference service /v1/models/load endpoint
    - Update ModelVersion.deployment_status
    - Record InferenceDeployment audit entry
    """

    def __init__(self, model_version: ModelVersion) -> None:
        self.model_version = model_version
        self.organization = model_version.model.organization

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy(
        self,
        target_env: str = "production",
        user=None,
        class_names: Optional[list[str]] = None,
    ) -> "InferenceDeployment":
        """
        Deploy model_version to the inference service.

        Steps:
        1. Validate model is trained and has an ONNX artifact
        2. Generate presigned URL for onnx_model_key
        3. POST /v1/models/load to the inference service
        4. Update ModelVersion.deployment_status
        5. If target_env == 'production': retire previous production version
        6. Record InferenceDeployment

        Returns:
            InferenceDeployment record

        Raises:
            ValueError: model not ready for deployment
            RuntimeError: inference service unreachable or rejected the model
        """
        from inferences.models import InferenceDeployment

        self._validate_for_deployment()

        storage_url = self._get_onnx_presigned_url()
        if class_names is None:
            class_names = self._get_class_names()

        task = self._get_task_name()
        inference_url = self._get_inference_service_url()
        version_id = str(self.model_version.version_id)

        logger.info(
            "Deploying model_version=%s target=%s to %s",
            version_id, target_env, inference_url,
        )

        self._call_load_endpoint(inference_url, version_id, storage_url, task, class_names)

        with transaction.atomic():
            # Retire any currently active deployment for this version
            InferenceDeployment.objects.filter(
                model_version=self.model_version, is_active=True
            ).update(is_active=False, undeployed_at=timezone.now())

            # If promoting to production, retire the previous production version
            if target_env == "production":
                self._retire_previous_production(InferenceDeployment)
                self.model_version.deployment_status = "production"
                if user:
                    self.model_version.deployed_by = user
                    self.model_version.deployed_at = timezone.now()
            else:
                self.model_version.deployment_status = "staging"

            self.model_version.save(update_fields=["deployment_status", "deployed_at", "deployed_by"])

            deployment = InferenceDeployment.objects.create(
                model_version=self.model_version,
                target_env=target_env,
                inference_service_url=inference_url,
                deployed_by=user,
                is_active=True,
            )

        logger.info("Deployment complete: version=%s env=%s", version_id, target_env)
        return deployment

    def undeploy(self, user=None) -> None:
        """
        Unload the model from the inference service and mark it retired.

        Raises:
            RuntimeError: if the inference service call fails
        """
        from inferences.models import InferenceDeployment

        inference_url = self._get_inference_service_url()
        version_id = str(self.model_version.version_id)

        logger.info("Undeploying model_version=%s from %s", version_id, inference_url)

        try:
            resp = requests.delete(
                f"{inference_url}/v1/models/{version_id}",
                timeout=30,
            )
            if resp.status_code not in (200, 404):
                raise RuntimeError(
                    f"Inference service returned {resp.status_code}: {resp.text}"
                )
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach inference service: {e}") from e

        with transaction.atomic():
            InferenceDeployment.objects.filter(
                model_version=self.model_version, is_active=True
            ).update(is_active=False, undeployed_at=timezone.now())

            self.model_version.deployment_status = "retired"
            self.model_version.save(update_fields=["deployment_status"])

        logger.info("Undeployment complete: version=%s", version_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_for_deployment(self) -> None:
        mv = self.model_version
        if mv.status != "trained":
            raise ValueError(
                f"ModelVersion {mv.version_id} status is '{mv.status}', expected 'trained'."
            )
        if not mv.onnx_model_key:
            raise ValueError(
                f"ModelVersion {mv.version_id} has no onnx_model_key. "
                "ONNX export must complete before deployment."
            )

    def _get_onnx_presigned_url(self) -> str:
        """Generate a presigned download URL for the ONNX artifact from the storage profile."""
        mv = self.model_version
        storage_profile = mv.storage_profile
        if storage_profile is None:
            raise ValueError(f"ModelVersion {mv.version_id} has no storage_profile set.")

        try:
            url = storage_profile.generate_presigned_url(mv.onnx_model_key, expiry=3600)
            return url
        except Exception as e:
            raise ValueError(f"Could not generate presigned URL for ONNX artifact: {e}") from e

    def _get_class_names(self) -> list[str]:
        """Retrieve ordered class names from the project's AnnotationGroup."""
        try:
            from annotations.models import AnnotationClass, AnnotationGroup
            project = self.model_version.model.project
            group = AnnotationGroup.objects.filter(project=project).first()
            if group is None:
                return []
            classes = AnnotationClass.objects.filter(
                annotation_group=group
            ).order_by("class_id").values_list("name", flat=True)
            return list(classes)
        except Exception as e:
            logger.warning("Could not retrieve class names: %s", e)
            return []

    def _get_task_name(self) -> str:
        """Map ModelTask name to inference service task string."""
        task = self.model_version.model.task
        if task is None:
            return "object-detection"
        name = task.name.lower().replace(" ", "-")
        return name

    def _get_inference_service_url(self) -> str:
        return os.getenv("INFERENCE_SERVICE_URL", "http://perceptra-inference:8080")

    def _call_load_endpoint(
        self,
        inference_url: str,
        version_id: str,
        storage_url: str,
        task: str,
        class_names: list[str],
    ) -> None:
        payload = {
            "version_id": version_id,
            "storage_url": storage_url,
            "task": task,
            "class_names": class_names,
        }
        try:
            resp = requests.post(
                f"{inference_url}/v1/models/load",
                json=payload,
                timeout=120,
            )
            if resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"Inference service load failed ({resp.status_code}): {resp.text}"
                )
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach inference service: {e}") from e

    def _retire_previous_production(self, InferenceDeployment) -> None:
        """Mark any previous production ModelVersion as 'retired'."""
        current_production = self.model_version.model.get_production_version()
        if current_production and current_production.pk != self.model_version.pk:
            # Unload from inference service (best-effort)
            try:
                inference_url = self._get_inference_service_url()
                requests.delete(
                    f"{inference_url}/v1/models/{current_production.version_id}",
                    timeout=15,
                )
            except Exception:
                pass  # Don't block promotion if old model unload fails

            InferenceDeployment.objects.filter(
                model_version=current_production, is_active=True
            ).update(is_active=False, undeployed_at=timezone.now())

            current_production.deployment_status = "retired"
            current_production.save(update_fields=["deployment_status"])
            logger.info("Retired previous production version: %s", current_production.version_id)

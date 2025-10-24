
import time
import django
django.setup()
from celery import shared_task
from django.utils import timezone
from training.models import ModelVersion
from common_utils.training.utils import trigger_trainml_training


TrainML_API = "http://server1.learning.test.want:29092/api/v1/train"
@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5}, ignore_result=True,
             name='train_model:execute')
def train_model(self, version_id:str, base_version:str):
    try:
        version = ModelVersion.objects.get(id=version_id)
        session = version.training_session

        # Start training
        session.status = "running"
        session.started_at = timezone.now()
        session.save(update_fields=["status", "started_at"])

        payload = {
            "project_id": version.model.project.name,
            "base_version": base_version,
            "model_name": version.model.name,
            "dataset_name": version.dataset_version.version_name,
            "dataset_version": version.dataset_version.version_number,
            "framework": version.model.framework.name,
            "task": version.model.task.name,
            "model_id": version.model.id,
            "model_version_id": version.id,
            "dataset_id": version.dataset_version.id,
            "session_id": session.id,
            "config": version.config or {}
            }

        res = trigger_trainml_training(TrainML_API, payload)
        return res

    except Exception as e:
        if "version" in locals():
            version.status = "failed"
            version.save()
        if "session" in locals():
            session.status = "failed"
            session.error_message = str(e)
            session.completed_at = timezone.now()
            session.save()
        raise
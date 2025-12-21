
from ml_models.models import Model, ModelVersion, ModelTag, ModelFramework, ModelTask
from asgiref.sync import sync_to_async

def serialize_model_version(version: ModelVersion) -> dict:
    """Serialize a model version to dict"""
    dataset_info = None
    if version.dataset_version:
        dataset = version.dataset_version
        dataset_info = {
            "id": str(dataset.id),
            "name": dataset.version_name or f"v{dataset.id}",
            "version": dataset.version_name or str(dataset.id),
            "item_count": dataset.project.project_images.count() if hasattr(dataset, 'project') else 0,
            "created_at": dataset.created_at
        }
    
    # Generate presigned URLs for artifacts
    checkpoint_url = version.get_checkpoint_url() if version.checkpoint_key else None
    onnx_url = None
    if version.onnx_model_key:
        # Similar to checkpoint URL generation
        from storage.services import get_storage_adapter_for_profile
        if version.storage_profile.backend == "local":
            onnx_url = f"http://localhost:81/{version.storage_profile.config['base_path']}/{version.onnx_model_key}"
        else:
            adapter = get_storage_adapter_for_profile(version.storage_profile)
            presigned = adapter.generate_presigned_url(version.onnx_model_key, expiration=3600, method='GET')
            onnx_url = presigned.url
    
    logs_url = version.get_logs_url() if version.training_logs_key else None
    
    return {
        "id": version.version_id,
        "version_number": version.version_number,
        "version_name": version.version_name,
        "status": version.status,
        "deployment_status": version.deployment_status,
        "metrics": version.metrics,
        "config": version.config,
        "dataset": dataset_info,
        "artifacts": {
            "checkpoint": checkpoint_url,
            "onnx": onnx_url,
            "logs": logs_url,
        },
        "created_by": version.created_by.email if version.created_by else None,
        "created_at": version.created_at,
        "deployed_at": version.deployed_at
    }


@sync_to_async
def serialize_model_detail(model: Model) -> dict:
    """Serialize full model details"""
    versions = [
        serialize_model_version(v) 
        for v in model.versions.filter(is_deleted=False).order_by('-version_number')
    ]
    
    latest_version = model.get_latest_version()
    production_version = model.get_production_version()
    
    return {
        "id": model.model_id,
        "name": model.name,
        "description": model.description,
        "task": model.task.name,
        "framework": model.framework.name,
        "tags": [tag.name for tag in model.tags.all()],
        "project_id": str(model.project.project_id),
        "project_name": model.project.name,
        "versions": versions,
        "latest_version": serialize_model_version(latest_version) if latest_version else None,
        "production_version": serialize_model_version(production_version) if production_version else None,
        "created_by": model.created_by.email if model.created_by else None,
        "created_at": model.created_at,
        "updated_at": model.updated_at
    }

@sync_to_async
def serialize_models(models):
    result = []
    for model in models:
        latest_version = model.get_latest_version()
        result.append({
            "id": model.model_id,
            "name": model.name,
            "description": model.description,
            "task": model.task.name,
            "framework": model.framework.name,
            "tags": [tag.name for tag in model.tags.all()],
            "version_count": model.versions.filter(is_deleted=False).count(),
            "latest_version_number": latest_version.version_number if latest_version else None,
            "latest_status": latest_version.status if latest_version else None,
            "created_at": model.created_at,
            "updated_at": model.updated_at
        })

    return result
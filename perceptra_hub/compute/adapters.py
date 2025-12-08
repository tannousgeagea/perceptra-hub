"""
Provider adapters for training job submission.
Each adapter handles provider-specific logic for job execution.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseComputeAdapter(ABC):
    """
    Base adapter for compute providers.
    All provider adapters must implement these methods.
    """
    
    def __init__(self, provider):
        self.provider = provider
        self.config = provider.system_config
    
    @abstractmethod
    def submit_job(
        self,
        job_config: Dict[str, Any],
        credentials: Optional[Dict] = None
    ) -> str:
        """
        Submit training job to provider.
        
        Args:
            job_config: Job configuration dict with:
                - job_id: Internal job ID
                - model_version_id: Model version being trained
                - training_session_id: Training session ID
                - instance_type: Instance type to use
                - storage_profile: Storage profile object
                - dataset_version: Dataset version object
                - config: Training hyperparameters
                - organization_id: Organization ID
                - framework: Framework name (yolo, pytorch, etc.)
                - task: Task type (object-detection, etc.)
                - parent_version_id: Optional parent version for transfer learning
            credentials: User-provided credentials (if required)
        
        Returns:
            External job ID from provider
        
        Raises:
            RuntimeError: If submission fails
        """
        pass
    
    @abstractmethod
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        """
        Get current status of training job.
        
        Returns:
            Dict with:
                - status: queued|running|completed|failed|cancelled
                - progress: float 0-100
                - metrics: Dict of current metrics
                - error: Error message if failed
        """
        pass
    
    @abstractmethod
    def cancel_job(self, external_job_id: str) -> bool:
        """
        Cancel running training job.
        Returns True if successfully cancelled.
        """
        pass
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        """
        Validate provider credentials.
        
        Returns:
            Dict with:
                - valid: bool
                - message: str
                - details: Dict (optional)
        """
        # Default implementation - override in subclasses
        return {
            'valid': True,
            'message': 'Validation not implemented for this provider'
        }


# ============= Platform GPU Adapter (Celery) =============

class PlatformGPUAdapter(BaseComputeAdapter):
    """
    Adapter for platform's own GPU workers via Celery.
    Handles job submission to Celery task queue.
    """
    
    def submit_job(
        self,
        job_config: Dict[str, Any],
        credentials: Optional[Dict] = None
    ) -> str:
        """Submit to platform GPU queue via Celery"""
        from perceptra_hub.api.tasks.training.train_model import train_model_on_platform_gpu
        
        # Determine queue based on provider type
        queue = (
            'gpu-training' 
            if self.provider.provider_type == 'platform-gpu' 
            else 'cpu-training'
        )
        
        logger.info(
            f"Submitting job {job_config['job_id']} to {queue} queue"
        )
        
        # Submit to Celery
        task = train_model_on_platform_gpu.apply_async(
            kwargs={
                'job_id': job_config['job_id'],
                'model_version_id': job_config['model_version_id'],
                'training_session_id': job_config['training_session_id'],
                'config': job_config['config'],
                'parent_version_id': job_config.get('parent_version_id'),
            },
            queue=queue,
            task_id=job_config['training_session_id'],  # Use session ID as task ID
            priority=5
        )
        
        logger.info(f"Celery task submitted: {task.id}")
        return task.id
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        """Get status from Celery task"""
        from celery.result import AsyncResult
        
        result = AsyncResult(external_job_id)
        
        # Map Celery states to our states
        state_mapping = {
            'PENDING': 'queued',
            'STARTED': 'running',
            'PROGRESS': 'running',
            'SUCCESS': 'completed',
            'FAILURE': 'failed',
            'RETRY': 'queued',
            'REVOKED': 'cancelled',
        }
        
        status = state_mapping.get(result.state, 'queued')
        
        # Get progress from task meta
        progress = 0.0
        metrics = {}
        error = None
        
        if result.state == 'PROGRESS':
            info = result.info or {}
            progress = info.get('progress', 0.0)
            metrics = info.get('metrics', {})
        elif result.state == 'FAILURE':
            error = str(result.info)
        
        return {
            'status': status,
            'progress': progress,
            'metrics': metrics,
            'error': error
        }
    
    def cancel_job(self, external_job_id: str) -> bool:
        """Cancel Celery task"""
        from celery.result import AsyncResult
        
        result = AsyncResult(external_job_id)
        result.revoke(terminate=True, signal='SIGTERM')
        
        logger.info(f"Cancelled Celery task: {external_job_id}")
        return True


# ============= AWS SageMaker Adapter =============

class SageMakerAdapter(BaseComputeAdapter):
    """Adapter for AWS SageMaker training jobs"""
    
    def __init__(self, provider):
        super().__init__(provider)
        self._sagemaker_client = None
        self._s3_client = None
    
    def _get_sagemaker_client(self, credentials: Optional[Dict] = None):
        """Get SageMaker client with credentials"""
        import boto3
        
        if credentials:
            return boto3.client(
                'sagemaker',
                region_name=credentials.get('region', 'us-east-1'),
                aws_access_key_id=credentials['access_key'],
                aws_secret_access_key=credentials['secret_key']
            )
        else:
            # Use platform credentials
            return boto3.client(
                'sagemaker',
                region_name=self.config.get('region', 'us-east-1'),
                aws_access_key_id=self.config.get('access_key'),
                aws_secret_access_key=self.config.get('secret_key')
            )
    
    def submit_job(
        self,
        job_config: Dict[str, Any],
        credentials: Optional[Dict] = None
    ) -> str:
        """Submit training job to SageMaker"""
        import time
        
        sagemaker = self._get_sagemaker_client(credentials)
        
        # Generate unique job name (SageMaker requirement)
        job_name = f"cv-train-{job_config['job_id'][:8]}-{int(time.time())}"
        
        # Get S3 paths from storage profile
        storage_profile = job_config['storage_profile']
        s3_bucket = storage_profile.config.get('bucket')
        
        if not s3_bucket:
            raise ValueError("Storage profile must have S3 bucket configured")
        
        # Prepare dataset S3 URI
        dataset_s3_uri = f"s3://{s3_bucket}/datasets/{job_config['dataset_version'].id}/"
        
        # Prepare output S3 URI
        output_s3_uri = (
            f"s3://{s3_bucket}/organizations/{job_config['organization_id']}/"
            f"models/{job_config['model_name']}/v{job_config['model_version_id']}/"
        )
        
        # Get training image
        training_image = self._get_training_image(job_config['framework'])
        
        # Get role ARN
        role_arn = credentials.get('role_arn') if credentials else self.config.get('role_arn')
        if not role_arn:
            raise ValueError("SageMaker role ARN not configured")
        
        logger.info(f"Creating SageMaker job: {job_name}")
        
        # Create training job
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn=role_arn,
            AlgorithmSpecification={
                'TrainingImage': training_image,
                'TrainingInputMode': 'File',
                'EnableSageMakerMetricsTimeSeries': True
            },
            InputDataConfig=[
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': dataset_s3_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/x-image',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': output_s3_uri
            },
            ResourceConfig={
                'InstanceType': job_config['instance_type'],
                'InstanceCount': 1,
                'VolumeSizeInGB': 100
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 86400  # 24 hours
            },
            HyperParameters=self._format_hyperparameters(job_config['config']),
            Environment={
                'JOB_ID': job_config['job_id'],
                'MODEL_VERSION_ID': job_config['model_version_id'],
                'TRAINING_SESSION_ID': job_config['training_session_id'],
                'FRAMEWORK': job_config['framework'],
                'TASK': job_config['task']
            },
            Tags=[
                {'Key': 'Organization', 'Value': job_config['organization_id']},
                {'Key': 'Model', 'Value': job_config['model_name']},
                {'Key': 'JobID', 'Value': job_config['job_id']}
            ]
        )
        
        logger.info(f"SageMaker job created: {response['TrainingJobArn']}")
        return response['TrainingJobArn']
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        """Get SageMaker job status"""
        sagemaker = self._get_sagemaker_client()
        
        # Extract job name from ARN
        job_name = external_job_id.split('/')[-1]
        
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        
        # Map SageMaker status to our status
        status_mapping = {
            'InProgress': 'running',
            'Completed': 'completed',
            'Failed': 'failed',
            'Stopping': 'running',
            'Stopped': 'cancelled'
        }
        
        status = status_mapping.get(response['TrainingJobStatus'], 'queued')
        
        # Extract metrics if available
        metrics = {}
        if 'FinalMetricDataList' in response:
            for metric in response['FinalMetricDataList']:
                metrics[metric['MetricName']] = metric['Value']
        
        error = response.get('FailureReason')
        
        return {
            'status': status,
            'progress': 50.0 if status == 'running' else (100.0 if status == 'completed' else 0.0),
            'metrics': metrics,
            'error': error
        }
    
    def cancel_job(self, external_job_id: str) -> bool:
        """Cancel SageMaker job"""
        sagemaker = self._get_sagemaker_client()
        job_name = external_job_id.split('/')[-1]
        
        sagemaker.stop_training_job(TrainingJobName=job_name)
        logger.info(f"Cancelled SageMaker job: {job_name}")
        return True
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        """Validate AWS credentials"""
        try:
            sagemaker = self._get_sagemaker_client(credentials)
            # Try to list training jobs (lightweight operation)
            sagemaker.list_training_jobs(MaxResults=1)
            
            return {
                'valid': True,
                'message': 'AWS credentials validated successfully',
                'details': {'region': credentials.get('region', 'us-east-1')}
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'AWS credential validation failed: {str(e)}'
            }
    
    def _get_training_image(self, framework: str) -> str:
        """Get Docker image URL for training"""
        # Use configured image or default
        return self.config.get(
            'training_image',
            f"{self.config.get('ecr_registry')}/cv-training-{framework}:latest"
        )
    
    def _format_hyperparameters(self, config: Dict) -> Dict[str, str]:
        """Convert config to SageMaker hyperparameters format (all strings)"""
        return {k: str(v) for k, v in config.items()}


# ============= GCP Vertex AI Adapter =============

class VertexAIAdapter(BaseComputeAdapter):
    """Adapter for GCP Vertex AI training"""
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """Submit to Vertex AI"""
        # TODO: Implement Vertex AI submission
        raise NotImplementedError("Vertex AI adapter coming soon")
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Vertex AI adapter coming soon")
    
    def cancel_job(self, external_job_id: str) -> bool:
        raise NotImplementedError("Vertex AI adapter coming soon")


# ============= Azure ML Adapter =============

class AzureMLAdapter(BaseComputeAdapter):
    """Adapter for Azure Machine Learning"""
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """Submit to Azure ML"""
        # TODO: Implement Azure ML submission
        raise NotImplementedError("Azure ML adapter coming soon")
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Azure ML adapter coming soon")
    
    def cancel_job(self, external_job_id: str) -> bool:
        raise NotImplementedError("Azure ML adapter coming soon")


# ============= Kubernetes Adapter =============

class KubernetesAdapter(BaseComputeAdapter):
    """Adapter for Kubernetes GPU pods"""
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """Submit Kubernetes job"""
        # TODO: Implement K8s submission
        raise NotImplementedError("Kubernetes adapter coming soon")
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Kubernetes adapter coming soon")
    
    def cancel_job(self, external_job_id: str) -> bool:
        raise NotImplementedError("Kubernetes adapter coming soon")


# ============= Modal Adapter =============

class ModalAdapter(BaseComputeAdapter):
    """Adapter for Modal Labs serverless GPU"""
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """Submit to Modal"""
        # TODO: Implement Modal submission
        raise NotImplementedError("Modal adapter coming soon")
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Modal adapter coming soon")
    
    def cancel_job(self, external_job_id: str) -> bool:
        raise NotImplementedError("Modal adapter coming soon")
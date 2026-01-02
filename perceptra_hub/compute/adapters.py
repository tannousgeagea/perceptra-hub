"""
Provider adapters for training job submission.
Each adapter handles provider-specific logic for job execution.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from compute.models import ComputeProvider

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


# ============= AWS SageMaker Adapter (COMPLETE) =============

class SageMakerAdapter(BaseComputeAdapter):
    """
    Complete AWS SageMaker training adapter.
    Handles job submission, monitoring, and artifact management.
    """
    
    def __init__(self, provider):
        super().__init__(provider)
        self._sagemaker = None
        self._s3 = None
        self._sts = None
    
    def _get_boto_session(self, credentials: Optional[Dict] = None):
        """Get boto3 session with credentials"""
        import boto3
        
        if credentials:
            return boto3.Session(
                aws_access_key_id=credentials['access_key'],
                aws_secret_access_key=credentials['secret_key'],
                region_name=credentials.get('region', 'us-east-1')
            )
        else:
            return boto3.Session(
                aws_access_key_id=self.config.get('access_key'),
                aws_secret_access_key=self.config.get('secret_key'),
                region_name=self.config.get('region', 'us-east-1')
            )
    
    def _get_sagemaker_client(self, credentials: Optional[Dict] = None):
        """Get SageMaker client"""
        session = self._get_boto_session(credentials)
        return session.client('sagemaker')
    
    def _get_s3_client(self, credentials: Optional[Dict] = None):
        """Get S3 client"""
        session = self._get_boto_session(credentials)
        return session.client('s3')
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """
        Submit training job to SageMaker.
        
        Process:
        1. Prepare dataset in S3
        2. Create training container spec
        3. Submit SageMaker training job
        4. Return job ARN
        """
        sagemaker = self._get_sagemaker_client(credentials)
        s3 = self._get_s3_client(credentials)
        
        # Generate unique job name
        job_name = self._generate_job_name(job_config)
        
        # Get storage configuration
        storage_profile = job_config['storage_profile']
        
        # Ensure dataset is in S3
        dataset_s3_uri = self._prepare_dataset_s3(
            job_config['dataset_version'],
            storage_profile,
            s3,
            credentials
        )
        
        # Output S3 path
        output_s3_uri = self._get_output_s3_uri(job_config, storage_profile)
        
        # Get training image
        training_image = self._get_training_image(
            job_config['framework'],
            credentials or self.config
        )
        
        # Get IAM role
        role_arn = self._get_role_arn(credentials)
        
        # Prepare hyperparameters
        hyperparameters = self._prepare_hyperparameters(job_config)
        
        # Prepare environment variables
        environment = {
            'JOB_ID': job_config['job_id'],
            'MODEL_VERSION_ID': job_config['model_version_id'],
            'TRAINING_SESSION_ID': job_config['training_session_id'],
            'FRAMEWORK': job_config['framework'],
            'TASK': job_config['task'],
            'ORGANIZATION_ID': job_config['organization_id'],
            'PARENT_VERSION_ID': job_config.get('parent_version_id', ''),
            # Storage credentials
            'AWS_ACCESS_KEY_ID': credentials.get('access_key', '') if credentials else '',
            'AWS_SECRET_ACCESS_KEY': credentials.get('secret_key', '') if credentials else '',
        }
        
        # Resource configuration
        instance_type = job_config['instance_type']
        instance_count = job_config.get('instance_count', 1)
        volume_size_gb = job_config['config'].get('volume_size_gb', 100)
        
        # Training configuration
        training_config = {
            'TrainingJobName': job_name,
            'RoleArn': role_arn,
            
            # Algorithm specification
            'AlgorithmSpecification': {
                'TrainingImage': training_image,
                'TrainingInputMode': 'File',
                'EnableSageMakerMetricsTimeSeries': True,
                'ContainerEntrypoint': ['python3', '/app/train.py'],
            },
            
            # Input data
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': dataset_s3_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/x-tar',
                    'CompressionType': 'None',
                }
            ],
            
            # Output data
            'OutputDataConfig': {
                'S3OutputPath': output_s3_uri
            },
            
            # Resource configuration
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': instance_count,
                'VolumeSizeInGB': volume_size_gb
            },
            
            # Stopping condition
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400  # 24 hours
            },
            
            # Hyperparameters
            'HyperParameters': hyperparameters,
            
            # Environment variables
            'Environment': environment,
            
            # Tagging
            'Tags': [
                {'Key': 'Organization', 'Value': job_config['organization_id']},
                {'Key': 'Model', 'Value': job_config['model_name']},
                {'Key': 'JobID', 'Value': job_config['job_id']},
                {'Key': 'ManagedBy', 'Value': 'CVPlatform'}
            ],
            
            # Enable spot instances if configured
            'EnableManagedSpotTraining': self.config.get('use_spot_instances', False),
            
            # Checkpointing for spot instance interruptions
            'CheckpointConfig': {
                'S3Uri': f"{output_s3_uri}/checkpoints",
                'LocalPath': '/opt/ml/checkpoints'
            } if self.config.get('use_spot_instances', False) else None,
        }
        
        # Remove None values
        training_config = {k: v for k, v in training_config.items() if v is not None}
        
        # Submit training job
        logger.info(f"Submitting SageMaker training job: {job_name}")
        
        try:
            response = sagemaker.create_training_job(**training_config)
            job_arn = response['TrainingJobArn']
            
            logger.info(f"SageMaker job submitted: {job_arn}")
            return job_arn
            
        except Exception as e:
            logger.error(f"Failed to submit SageMaker job: {e}")
            raise RuntimeError(f"SageMaker submission failed: {str(e)}")
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        """Get SageMaker training job status"""
        sagemaker = self._get_sagemaker_client()
        
        # Extract job name from ARN
        job_name = external_job_id.split('/')[-1]
        
        try:
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
            
            # Extract progress
            progress = 0.0
            if 'SecondaryStatusTransitions' in response:
                transitions = response['SecondaryStatusTransitions']
                if transitions:
                    last_status = transitions[-1]['Status']
                    if last_status == 'Completed':
                        progress = 100.0
                    elif last_status in ['Training', 'Downloading']:
                        progress = 50.0
                    elif last_status == 'Starting':
                        progress = 10.0
            
            # Extract metrics
            metrics = {}
            if 'FinalMetricDataList' in response:
                for metric in response['FinalMetricDataList']:
                    metrics[metric['MetricName']] = metric['Value']
            
            # Extract error
            error = response.get('FailureReason')
            
            # Training time
            training_time = None
            if 'TrainingStartTime' in response and 'TrainingEndTime' in response:
                training_time = (
                    response['TrainingEndTime'] - response['TrainingStartTime']
                ).total_seconds()
            
            return {
                'status': status,
                'progress': progress,
                'metrics': metrics,
                'error': error,
                'training_time_seconds': training_time,
                'billable_time_seconds': response.get('BillableTimeInSeconds'),
                'instance_type': response['ResourceConfig']['InstanceType'],
            }
            
        except sagemaker.exceptions.ResourceNotFound:
            logger.error(f"SageMaker job not found: {job_name}")
            return {
                'status': 'failed',
                'progress': 0.0,
                'metrics': {},
                'error': 'Job not found'
            }
        except Exception as e:
            logger.error(f"Error getting SageMaker status: {e}")
            raise
    
    def cancel_job(self, external_job_id: str) -> bool:
        """Cancel SageMaker training job"""
        sagemaker = self._get_sagemaker_client()
        job_name = external_job_id.split('/')[-1]
        
        try:
            sagemaker.stop_training_job(TrainingJobName=job_name)
            logger.info(f"Cancelled SageMaker job: {job_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel SageMaker job: {e}")
            return False
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        """Validate AWS credentials"""
        try:
            sagemaker = self._get_sagemaker_client(credentials)
            
            # Test API call
            sagemaker.list_training_jobs(MaxResults=1)
            
            # Get account info
            session = self._get_boto_session(credentials)
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            
            return {
                'valid': True,
                'message': 'AWS credentials validated successfully',
                'details': {
                    'account_id': identity['Account'],
                    'user_arn': identity['Arn'],
                    'region': credentials.get('region', 'us-east-1')
                }
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'AWS credential validation failed: {str(e)}'
            }
    
    # ============= Helper Methods =============
    
    def _generate_job_name(self, job_config: Dict) -> str:
        """Generate unique SageMaker job name"""
        # SageMaker job names: max 63 chars, alphanumeric + hyphens
        job_id_short = job_config['job_id'][:8]
        timestamp = int(time.time())
        return f"cv-train-{job_id_short}-{timestamp}"
    
    def _prepare_dataset_s3(
        self,
        dataset_version,
        storage_profile,
        s3_client,
        credentials
    ) -> str:
        """
        Ensure dataset is in S3 and return S3 URI.
        If dataset is in different storage, copy to S3.
        """
        # If storage profile is already S3
        if storage_profile.backend == 's3':
            bucket = storage_profile.config['bucket']
            dataset_key = f"datasets/{dataset_version.id}/data.tar.gz"
            return f"s3://{bucket}/{dataset_key}"
        
        # Otherwise, copy from current storage to S3
        # (Implementation depends on your storage abstraction)
        logger.info("Dataset not in S3, copying...")
        
        # Get target S3 bucket
        target_bucket = credentials.get('training_bucket') or self.config.get('training_bucket')
        if not target_bucket:
            raise ValueError("No S3 bucket configured for training data")
        
        dataset_key = f"training-data/datasets/{dataset_version.id}/data.tar.gz"
        s3_uri = f"s3://{target_bucket}/{dataset_key}"
        
        # TODO: Implement actual copy logic
        # For now, assume dataset is already in S3
        logger.warning("Dataset copy not implemented, assuming data is in S3")
        
        return s3_uri
    
    def _get_output_s3_uri(self, job_config: Dict, storage_profile) -> str:
        """Get S3 URI for training outputs"""
        if storage_profile.backend == 's3':
            bucket = storage_profile.config['bucket']
        else:
            bucket = self.config.get('output_bucket') or self.config.get('training_bucket')
        
        org_id = job_config['organization_id']
        model_name = job_config['model_name']
        version_id = job_config['model_version_id']
        
        return f"s3://{bucket}/organizations/{org_id}/models/{model_name}/{version_id}"
    
    def _get_training_image(self, framework: str, config: Dict) -> str:
        """
        Get Docker image URI for training.
        Uses ECR repository or provided image URI.
        """
        # Check if custom image provided
        custom_image = config.get(f'training_image_{framework}')
        if custom_image:
            return custom_image
        
        # Use default ECR image
        region = config.get('region', 'us-east-1')
        account_id = config.get('account_id')
        ecr_repo = config.get('ecr_repository', 'cv-training')
        
        if not account_id:
            raise ValueError("AWS account ID not configured")
        
        return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repo}:{framework}-latest"
    
    def _get_role_arn(self, credentials: Optional[Dict]) -> str:
        """Get IAM role ARN for SageMaker"""
        if credentials and 'role_arn' in credentials:
            return credentials['role_arn']
        
        role_arn = self.config.get('sagemaker_role_arn') or self.config.get('role_arn')
        if not role_arn:
            raise ValueError(
                "SageMaker IAM role not configured. "
                "Set 'role_arn' in compute provider config or user credentials"
            )
        
        return role_arn
    
    def _prepare_hyperparameters(self, job_config: Dict) -> Dict[str, str]:
        """Convert training config to SageMaker hyperparameters (all strings)"""
        config = job_config['config']
        
        hyperparameters = {
            'epochs': str(config.get('epochs', 100)),
            'batch-size': str(config.get('batch_size', 16)),
            'learning-rate': str(config.get('learning_rate', 0.001)),
            'image-size': str(config.get('image_size', 640)),
            'workers': str(config.get('workers', 4)),
            'optimizer': str(config.get('optimizer', 'Adam')),
        }
        
        # Add model-specific params
        if 'model_params' in config:
            for key, value in config['model_params'].items():
                hyperparameters[key.replace('_', '-')] = str(value)
        
        return hyperparameters


# ============= GCP Vertex AI Adapter (Template) =============

class VertexAIAdapter(BaseComputeAdapter):
    """Google Cloud Vertex AI training adapter"""
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """Submit to Vertex AI"""
        from google.cloud import aiplatform
        from google.oauth2 import service_account
        
        # Initialize Vertex AI
        if credentials:
            creds = service_account.Credentials.from_service_account_info(
                credentials['service_account']
            )
            project_id = credentials['project_id']
            region = credentials.get('region', 'us-central1')
        else:
            creds = None
            project_id = self.config['project_id']
            region = self.config.get('region', 'us-central1')
        
        aiplatform.init(
            project=project_id,
            location=region,
            credentials=creds
        )
        
        # Create custom training job
        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"cv-train-{job_config['job_id'][:8]}",
            container_uri=self._get_training_image(job_config['framework']),
            command=['python3', '/app/train.py'],
            model_serving_container_image_uri=None  # No serving for training-only jobs
        )
        
        # Prepare dataset (must be in GCS)
        dataset_gcs_uri = self._prepare_dataset_gcs(job_config)
        output_gcs_uri = self._get_output_gcs_uri(job_config)
        
        # Submit job
        model = job.run(
            replica_count=1,
            machine_type=job_config['instance_type'],
            accelerator_type=self._get_gpu_type(job_config['instance_type']),
            accelerator_count=1,
            args=[
                f"--job-id={job_config['job_id']}",
                f"--dataset-uri={dataset_gcs_uri}",
                f"--output-uri={output_gcs_uri}",
                f"--config={json.dumps(job_config['config'])}"
            ],
            environment_variables={
                'JOB_ID': job_config['job_id'],
                'FRAMEWORK': job_config['framework'],
                'TASK': job_config['task'],
            }
        )
        
        logger.info(f"Vertex AI job submitted: {job.resource_name}")
        return job.resource_name
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        from google.cloud import aiplatform
        
        job = aiplatform.CustomTrainingJob.get(external_job_id)
        
        state_mapping = {
            'JOB_STATE_QUEUED': 'queued',
            'JOB_STATE_PENDING': 'queued',
            'JOB_STATE_RUNNING': 'running',
            'JOB_STATE_SUCCEEDED': 'completed',
            'JOB_STATE_FAILED': 'failed',
            'JOB_STATE_CANCELLED': 'cancelled',
        }
        
        status = state_mapping.get(job.state.name, 'queued')
        
        return {
            'status': status,
            'progress': 50.0 if status == 'running' else (100.0 if status == 'completed' else 0.0),
            'metrics': {},
            'error': job.error.message if job.error else None
        }
    
    def cancel_job(self, external_job_id: str) -> bool:
        from google.cloud import aiplatform
        
        job = aiplatform.CustomTrainingJob.get(external_job_id)
        job.cancel()
        return True
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        try:
            from google.oauth2 import service_account
            from google.cloud import aiplatform
            
            creds = service_account.Credentials.from_service_account_info(
                credentials['service_account']
            )
            
            aiplatform.init(
                project=credentials['project_id'],
                location=credentials.get('region', 'us-central1'),
                credentials=creds
            )
            
            # Test API access
            list(aiplatform.CustomTrainingJob.list(limit=1))
            
            return {
                'valid': True,
                'message': 'GCP credentials validated',
                'details': {'project_id': credentials['project_id']}
            }
        except Exception as e:
            return {'valid': False, 'message': f'GCP validation failed: {str(e)}'}
    
    def _get_training_image(self, framework: str) -> str:
        region = self.config.get('region', 'us-central1')
        project_id = self.config['project_id']
        return f"{region}-docker.pkg.dev/{project_id}/cv-training/{framework}:latest"
    
    def _prepare_dataset_gcs(self, job_config: Dict) -> str:
        # Implement GCS dataset preparation
        bucket = self.config.get('training_bucket')
        dataset_id = job_config['dataset_version'].id
        return f"gs://{bucket}/datasets/{dataset_id}/data.tar.gz"
    
    def _get_output_gcs_uri(self, job_config: Dict) -> str:
        bucket = self.config.get('output_bucket')
        org_id = job_config['organization_id']
        return f"gs://{bucket}/organizations/{org_id}/models/"
    
    def _get_gpu_type(self, instance_type: str) -> str:
        # Map instance type to GPU type
        gpu_mapping = {
            'n1-standard-4': 'NVIDIA_TESLA_T4',
            'n1-standard-8': 'NVIDIA_TESLA_V100',
            'a2-highgpu-1g': 'NVIDIA_TESLA_A100',
        }
        return gpu_mapping.get(instance_type, 'NVIDIA_TESLA_T4')


# ============= Azure ML Adapter (Template) =============

class AzureMLAdapter(BaseComputeAdapter):
    """Azure Machine Learning training adapter"""
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml.entities import Command, Environment
        
        # Initialize ML Client
        if credentials:
            credential = DefaultAzureCredential()  # Use managed identity or service principal
            subscription_id = credentials['subscription_id']
            resource_group = credentials['resource_group']
            workspace_name = credentials['workspace_name']
        else:
            credential = DefaultAzureCredential()
            subscription_id = self.config['subscription_id']
            resource_group = self.config['resource_group']
            workspace_name = self.config['workspace_name']
        
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        
        # Create job
        job = Command(
            code="./src",  # Path to training code
            command="python train.py --job-id ${{inputs.job_id}}",
            inputs={
                "job_id": job_config['job_id'],
                "dataset_uri": self._prepare_dataset_azure(job_config),
            },
            environment=Environment(
                image=self._get_training_image(job_config['framework']),
                conda_file="environment.yml"
            ),
            compute=job_config['instance_type'],
            display_name=f"cv-train-{job_config['job_id'][:8]}",
        )
        
        # Submit
        returned_job = ml_client.jobs.create_or_update(job)
        
        logger.info(f"Azure ML job submitted: {returned_job.name}")
        return returned_job.name
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        # Implement Azure ML status check
        raise NotImplementedError("Azure ML adapter in development")
    
    def cancel_job(self, external_job_id: str) -> bool:
        raise NotImplementedError("Azure ML adapter in development")
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential,
                credentials['subscription_id'],
                credentials['resource_group'],
                credentials['workspace_name']
            )
            
            # Test API access
            list(ml_client.data.list(max_results=1))
            
            return {'valid': True, 'message': 'Azure credentials validated'}
        except Exception as e:
            return {'valid': False, 'message': f'Azure validation failed: {str(e)}'}
    
    def _get_training_image(self, framework: str) -> str:
        registry = self.config.get('container_registry')
        return f"{registry}/cv-training:{framework}-latest"
    
    def _prepare_dataset_azure(self, job_config: Dict) -> str:
        # Implement Azure Blob storage dataset preparation
        storage_account = self.config.get('storage_account')
        container = self.config.get('training_container')
        dataset_id = job_config['dataset_version'].id
        return f"https://{storage_account}.blob.core.windows.net/{container}/datasets/{dataset_id}/"


"""
Kubernetes GPU Training Adapter - Complete Implementation
Submits training jobs as Kubernetes Jobs with GPU allocation.
"""

class KubernetesAdapter(BaseComputeAdapter):
    """
    Kubernetes training adapter for GPU workloads.
    Creates Kubernetes Jobs with GPU resource requests.
    """
    
    def __init__(self, provider):
        super().__init__(provider)
        self.namespace = self.config.get('namespace', 'default')
        self.image_pull_secret = self.config.get('image_pull_secret')
        self.service_account = self.config.get('service_account', 'default')
    
    def submit_job(self, job_config: Dict[str, Any], credentials: Optional[Dict] = None) -> str:
        """Submit training job as Kubernetes Job"""
        from kubernetes import client
        from kubernetes.client.rest import ApiException
        
        # Initialize K8s client
        self._init_k8s_client(credentials)
        
        batch_v1 = client.BatchV1Api()
        
        # Generate job name (K8s naming: lowercase, max 63 chars, alphanumeric + hyphens)
        job_name = self._generate_job_name(job_config['job_id'])
        
        # Build Job spec
        job_spec = self._build_job_spec(job_name, job_config, credentials)
        
        try:
            # Submit job
            api_response = batch_v1.create_namespaced_job(self.namespace, job_spec)
            
            job_uid = api_response.metadata.uid
            logger.info(f"Kubernetes job created: {job_name} (UID: {job_uid})")
            
            return f"{self.namespace}/{job_name}"
            
        except ApiException as e:
            logger.error(f"Failed to create K8s job: {e}")
            raise RuntimeError(f"Kubernetes job submission failed: {e.reason}")
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        """Get Kubernetes Job status"""
        from kubernetes import client
        from kubernetes.client.rest import ApiException
        
        self._init_k8s_client()
        
        batch_v1 = client.BatchV1Api()
        core_v1 = client.CoreV1Api()
        
        namespace, job_name = self._parse_job_id(external_job_id)
        
        try:
            # Get job
            job = batch_v1.read_namespaced_job(job_name, namespace)
            
            # Determine status
            status = 'queued'
            progress = 0.0
            error = None
            
            if job.status.succeeded:
                status = 'completed'
                progress = 100.0
            elif job.status.failed:
                status = 'failed'
                progress = 0.0
                error = self._get_failure_reason(job_name, namespace, core_v1)
            elif job.status.active:
                status = 'running'
                progress = 50.0
            
            # Extract metrics from logs if available
            metrics = self._extract_metrics_from_logs(job_name, namespace, core_v1)
            
            return {
                'status': status,
                'progress': progress,
                'metrics': metrics,
                'error': error
            }
            
        except ApiException as e:
            if e.status == 404:
                return {'status': 'failed', 'progress': 0.0, 'metrics': {}, 'error': 'Job not found'}
            logger.error(f"Error getting K8s job status: {e}")
            raise
    
    def cancel_job(self, external_job_id: str) -> bool:
        """Cancel Kubernetes Job"""
        from kubernetes import client
        from kubernetes.client.rest import ApiException
        
        self._init_k8s_client()
        
        batch_v1 = client.BatchV1Api()
        namespace, job_name = self._parse_job_id(external_job_id)
        
        try:
            # Delete job (cascade deletes pods)
            batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy='Background')
            )
            
            logger.info(f"Cancelled K8s job: {job_name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to cancel K8s job: {e}")
            return False
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        """Validate Kubernetes credentials"""
        try:
            from kubernetes import client
            
            self._init_k8s_client(credentials)
            
            # Test API access
            core_v1 = client.CoreV1Api()
            core_v1.list_namespaced_pod(self.namespace, limit=1)
            
            # Get cluster info
            version_api = client.VersionApi()
            version = version_api.get_code()
            
            return {
                'valid': True,
                'message': 'Kubernetes credentials validated',
                'details': {
                    'namespace': self.namespace,
                    'kubernetes_version': version.git_version
                }
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'Kubernetes validation failed: {str(e)}'
            }
    
    # ============= Helper Methods =============
    
    def _init_k8s_client(self, credentials: Optional[Dict] = None):
        """Initialize Kubernetes client with credentials"""
        from kubernetes import client, config
        
        if credentials:
            if 'kubeconfig' in credentials:
                # Load from kubeconfig dict
                config.load_kube_config_from_dict(credentials['kubeconfig'])
            elif 'token' in credentials:
                # Use token authentication
                configuration = client.Configuration()
                configuration.host = credentials['api_server']
                configuration.api_key = {"authorization": f"Bearer {credentials['token']}"}
                configuration.verify_ssl = credentials.get('verify_ssl', True)
                client.Configuration.set_default(configuration)
            else:
                # Default: load from ~/.kube/config
                config.load_kube_config()
        else:
            try:
                # Try in-cluster config first (if running inside K8s)
                config.load_incluster_config()
            except:
                # Fall back to local kubeconfig
                config.load_kube_config()
    
    def _generate_job_name(self, job_id: str) -> str:
        """Generate K8s-compliant job name"""
        # Max 63 chars, lowercase alphanumeric + hyphens
        import re
        job_id_short = job_id[:8].lower()
        timestamp = int(time.time())
        name = f"cv-train-{job_id_short}-{timestamp}"
        # Ensure valid K8s name
        name = re.sub(r'[^a-z0-9-]', '-', name)
        return name[:63]
    
    def _build_job_spec(self, job_name: str, job_config: Dict, credentials: Optional[Dict]):
        """Build Kubernetes Job specification"""
        from kubernetes import client
        
        # Container image
        image = self._get_training_image(job_config['framework'], credentials)
        
        # GPU count
        gpu_count = job_config.get('gpu_count', 1)
        
        # Environment variables
        env_vars = [
            client.V1EnvVar(name="JOB_ID", value=job_config['job_id']),
            client.V1EnvVar(name="MODEL_VERSION_ID", value=job_config['model_version_id']),
            client.V1EnvVar(name="TRAINING_SESSION_ID", value=job_config['training_session_id']),
            client.V1EnvVar(name="FRAMEWORK", value=job_config['framework']),
            client.V1EnvVar(name="TASK", value=job_config['task']),
            client.V1EnvVar(name="ORGANIZATION_ID", value=job_config['organization_id']),
        ]
        
        # Storage credentials as secrets
        if credentials:
            env_vars.extend([
                client.V1EnvVar(name="AWS_ACCESS_KEY_ID", value=credentials.get('aws_access_key', '')),
                client.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=credentials.get('aws_secret_key', '')),
            ])
        
        # Resource requirements
        resources = client.V1ResourceRequirements(
            requests={
                "nvidia.com/gpu": str(gpu_count),
                "memory": job_config.get('memory', '16Gi'),
                "cpu": str(job_config.get('cpu_cores', 4))
            },
            limits={
                "nvidia.com/gpu": str(gpu_count),
                "memory": job_config.get('memory', '16Gi'),
            }
        )
        
        # Volume mounts (for dataset caching)
        volumes = []
        volume_mounts = []
        
        if self.config.get('enable_shared_cache'):
            volumes.append(
                client.V1Volume(
                    name="cache",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self.config.get('cache_pvc', 'training-cache')
                    )
                )
            )
            volume_mounts.append(
                client.V1VolumeMount(name="cache", mount_path="/cache")
            )
        
        # Container spec
        container = client.V1Container(
            name="training",
            image=image,
            image_pull_policy="Always",
            command=["python3", "/app/training/tasks.py"],
            args=[
                "--job-id", job_config['job_id'],
                "--config", json.dumps(job_config['config'])
            ],
            env=env_vars,
            resources=resources,
            volume_mounts=volume_mounts if volume_mounts else None
        )
        
        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            service_account_name=self.service_account,
            volumes=volumes if volumes else None,
            image_pull_secrets=[
                client.V1LocalObjectReference(name=self.image_pull_secret)
            ] if self.image_pull_secret else None,
            # Node selector for GPU nodes
            node_selector={"accelerator": "nvidia-gpu"} if gpu_count > 0 else None,
            # Tolerations for GPU taints
            tolerations=[
                client.V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule"
                )
            ] if gpu_count > 0 else None
        )
        
        # Job spec
        job_spec = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app": "cv-training",
                    "job-id": job_config['job_id'],
                    "framework": job_config['framework'],
                    "organization": job_config['organization_id']
                }
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "cv-training",
                            "job-id": job_config['job_id']
                        }
                    ),
                    spec=pod_spec
                ),
                backoff_limit=3,  # Retry on failure
                ttl_seconds_after_finished=3600,  # Clean up after 1 hour
                active_deadline_seconds=86400  # 24-hour timeout
            )
        )
        
        return job_spec
    
    def _get_training_image(self, framework: str, credentials: Optional[Dict]) -> str:
        """Get Docker image for training"""
        # Use custom registry if provided
        if credentials and 'container_registry' in credentials:
            registry = credentials['container_registry']
        else:
            registry = self.config.get('container_registry', 'your-registry.io')
        
        repo = self.config.get('repository', 'cv-training')
        tag = self.config.get('image_tag', 'latest')
        
        return f"{registry}/{repo}:{framework}-{tag}"
    
    def _parse_job_id(self, external_job_id: str) -> tuple:
        """Parse namespace/job-name from external job ID"""
        if '/' in external_job_id:
            return external_job_id.split('/', 1)
        return self.namespace, external_job_id
    
    def _get_failure_reason(self, job_name: str, namespace: str, core_v1) -> str:
        """Extract failure reason from pod logs"""
        try:
            # Get pods for this job
            pods = core_v1.list_namespaced_pod(
                namespace,
                label_selector=f"job-name={job_name}"
            )
            
            if not pods.items:
                return "No pods found"
            
            # Get logs from failed pod
            pod = pods.items[0]
            
            if pod.status.phase == 'Failed':
                # Check container statuses
                if pod.status.container_statuses:
                    for status in pod.status.container_statuses:
                        if status.state.terminated:
                            return status.state.terminated.reason or "Unknown failure"
                
                # Try to get logs
                try:
                    logs = core_v1.read_namespaced_pod_log(
                        pod.metadata.name,
                        namespace,
                        tail_lines=50
                    )
                    # Extract last few lines as error
                    return logs.split('\n')[-10:] if logs else "No logs available"
                except:
                    return "Failed to retrieve logs"
            
            return "Job failed"
            
        except Exception as e:
            logger.warning(f"Could not get failure reason: {e}")
            return "Unknown failure"
    
    def _extract_metrics_from_logs(self, job_name: str, namespace: str, core_v1) -> Dict:
        """Extract training metrics from pod logs"""
        try:
            # Get pods for this job
            pods = core_v1.list_namespaced_pod(
                namespace,
                label_selector=f"job-name={job_name}"
            )
            
            if not pods.items:
                return {}
            
            pod = pods.items[0]
            
            # Get recent logs
            logs = core_v1.read_namespaced_pod_log(
                pod.metadata.name,
                namespace,
                tail_lines=100
            )
            
            # Parse metrics from logs (simple JSON parsing)
            # Assumes training code logs metrics as: METRICS: {"loss": 0.5, "accuracy": 0.95}
            metrics = {}
            for line in logs.split('\n'):
                if 'METRICS:' in line:
                    try:
                        import json
                        metrics_str = line.split('METRICS:')[1].strip()
                        metrics = json.loads(metrics_str)
                        break
                    except:
                        continue
            
            return metrics
            
        except Exception as e:
            logger.debug(f"Could not extract metrics: {e}")
            return {}
        

# ============= Add this class to compute/adapters.py =============

class OnPremiseAgentAdapter(BaseComputeAdapter):
    """
    Adapter for on-premise training agents.
    Submits jobs via Redis queue for agents to pick up.
    """
    
    def submit_job(
        self,
        job_config: Dict[str, Any],
        credentials: Optional[Dict] = None
    ) -> str:
        """
        Submit job to on-premise agent via Redis queue.
        
        Process:
        1. Find available agent for organization
        2. Assign job to agent via Redis
        3. Return agent job identifier
        """
        from compute.services.agent_manager import AgentManager
        from compute.models import Agent, TrainingJob
        from training.models import TrainingSession
        
        # Get organization
        org_id = job_config['organization_id']
        
        # Find available agent
        agent = AgentManager.get_available_agent(
            organization_id=org_id,
            required_gpus=job_config.get('gpu_count', 1)
        )
        
        if not agent:
            raise RuntimeError(
                f"No available agents for organization {org_id}. "
                "Please register an agent or wait for existing agents to become available."
            )
        
        logger.info(f"Selected agent {agent.agent_id} for job {job_config['job_id']}")
        
        # Get training job and session
        training_job = TrainingJob.objects.select_related('training_session').get(
            job_id=job_config['job_id']
        )
        
        # Assign job to agent
        job_assignment = AgentManager.assign_job_to_agent(
            training_job=training_job,
            agent=agent
        )
        
        # Return external job ID
        external_job_id = f"agent:{agent.agent_id}:{job_config['job_id']}"
        
        logger.info(
            f"Job {job_config['job_id']} assigned to agent {agent.agent_id}"
        )
        
        return external_job_id
    
    def get_job_status(self, external_job_id: str) -> Dict[str, Any]:
        """
        Get job status from database.
        Agent updates status via API, we just read it.
        """
        from compute.models import TrainingJob
        
        try:
            # Parse external_job_id: "agent:{agent_id}:{job_id}"
            parts = external_job_id.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid external_job_id format: {external_job_id}")
            
            job_id = parts[2]
            
            # Get training job
            training_job = TrainingJob.objects.select_related(
                'training_session'
            ).get(job_id=job_id)
            
            session = training_job.training_session
            
            # Map session status to adapter status
            status_mapping = {
                'queued': 'queued',
                'initializing': 'running',
                'running': 'running',
                'completed': 'completed',
                'failed': 'failed',
                'cancelled': 'cancelled',
            }
            
            status = status_mapping.get(session.status, 'queued')
            
            return {
                'status': status,
                'progress': session.progress,
                'metrics': session.metrics or {},
                'error': session.error_message
            }
            
        except TrainingJob.DoesNotExist:
            logger.error(f"Training job not found for {external_job_id}")
            return {
                'status': 'failed',
                'progress': 0.0,
                'metrics': {},
                'error': 'Job not found'
            }
        except Exception as e:
            logger.error(f"Error getting agent job status: {e}")
            raise
    
    def cancel_job(self, external_job_id: str) -> bool:
        """
        Cancel agent job.
        Sets status to cancelled, agent will stop on next poll.
        """
        from compute.models import TrainingJob
        from django.utils import timezone
        
        try:
            # Parse job_id from external_job_id
            parts = external_job_id.split(':')
            if len(parts) != 3:
                return False
            
            job_id = parts[2]
            
            # Get training job
            training_job = TrainingJob.objects.select_related(
                'training_session'
            ).get(job_id=job_id)
            
            session = training_job.training_session
            
            # Update status
            session.status = 'cancelled'
            session.completed_at = timezone.now()
            session.save()
            
            logger.info(f"Cancelled agent job: {job_id}")
            return True
            
        except TrainingJob.DoesNotExist:
            logger.error(f"Training job not found: {job_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel agent job: {e}")
            return False
    
    def validate_credentials(self, credentials: Dict) -> Dict[str, Any]:
        """
        Validate agent availability.
        On-premise agents don't need credentials, just check if agents exist.
        """
        from compute.models import Agent
        from django.utils import timezone
        from datetime import timedelta
        
        # credentials dict should have 'organization_id'
        org_id = credentials.get('organization_id')
        if not org_id:
            return {
                'valid': False,
                'message': 'Organization ID required'
            }
        
        # Check for available agents
        cutoff = timezone.now() - timedelta(seconds=120)  # 2 minutes
        
        active_agents = Agent.objects.filter(
            organization__org_id=org_id,
            status__in=['ready', 'busy'],
            last_heartbeat__gte=cutoff
        ).count()
        
        if active_agents == 0:
            return {
                'valid': False,
                'message': 'No active agents available for organization. Please register an agent.',
                'details': {'active_agents': 0}
            }
        
        return {
            'valid': True,
            'message': f'{active_agents} active agent(s) available',
            'details': {'active_agents': active_agents}
        }
        
def get_adapter_for_provider(provider: ComputeProvider):
    """
    Factory function to get appropriate adapter for provider.
    """
    from compute.adapters import (
        PlatformGPUAdapter,
        SageMakerAdapter,
        VertexAIAdapter,
        AzureMLAdapter,
        KubernetesAdapter,
        OnPremiseAgentAdapter,  # ADD THIS
    )
    
    adapters = {
        'platform-gpu': PlatformGPUAdapter,
        'platform-cpu': PlatformGPUAdapter,
        'on-premise-agent': OnPremiseAgentAdapter,  # ADD THIS
        'aws-sagemaker': SageMakerAdapter,
        'gcp-vertex': VertexAIAdapter,
        'azure-ml': AzureMLAdapter,
        'kubernetes': KubernetesAdapter,
    }
    
    adapter_class = adapters.get(provider.provider_type)
    if not adapter_class:
        raise ValueError(f"No adapter for provider type: {provider.provider_type}")
    
    return adapter_class(provider)
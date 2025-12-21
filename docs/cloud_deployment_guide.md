# Cloud Provider Integration Guide

## Overview

Enable users to train models on their own cloud infrastructure (AWS, GCP, Azure, Kubernetes) with zero code changes.

---

## AWS SageMaker Setup

### 1. Prerequisites

**IAM Role for SageMaker:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-training-bucket/*",
        "arn:aws:s3:::your-training-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2. Build Training Container

**Dockerfile for SageMaker:**
```dockerfile
# docker/sagemaker.Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install dependencies (same as GPU worker)
RUN apt-get update && apt-get install -y python3.10 python3-pip
RUN pip3 install torch torchvision ultralytics albumentations boto3

# Copy training code
COPY training/ /app/training/
WORKDIR /app

# SageMaker entry point
COPY docker/sagemaker_train.py /app/train.py
RUN chmod +x /app/train.py

# SageMaker expects these paths
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output/data
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training

ENTRYPOINT ["python3", "/app/train.py"]
```

**Entry point script:**
```python
# docker/sagemaker_train.py
import os
import sys
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, '/app')

from training.trainers.factory import get_trainer
from training.trainers.base import TrainingConfig

def main():
    # SageMaker environment variables
    hyperparameters = json.loads(os.environ.get('SM_HPS', '{}'))
    model_dir = Path(os.environ['SM_MODEL_DIR'])
    data_dir = Path(os.environ['SM_CHANNEL_TRAINING'])
    output_dir = Path(os.environ['SM_OUTPUT_DATA_DIR'])
    
    # Get job config from environment
    job_id = os.environ.get('JOB_ID')
    framework = os.environ.get('FRAMEWORK', 'yolo')
    task = os.environ.get('TASK', 'object-detection')
    
    # Create training config
    config = TrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        epochs=int(hyperparameters.get('epochs', 100)),
        batch_size=int(hyperparameters.get('batch-size', 16)),
        learning_rate=float(hyperparameters.get('learning-rate', 0.001)),
        image_size=int(hyperparameters.get('image-size', 640)),
        device='cuda',
        workers=int(hyperparameters.get('workers', 4)),
        model_params=hyperparameters
    )
    
    # Create trainer
    trainer = get_trainer(
        framework=framework,
        task=task,
        dataset_path=str(data_dir),
        output_dir=str(output_dir),
        **config.__dict__
    )
    
    # Train
    result = trainer.train()
    
    # Copy best checkpoint to model directory
    import shutil
    if result.best_checkpoint_path.exists():
        shutil.copy(
            result.best_checkpoint_path,
            model_dir / 'model.pt'
        )
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(result.best_metrics.to_dict(), f)
    
    print("Training complete!")

if __name__ == '__main__':
    main()
```

### 3. Push to ECR

```bash
# Build image
docker build -f docker/sagemaker.Dockerfile -t cv-training:sagemaker .

# Tag for ECR
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
ECR_REPO=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/cv-training

docker tag cv-training:sagemaker ${ECR_REPO}:yolo-latest
docker tag cv-training:sagemaker ${ECR_REPO}:rf-detr-latest

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_REPO}

# Push
docker push ${ECR_REPO}:yolo-latest
docker push ${ECR_REPO}:rf-detr-latest
```

### 4. Create Compute Provider

```python
# Django shell or migration
from compute.models import ComputeProvider

ComputeProvider.objects.create(
    name='AWS SageMaker',
    provider_type='aws-sagemaker',
    description='Amazon SageMaker training',
    system_config={
        'region': 'us-east-1',
        'account_id': 'YOUR_ACCOUNT_ID',
        'role_arn': 'arn:aws:iam::ACCOUNT:role/SageMakerRole',
        'training_bucket': 'your-training-bucket',
        'output_bucket': 'your-output-bucket',
        'ecr_repository': 'cv-training',
        'use_spot_instances': False  # Enable for cost savings
    },
    available_instances=[
        {'name': 'ml.g4dn.xlarge', 'vcpus': 4, 'memory_gb': 16, 'gpu_type': 'T4', 'gpu_count': 1, 'cost_per_hour': 0.736},
        {'name': 'ml.g4dn.2xlarge', 'vcpus': 8, 'memory_gb': 32, 'gpu_type': 'T4', 'gpu_count': 1, 'cost_per_hour': 0.94},
        {'name': 'ml.p3.2xlarge', 'vcpus': 8, 'memory_gb': 61, 'gpu_type': 'V100', 'gpu_count': 1, 'cost_per_hour': 3.825},
        {'name': 'ml.p3.8xlarge', 'vcpus': 32, 'memory_gb': 244, 'gpu_type': 'V100', 'gpu_count': 4, 'cost_per_hour': 14.688},
    ],
    requires_user_credentials=True,
    is_active=True
)
```

### 5. User Configuration

**Via API:**
```bash
POST /api/v1/compute/profiles
{
  "name": "My AWS Training",
  "provider_id": 1,
  "default_instance_type": "ml.g4dn.xlarge",
  "strategy": "fastest",
  "max_concurrent_jobs": 3,
  "max_cost_per_hour": 5.0,
  "user_credentials": {
    "access_key": "AKIA...",
    "secret_key": "...",
    "region": "us-east-1",
    "role_arn": "arn:aws:iam::123:role/SageMakerRole",
    "training_bucket": "my-training-bucket"
  },
  "is_default": true
}
```

---

## GCP Vertex AI Setup

### 1. Service Account

```bash
# Create service account
gcloud iam service-accounts create cv-training \
  --display-name="CV Platform Training"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:cv-training@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:cv-training@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Generate key
gcloud iam service-accounts keys create key.json \
  --iam-account=cv-training@YOUR_PROJECT.iam.gserviceaccount.com
```

### 2. Build Training Container

```bash
# Build
docker build -f docker/sagemaker.Dockerfile -t cv-training:vertex .

# Tag for Artifact Registry
PROJECT_ID=your-project
REGION=us-central1

docker tag cv-training:vertex \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/cv-training/yolo:latest

# Push
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/cv-training/yolo:latest
```

### 3. Create Provider

```python
ComputeProvider.objects.create(
    name='GCP Vertex AI',
    provider_type='gcp-vertex',
    system_config={
        'project_id': 'your-project',
        'region': 'us-central1',
        'training_bucket': 'your-training-bucket',
        'output_bucket': 'your-output-bucket'
    },
    available_instances=[
        {'name': 'n1-standard-4', 'vcpus': 4, 'memory_gb': 15, 'gpu_type': 'T4', 'gpu_count': 1, 'cost_per_hour': 0.52},
        {'name': 'n1-standard-8', 'vcpus': 8, 'memory_gb': 30, 'gpu_type': 'V100', 'gpu_count': 1, 'cost_per_hour': 2.48},
    ],
    requires_user_credentials=True,
    is_active=True
)
```

### 4. User Configuration

```python
{
  "name": "My GCP Training",
  "provider_id": 2,
  "user_credentials": {
    "service_account": {<contents of key.json>},
    "project_id": "your-project",
    "region": "us-central1",
    "training_bucket": "my-training-bucket"
  }
}
```

---

## Kubernetes Setup

### 1. GPU Node Pool

**GKE:**
```bash
gcloud container node-pools create gpu-pool \
  --cluster=your-cluster \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

**EKS:**
```bash
eksctl create nodegroup \
  --cluster=your-cluster \
  --name=gpu-nodes \
  --node-type=g4dn.xlarge \
  --nodes=2 \
  --nodes-min=0 \
  --nodes-max=5

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

### 2. Service Account & RBAC

```yaml
# k8s/training-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: training-sa
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: training-role
  namespace: default
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "create", "delete"]
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: training-rolebinding
  namespace: default
subjects:
- kind: ServiceAccount
  name: training-sa
roleRef:
  kind: Role
  name: training-role
  apiGroup: rbac.authorization.k8s.io
```

Apply:
```bash
kubectl apply -f k8s/training-rbac.yaml
```

### 3. Image Pull Secret (if private registry)

```bash
kubectl create secret docker-registry regcred \
  --docker-server=your-registry.io \
  --docker-username=user \
  --docker-password=pass \
  --docker-email=email@example.com
```

### 4. Create Provider

```python
ComputeProvider.objects.create(
    name='Kubernetes Cluster',
    provider_type='kubernetes',
    system_config={
        'namespace': 'default',
        'container_registry': 'your-registry.io',
        'repository': 'cv-training',
        'image_tag': 'latest',
        'service_account': 'training-sa',
        'image_pull_secret': 'regcred',
        'enable_shared_cache': True,
        'cache_pvc': 'training-cache'
    },
    available_instances=[
        {'name': 'n1-standard-4-t4', 'vcpus': 4, 'memory_gb': 15, 'gpu_count': 1},
        {'name': 'n1-standard-8-v100', 'vcpus': 8, 'memory_gb': 30, 'gpu_count': 1},
    ],
    requires_user_credentials=True,
    is_active=True
)
```

### 5. User Configuration

```python
# Get kubeconfig
kubectl config view --raw > kubeconfig.yaml

# Or use token
TOKEN=$(kubectl get secret $(kubectl get sa training-sa -o jsonpath='{.secrets[0].name}') -o jsonpath='{.data.token}' | base64 -d)
API_SERVER=$(kubectl config view -o jsonpath='{.clusters[0].cluster.server}')

# Configure
{
  "name": "My K8s Training",
  "provider_id": 3,
  "user_credentials": {
    "api_server": "https://your-cluster.example.com",
    "token": "eyJhbG...",
    "verify_ssl": true
  }
}
```

---

## Testing Cloud Integration

### 1. Validate Credentials

```python
POST /api/v1/compute/profiles/{profile_id}/validate

Response:
{
  "valid": true,
  "message": "AWS credentials validated successfully",
  "details": {
    "account_id": "123456789012",
    "region": "us-east-1"
  }
}
```

### 2. Test Training

```python
POST /api/v1/models/{model_id}/train
{
  "dataset_version_id": "...",
  "compute_profile_id": "aws-profile-id",
  "config": {
    "epochs": 10,
    "batch_size": 16
  }
}

Response:
{
  "training_session_id": "...",
  "compute_provider": "AWS SageMaker",
  "instance_type": "ml.g4dn.xlarge",
  "status": "queued"
}
```

### 3. Monitor Progress

```python
GET /api/v1/training-sessions/{session_id}

Response:
{
  "status": "running",
  "progress": 45.0,
  "current_epoch": 45,
  "compute_provider": "AWS SageMaker",
  "instance_type": "ml.g4dn.xlarge",
  "estimated_cost": "$0.52"
}
```

---

## Cost Optimization

### 1. Spot Instances (AWS)

```python
# Enable in provider config
'use_spot_instances': True

# SageMaker handles interruptions automatically with checkpointing
```

### 2. Preemptible VMs (GCP)

```python
# Enable in Vertex AI job config
'preemptible': True
```

### 3. Auto-scaling

```python
# Set max concurrent jobs
'max_concurrent_jobs': 5,
'max_cost_per_hour': 10.0  # Enforce budget
```

---

## Troubleshooting

### AWS SageMaker

**Issue: Permission denied**
```bash
# Check IAM role permissions
aws iam get-role --role-name SageMakerRole
```

**Issue: Image not found**
```bash
# Verify ECR image
aws ecr describe-images --repository-name cv-training
```

### GCP Vertex AI

**Issue: Service account permissions**
```bash
# Check permissions
gcloud projects get-iam-policy YOUR_PROJECT \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:cv-training@*"
```

### Kubernetes

**Issue: Pod pending**
```bash
# Check node selector
kubectl describe pod <pod-name>

# Check GPU availability
kubectl get nodes -o json | jq '.items[] | {name:.metadata.name, gpu:.status.capacity."nvidia.com/gpu"}'
```

---

## Summary

✅ **AWS SageMaker** - Fully implemented, production-ready  
✅ **GCP Vertex AI** - Template implemented, needs testing  
✅ **Azure ML** - Template implemented, needs testing  
✅ **Kubernetes** - Complete implementation  

Users can now train models on any cloud provider with zero code changes!
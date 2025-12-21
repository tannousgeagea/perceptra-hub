# GPU Training Workers - Complete Setup Guide

## Overview

Production-ready GPU training infrastructure using **Docker + Celery + NVIDIA GPUs**.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Server                            │
│              (Triggers training jobs)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
             ┌───────────────┐
             │  Redis Queue  │
             │  (Broker)     │
             └───────┬───────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│ GPU Worker 0  │         │ GPU Worker 1  │
│   (GPU 0)     │         │   (GPU 1)     │
│               │         │               │
│ ▸ Train YOLO  │         │ ▸ Train DETR  │
│ ▸ Export ONNX │         │ ▸ Upload S3   │
│ ▸ Track Metrics│         │ ▸ Update DB   │
└───────────────┘         └───────────────┘
```

---

## Prerequisites

### 1. NVIDIA GPU Setup

```bash
# Verify GPU
nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Docker & Docker Compose

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Verify
docker compose version
```

---

## Quick Start

### 1. Setup

```bash
# Clone repo
cd your-project

# Make script executable
chmod +x scripts/gpu-workers.sh

# Check prerequisites
./scripts/gpu-workers.sh check

# Setup environment
./scripts/gpu-workers.sh setup
# Edit .env file with your credentials
```

### 2. Configure Environment

Edit `.env`:
```bash
# Celery
CELERY_BROKER_URL=redis://redis:6379/0

# Database
DATABASE_URL=postgresql://user:pass@db:5432/cvplatform

# Storage (S3/GCS/Azure)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Credentials encryption
COMPUTE_CREDENTIALS_KEY=<generate-with-fernet>
```

Generate encryption key:
```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

### 3. Build & Start

```bash
# Build images
./scripts/gpu-workers.sh build

# Start workers (auto-detects GPU count)
./scripts/gpu-workers.sh start

# View status
./scripts/gpu-workers.sh status
```

### 4. Monitor

```bash
# Flower UI
open http://localhost:5555

# View logs
./scripts/gpu-workers.sh logs gpu-worker-0

# Health check
./scripts/gpu-workers.sh health

# GPU stats
docker exec cv-gpu-worker-0 nvidia-smi
```

---

## Production Deployment

### Multi-GPU Setup

Automatically scales based on detected GPUs:

```bash
# Detects all GPUs and starts workers
./scripts/gpu-workers.sh start

# Manual scaling
docker-compose -f docker/docker-compose.gpu.yml \
  --profile multi-gpu up -d
```

Each GPU gets its own worker with:
- Isolated CUDA context (`CUDA_VISIBLE_DEVICES`)
- Dedicated cache directory
- Independent task queue

### Resource Limits

Edit `docker-compose.gpu.yml`:

```yaml
gpu-worker-0:
  deploy:
    resources:
      limits:
        cpus: '8'
        memory: 32G
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']
            capabilities: [gpu]
```

### High Availability

```yaml
# Add restart policy
restart: unless-stopped

# Add health checks
healthcheck:
  test: ["CMD", "python3", "/healthcheck.py"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Configuration

### Celery Settings

**Key configurations in `celery.py`:**

```python
# Worker settings
worker_prefetch_multiplier = 1      # One task at a time
worker_max_tasks_per_child = 1     # Restart after each task (GPU memory)
task_acks_late = True               # Acknowledge after completion

# Task limits
task_time_limit = 86400             # 24 hours max
task_soft_time_limit = 82800        # 23 hours soft limit

# Queue configuration
task_queues = (
    Queue('gpu-training', priority=10),  # High priority
    Queue('cpu-training', priority=5),
)
```

### Training Task Configuration

```python
# In your training trigger
training_config = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.001,
    'image_size': 640,
    'workers': 4,
    'optimizer': 'Adam',
    'scheduler': 'cosine',
    'augmentation': True,
    'model_params': {
        'model_size': 'n',  # YOLO specific
        'version': '11'
    }
}
```

---

## Monitoring & Debugging

### View Active Jobs

```bash
# Via Flower UI
open http://localhost:5555

# Via CLI
docker exec cv-gpu-worker-0 celery -A your_project inspect active

# Query Redis
docker exec -it cv-redis redis-cli
> LLEN gpu-training  # Queue length
> KEYS celery-task-meta-*  # Task results
```

### GPU Monitoring

```python
# API endpoint
GET /api/v1/monitoring/gpu

Response:
{
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA RTX 4090",
      "memory": {
        "used_gb": 12.5,
        "total_gb": 24.0,
        "utilization_percent": 52
      },
      "utilization_percent": 85,
      "temperature_c": 68,
      "processes": {"count": 1}
    }
  ],
  "health": {
    "healthy": true,
    "warnings": [],
    "issues": []
  }
}
```

### Log Aggregation

```bash
# Real-time logs
docker-compose -f docker/docker-compose.gpu.yml logs -f

# Specific worker
docker logs -f cv-gpu-worker-0

# Training session logs (from storage)
GET /api/v1/training-sessions/{id}
# Returns logs_url for download
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size in config
'batch_size': 8  # Instead of 16

# Enable gradient accumulation
'gradient_accumulation': 2
```

Workers automatically restart after each task to clear memory.

### Issue: Worker Not Starting

```bash
# Check logs
./scripts/gpu-workers.sh logs gpu-worker-0

# Run health check
./scripts/gpu-workers.sh health

# Verify GPU access
docker exec cv-gpu-worker-0 nvidia-smi
```

### Issue: Tasks Stuck in Queue

```bash
# Check worker connection
docker exec cv-gpu-worker-0 celery -A your_project inspect ping

# Purge queue (CAUTION)
./scripts/gpu-workers.sh purge

# Restart workers
./scripts/gpu-workers.sh restart
```

### Issue: Slow Training

**Checklist:**
- ✓ Using GPU (not CPU fallback)
- ✓ Batch size optimal for GPU
- ✓ Dataset on fast storage (SSD)
- ✓ Sufficient CPU workers for data loading
- ✓ No other processes using GPU

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Should see ~90-100% GPU utilization during training
```

---

## Maintenance

### Update Workers

```bash
# Pull latest code
git pull

# Rebuild images
./scripts/gpu-workers.sh build

# Restart workers (zero-downtime)
docker-compose -f docker/docker-compose.gpu.yml up -d --no-deps --build gpu-worker-0
```

### Clean Up

```bash
# Stop workers
./scripts/gpu-workers.sh stop

# Remove containers and volumes
docker-compose -f docker/docker-compose.gpu.yml down -v

# Clean Docker cache
docker system prune -a

# Clean training temp files
docker exec cv-gpu-worker-0 rm -rf /tmp/training/*
```

### Backup

```bash
# Backup Redis (task queue)
docker exec cv-redis redis-cli BGSAVE

# Backup training metrics
pg_dump -t gpu_metrics > gpu_metrics_backup.sql
```

---

## Scaling

### Horizontal Scaling

```bash
# Add more GPUs on same machine
./scripts/gpu-workers.sh scale 4

# Add workers on different machines
# On machine 2:
CELERY_BROKER_URL=redis://machine1:6379/0 \
docker-compose -f docker/docker-compose.gpu.yml up -d
```

### Vertical Scaling

```yaml
# Increase resources per worker
gpu-worker-0:
  deploy:
    resources:
      limits:
        memory: 64G  # More RAM
        cpus: '16'   # More CPUs
```

---

## Cost Optimization

### 1. Auto-Scaling

Implement auto-scaling based on queue length:

```python
# Monitor queue
queue_length = celery_app.control.inspect().active()

# Scale workers
if queue_length > 10:
    scale_up()
elif queue_length == 0:
    scale_down()
```

### 2. Spot Instances (Cloud)

For cloud deployments, use spot/preemptible instances:

```bash
# AWS EC2 Spot
aws ec2 run-instances --instance-type g4dn.xlarge \
  --spot-price "0.50" ...

# Handle interruptions gracefully
# Workers checkpoint every N epochs
```

### 3. Scheduled Training

Train during off-peak hours:

```python
# Celery Beat schedule
'train-nightly': {
    'task': 'training.tasks.scheduled_training',
    'schedule': crontab(hour=2, minute=0),  # 2 AM
}
```

---

## Security

### 1. Credentials

```bash
# Encrypt credentials
from cryptography.fernet import Fernet

cipher = Fernet(settings.COMPUTE_CREDENTIALS_KEY.encode())
encrypted = cipher.encrypt(json.dumps(credentials).encode())
```

### 2. Network Isolation

```yaml
# docker-compose.yml
networks:
  training:
    driver: bridge
    internal: true  # No external access
```

### 3. Resource Limits

```yaml
# Prevent resource exhaustion
ulimits:
  nproc: 65535
  nofile:
    soft: 65535
    hard: 65535
```

---

## Performance Benchmarks

**Typical training times (YOLO11n, COCO dataset):**

| GPU | Epochs | Time | Cost/hour |
|-----|--------|------|-----------|
| RTX 4090 | 100 | 3h | $0 (platform) |
| V100 | 100 | 5h | $2.48 (AWS) |
| T4 | 100 | 8h | $0.526 (GCP) |
| CPU | 100 | 48h | Not recommended |

**Optimization tips:**
- Use mixed precision (AMP): 2x faster
- Increase batch size: Better GPU utilization
- Use SSD for dataset: Faster data loading
- Enable multi-worker data loading

---

## Next Steps

✅ **Step 3 Complete!** Your GPU workers are running.

**What's next:**
1. Test end-to-end training workflow
2. Set up monitoring dashboards (Grafana)
3. Implement cloud provider adapters (SageMaker, Vertex AI)
4. Add advanced features (distributed training, hyperparameter tuning)

---

## Support

**Common Commands:**
```bash
./scripts/gpu-workers.sh status    # Check status
./scripts/gpu-workers.sh logs      # View logs
./scripts/gpu-workers.sh health    # Health check
./scripts/gpu-workers.sh monitor   # Open Flower UI
```

**Resources:**
- Celery Docs: https://docs.celeryq.dev
- NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker
- Ultralytics YOLO: https://docs.ultralytics.com
# Dockerfile for GPU Training Workers
# Location: docker/gpu-worker.Dockerfile

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install training frameworks
RUN pip3 install --no-cache-dir \
    ultralytics==8.1.0 \
    albumentations==1.3.1 \
    opencv-python-headless==4.8.1.78 \
    torchmetrics==1.2.0 \
    pycocotools==2.0.7

# Install Celery and dependencies
RUN pip3 install --no-cache-dir \
    celery[redis]==5.3.4 \
    redis==5.0.1 \
    boto3==1.34.10 \
    google-cloud-storage==2.14.0 \
    azure-storage-blob==12.19.0

# Install monitoring and utilities
RUN pip3 install --no-cache-dir \
    psutil==5.9.6 \
    gpustat==1.1.1 \
    prometheus-client==0.19.0

# Install Django and platform dependencies
RUN pip3 install --no-cache-dir \
    django==4.2.8 \
    psycopg2-binary==2.9.9 \
    cryptography==41.0.7

# Create app directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create directories for training
RUN mkdir -p /tmp/training /tmp/datasets

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check script
COPY docker/healthcheck.py /healthcheck.py
RUN chmod +x /healthcheck.py

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /healthcheck.py

# Entry point
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["celery", "-A", "your_project", "worker", \
     "--queue=gpu-training", \
     "--concurrency=1", \
     "--pool=solo", \
     "--loglevel=info", \
     "--max-tasks-per-child=1"]
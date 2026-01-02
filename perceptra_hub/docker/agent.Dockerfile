FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.1
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install training frameworks
RUN pip3 install ultralytics==8.3.0  # YOLO
RUN pip3 install transformers timm  # RT-DETR

# Install agent dependencies
RUN pip3 install \
    requests==2.31.0 \
    gputil==1.4.0 \
    psutil==5.9.8 \
    pyyaml==6.0.1 \
    pillow==10.3.0 \
    opencv-python==4.9.0.80 \
    numpy==1.26.4 \
    pandas==2.2.2

# Create work directories
RUN mkdir -p /app /tmp/agent-work/datasets /tmp/agent-work/outputs

# Copy agent application
COPY agent/ /app/agent/
COPY training/trainers/ /app/training/trainers/

# Set working directory
WORKDIR /app

# Set environment
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Run agent
CMD ["python3", "agent/main.py"]

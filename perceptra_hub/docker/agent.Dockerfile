# syntax=docker/dockerfile:1
# Compute agent: runs training jobs on a GPU machine and reports back to the hub.
#
# CI publishes it with the hub's release tag (.github/workflows/docker-build.yml),
# so a hub at vX.Y.Z hands new agents tannousgeagea/perceptra-agent:vX.Y.Z.
# Local build, from perceptra_hub/:
#   docker build -f docker/agent.Dockerfile --build-arg AGENT_VERSION=dev -t perceptra-agent:dev .

# Python, PyTorch, torchvision and CUDA come with the base image; CUDA 12.8
# also covers the newest (Blackwell) cards.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

# OpenCV, pulled in by ultralytics, needs these at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Agent and trainer dependencies (torch itself stays the base image's). The base
# image installs its packages into the system Python too, hence the flag.
COPY agent/requirements.txt /tmp/agent-requirements.txt
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-cache-dir -r /tmp/agent-requirements.txt

# Create work directories
RUN mkdir -p /app /tmp/agent-work/datasets /tmp/agent-work/outputs

# Copy agent application. --link keeps these layers independent of the ones
# above, so a code-only change rebuilds (and pushes) just these.
COPY --link agent/ /app/agent/
COPY --link training/trainers/ /app/training/trainers/

# Set working directory
WORKDIR /app

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# The release this image belongs to (the hub's tag); sent with every heartbeat
ARG AGENT_VERSION=dev
ENV AGENT_VERSION=${AGENT_VERSION}
LABEL org.opencontainers.image.title="Malumetrix compute agent" \
      org.opencontainers.image.version="${AGENT_VERSION}"

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Run agent
CMD ["python3", "agent/main.py"]

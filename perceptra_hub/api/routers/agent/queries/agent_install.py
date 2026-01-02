"""
Agent install script endpoint.
File: api/routers/install.py
"""

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/agents")

INSTALL_SCRIPT_TEMPLATE = """#!/bin/bash
set -e

# Colors
GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

# Parse arguments
API_URL=""
AGENT_KEY=""
AGENT_SECRET=""
AGENT_ID=""
REGISTRY="{registry}"
IMAGE_TAG="{image_tag}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --api-url) API_URL="$2"; shift 2 ;;
    --key) AGENT_KEY="$2"; shift 2 ;;
    --secret) AGENT_SECRET="$2"; shift 2 ;;
    --id) AGENT_ID="$2"; shift 2 ;;
    --registry) REGISTRY="$2"; shift 2 ;;
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Validate
if [ -z "$API_URL" ] || [ -z "$AGENT_KEY" ] || [ -z "$AGENT_SECRET" ]; then
    echo -e "${{RED}}Error: Missing required arguments${{NC}}"
    echo "Usage: bash install-agent.sh --api-url URL --key KEY --secret SECRET --id ID"
    exit 1
fi

echo -e "${{GREEN}}CV Training Agent Installer${{NC}}"
echo "================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${{RED}}Error: Docker not installed${{NC}}"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check NVIDIA Docker
echo "Checking NVIDIA Docker support..."
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${{RED}}Error: NVIDIA Docker not working${{NC}}"
    echo "Install nvidia-container-toolkit:"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    exit 1
fi

echo -e "${{GREEN}}✓ Docker and NVIDIA support OK${{NC}}"

# Pull image
echo "Pulling agent image..."
docker pull $REGISTRY/cv-training-agent:$IMAGE_TAG

# Stop existing agent
if [ "$(docker ps -q -f name=cv-training-agent)" ]; then
    echo "Stopping existing agent..."
    docker stop cv-training-agent
    docker rm cv-training-agent
fi

# Run agent
echo "Starting agent..."
docker run -d \\
  --name cv-training-agent \\
  --gpus all \\
  --restart unless-stopped \\
  -e API_URL="$API_URL" \\
  -e AGENT_KEY="$AGENT_KEY" \\
  -e AGENT_SECRET="$AGENT_SECRET" \\
  -e AGENT_ID="$AGENT_ID" \\
  -v agent-datasets:/tmp/agent-work/datasets \\
  -v agent-outputs:/tmp/agent-work/outputs \\
  $REGISTRY/cv-training-agent:$IMAGE_TAG

echo ""
echo -e "${{GREEN}}✓ Agent installed and running!${{NC}}"
echo ""
echo "Commands:"
echo "  View logs:    docker logs -f cv-training-agent"
echo "  Stop agent:   docker stop cv-training-agent"
echo "  Start agent:  docker start cv-training-agent"
echo "  Restart:      docker restart cv-training-agent"
echo "  Remove:       docker stop cv-training-agent && docker rm cv-training-agent"
"""

@router.get("/install-agent.sh", response_class=PlainTextResponse)
async def get_install_script(
    registry: str = Query("your-registry.io", description="Docker registry"),
    tag: str = Query("latest", description="Image tag")
):
    """
    Generate agent install script.
    Usage: curl -fsSL https://platform.com/install-agent.sh | bash -s -- --key KEY --secret SECRET
    """
    return INSTALL_SCRIPT_TEMPLATE.format(
        registry=registry,
        image_tag=tag
    )
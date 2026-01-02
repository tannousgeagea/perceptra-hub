# On-Premise Training Agent Setup

## Quick Install

### Option 1: One-liner (Recommended)
```bash
curl -fsSL https://your-platform.com/install-agent.sh | bash -s -- \\
  --api-url "https://your-platform.com" \\
  --key "agent_abc123" \\
  --secret "your-secret-key" \\
  --id "agent_xyz789"
```

### Option 2: Docker Run
```bash
docker run -d \\
  --name cv-training-agent \\
  --gpus all \\
  --restart unless-stopped \\
  -e API_URL="https://your-platform.com" \\
  -e AGENT_KEY="agent_abc123" \\
  -e AGENT_SECRET="your-secret-key" \\
  -e AGENT_ID="agent_xyz789" \\
  -v agent-datasets:/tmp/agent-work/datasets \\
  -v agent-outputs:/tmp/agent-work/outputs \\
  your-registry.io/cv-training-agent:latest
```

### Option 3: Docker Compose
Save as `docker-compose.yml`:
```yaml
version: '3.8'
services:
  agent:
    image: your-registry.io/cv-training-agent:latest
    container_name: cv-training-agent
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - API_URL=https://your-platform.com
      - AGENT_KEY=agent_abc123
      - AGENT_SECRET=your-secret-key
      - AGENT_ID=agent_xyz789
    volumes:
      - agent-datasets:/tmp/agent-work/datasets
      - agent-outputs:/tmp/agent-work/outputs

volumes:
  agent-datasets:
  agent-outputs:
```

Run: `docker compose up -d`

## Prerequisites

1. **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
2. **NVIDIA GPU drivers** - Driver version 450+ 
3. **NVIDIA Container Toolkit**:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Verify Installation

```bash
# Check agent is running
docker ps -f name=cv-training-agent

# View logs
docker logs -f cv-training-agent

# Test GPU access
docker exec cv-training-agent nvidia-smi
```

## Management

```bash
# View logs
docker logs -f cv-training-agent

# Stop agent
docker stop cv-training-agent

# Start agent
docker start cv-training-agent

# Restart agent
docker restart cv-training-agent

# Remove agent
docker stop cv-training-agent && docker rm cv-training-agent

# Update agent
docker pull your-registry.io/cv-training-agent:latest
docker stop cv-training-agent
docker rm cv-training-agent
# Re-run install command
```

## Multi-GPU Selection

Use specific GPUs:
```bash
# Use GPU 0 only
docker run -d ... -e CUDA_VISIBLE_DEVICES=0 ...

# Use GPU 1 and 2
docker run -d ... -e CUDA_VISIBLE_DEVICES=1,2 ...

# Use all GPUs (default)
docker run -d ... --gpus all ...
```

## Troubleshooting

**Agent not connecting:**
- Check API_URL is correct
- Verify AGENT_KEY and AGENT_SECRET
- Check network connectivity: `docker exec cv-training-agent curl https://your-platform.com`

**GPU not detected:**
- Verify NVIDIA drivers: `nvidia-smi`
- Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Restart Docker daemon: `sudo systemctl restart docker`

**Out of memory:**
- Reduce batch size in training config
- Use fewer concurrent jobs per agent
- Monitor GPU memory: `watch nvidia-smi`
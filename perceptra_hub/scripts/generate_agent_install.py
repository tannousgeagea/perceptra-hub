"""
Generate one-liner install command for agents.
Used by API when registering new agent.
"""

def generate_install_command(
    api_url: str,
    agent_key: str,
    agent_secret: str,
    agent_id: str,
    image: str,
    gpu_ids: str = "all"
) -> dict:
    """
    Generate install commands for agent deployment.

    image: the full image reference, e.g. tannousgeagea/perceptra-agent:v0.14.0

    Returns:
        dict with 'docker_run', 'docker_compose', 'curl_script'
    """
    
    # Clean URL
    api_url = api_url.rstrip('/')
    
    # Docker run command (one-liner)
    docker_run = f"""docker run -d \\
  --name cv-training-agent \\
  --gpus {gpu_ids} \\
  --shm-size=8g \\
  --restart unless-stopped \\
  -e API_URL="{api_url}" \\
  -e AGENT_KEY="{agent_key}" \\
  -e AGENT_SECRET="{agent_secret}" \\
  -e AGENT_ID="{agent_id}" \\
  -v agent-datasets:/tmp/agent-work/datasets \\
  -v agent-outputs:/tmp/agent-work/outputs \\
  {image}"""
    
    # Docker Compose
    docker_compose = f"""version: '3.8'

services:
  agent:
    image: {image}
    container_name: cv-training-agent
    restart: unless-stopped
    # PyTorch dataloader workers share batches through /dev/shm (Docker's 64 MB default isn't enough)
    shm_size: 8gb
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {gpu_ids}
              capabilities: [gpu]
    environment:
      - API_URL={api_url}
      - AGENT_KEY={agent_key}
      - AGENT_SECRET={agent_secret}
      - AGENT_ID={agent_id}
    volumes:
      - agent-datasets:/tmp/agent-work/datasets
      - agent-outputs:/tmp/agent-work/outputs

volumes:
  agent-datasets:
  agent-outputs:"""
    
    # Curl install script (downloads and runs)
    curl_script = f"""curl -fsSL {api_url}/api/v1/agents/install-agent.sh | bash -s -- \\
  --image "{image}" \\
  --key "{agent_key}" \\
  --secret "{agent_secret}" \\
  --id "{agent_id}" \\
  --api-url "{api_url}" """
    
    return {
        'docker_run': docker_run,
        'docker_compose': docker_compose,
        'curl_script': curl_script
    }
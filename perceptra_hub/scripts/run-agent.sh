#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}CV Training Agent - Runner${NC}"
echo "==============================="

# Check if .env exists
if [ ! -f "agent/.env" ]; then
    echo -e "${RED}Error: agent/.env not found${NC}"
    echo "Copy agent/.env.example to agent/.env and configure your credentials"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    exit 1
fi

# Check NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA Docker not working${NC}"
    echo "Install nvidia-container-toolkit and restart Docker daemon"
    exit 1
fi

case "$1" in
    start)
        echo -e "${GREEN}Starting agent...${NC}"
        docker compose -f docker/agent-compose.yml up -d
        echo -e "${GREEN}Agent started!${NC}"
        echo "View logs: docker logs -f cv-training-agent"
        ;;
    
    stop)
        echo -e "${YELLOW}Stopping agent...${NC}"
        docker compose -f docker/agent-compose.yml down
        echo -e "${GREEN}Agent stopped${NC}"
        ;;
    
    restart)
        echo -e "${YELLOW}Restarting agent...${NC}"
        docker compose -f docker/agent-compose.yml restart
        echo -e "${GREEN}Agent restarted${NC}"
        ;;
    
    logs)
        docker logs -f cv-training-agent
        ;;
    
    status)
        docker ps -f name=cv-training-agent
        ;;
    
    build)
        echo -e "${GREEN}Building agent image...${NC}"
        docker compose -f docker/agent-compose.yml build
        echo -e "${GREEN}Build complete${NC}"
        ;;
    
    shell)
        docker exec -it cv-training-agent /bin/bash
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|build|shell}"
        exit 1
        ;;
esac
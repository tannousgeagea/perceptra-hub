#!/bin/bash
# GPU Worker Management Scripts
# Location: scripts/gpu-workers.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker runtime not configured"
        log_info "Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    
    log_info "✓ Prerequisites OK"
}

# Setup environment
setup_env() {
    log_info "Setting up environment..."
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_warn ".env file not found, creating from template..."
        cat > "$PROJECT_ROOT/.env" << EOF
# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Database
DATABASE_URL=postgresql://user:password@db:5432/cvplatform

# AWS (if using S3)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

# Credentials encryption key (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
COMPUTE_CREDENTIALS_KEY=generate_a_key_here

# Django
DJANGO_SECRET_KEY=your_django_secret_key
DEBUG=False
EOF
        log_warn "Please edit .env file with your credentials"
        exit 1
    fi
    
    log_info "✓ Environment configured"
}

# Build images
build_images() {
    log_info "Building GPU worker images..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.gpu.yml build
    
    log_info "✓ Images built"
}

# Start workers
start_workers() {
    log_info "Starting GPU workers..."
    
    cd "$PROJECT_ROOT"
    
    # Detect GPU count
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log_info "Detected $GPU_COUNT GPU(s)"
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        log_info "Starting multi-GPU setup..."
        docker-compose -f docker/docker-compose.gpu.yml --profile multi-gpu up -d
    else
        log_info "Starting single-GPU setup..."
        docker-compose -f docker/docker-compose.gpu.yml up -d redis gpu-worker-0 celery-beat flower
    fi
    
    log_info "✓ Workers started"
    log_info "Flower UI: http://localhost:5555"
}

# Stop workers
stop_workers() {
    log_info "Stopping GPU workers..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.gpu.yml down
    
    log_info "✓ Workers stopped"
}

# Restart workers
restart_workers() {
    stop_workers
    sleep 2
    start_workers
}

# View logs
view_logs() {
    local service=${1:-gpu-worker-0}
    
    log_info "Viewing logs for $service..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.gpu.yml logs -f "$service"
}

# Show status
show_status() {
    log_info "GPU Worker Status:"
    echo ""
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.gpu.yml ps
    
    echo ""
    log_info "GPU Status:"
    docker exec cv-gpu-worker-0 nvidia-smi
    
    echo ""
    log_info "Active Training Jobs:"
    docker exec cv-gpu-worker-0 celery -A your_project inspect active
}

# Run healthcheck
healthcheck() {
    log_info "Running health checks..."
    
    for worker in $(docker ps --filter "name=gpu-worker" --format "{{.Names}}"); do
        log_info "Checking $worker..."
        if docker exec "$worker" python3 /healthcheck.py; then
            log_info "✓ $worker healthy"
        else
            log_error "✗ $worker unhealthy"
        fi
    done
}

# Scale workers
scale_workers() {
    local count=$1
    
    if [ -z "$count" ]; then
        log_error "Usage: $0 scale <number>"
        exit 1
    fi
    
    log_info "Scaling to $count workers..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.gpu.yml up -d --scale gpu-worker-0="$count"
    
    log_info "✓ Scaled to $count workers"
}

# Purge tasks
purge_tasks() {
    log_warn "⚠️  This will purge all queued tasks!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Purging tasks..."
        docker exec cv-gpu-worker-0 celery -A your_project purge -f
        log_info "✓ Tasks purged"
    fi
}

# Monitor workers
monitor() {
    log_info "Opening Flower monitoring..."
    log_info "URL: http://localhost:5555"
    
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:5555
    elif command -v open &> /dev/null; then
        open http://localhost:5555
    fi
}

# Usage
usage() {
    cat << EOF
GPU Worker Management

Usage: $0 <command> [options]

Commands:
    check           Check prerequisites
    setup           Setup environment
    build           Build Docker images
    start           Start workers
    stop            Stop workers
    restart         Restart workers
    status          Show worker status
    logs [service]  View logs (default: gpu-worker-0)
    health          Run health checks
    scale <n>       Scale to n workers
    purge           Purge all queued tasks
    monitor         Open monitoring UI
    
Examples:
    $0 start                    # Start workers
    $0 logs gpu-worker-0        # View worker logs
    $0 scale 2                  # Scale to 2 workers
    $0 status                   # Show status
    
EOF
}

# Main
main() {
    case "${1:-}" in
        check)
            check_prerequisites
            ;;
        setup)
            check_prerequisites
            setup_env
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        start)
            check_prerequisites
            start_workers
            ;;
        stop)
            stop_workers
            ;;
        restart)
            restart_workers
            ;;
        status)
            show_status
            ;;
        logs)
            view_logs "${2:-gpu-worker-0}"
            ;;
        health)
            healthcheck
            ;;
        scale)
            scale_workers "$2"
            ;;
        purge)
            purge_tasks
            ;;
        monitor)
            monitor
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
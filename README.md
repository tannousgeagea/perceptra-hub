## Project Overview

Perceptra Hub is a Computer Vision MLOps platform for managing datasets, training ML models, and orchestrating inference pipelines. It uses a **Django + FastAPI hybrid** architecture with Celery for async task processing.

## Development Environment

All services run via Docker Compose:
```bash
docker-compose build
docker-compose up -d
```

Services: PostgreSQL, Redis, RabbitMQ, Nginx (media proxy), and the main app (perceptrahub).

### Ports
- **29081** — Django admin (Gunicorn WSGI)
- **29082-29083** — FastAPI data API (Gunicorn + Uvicorn)
- **29084** — FastAPI event API

### Running Django Commands
```bash
docker-compose exec perceptrahub python /perceptra_hub/manage.py makemigrations
docker-compose exec perceptrahub python /perceptra_hub/manage.py migrate
docker-compose exec perceptrahub python /perceptra_hub/manage.py test
```

## Architecture

### Dual Framework Design
- **Django 4.2**: ORM, migrations, admin UI (django-unfold), user model, and all data models
- **FastAPI**: Two separate apps — `api/` (data ingestion) and `event_api/` (async event processing)
- Both frameworks share the same Django ORM models via `django.setup()` in FastAPI entry points

### Entry Points
- `perceptra_hub/perceptra_hub/wsgi.py` — Django WSGI app
- `perceptra_hub/api/main.py` — FastAPI data API (`create_app()` factory)
- `perceptra_hub/event_api/main.py` — FastAPI event API (`create_app()` factory)
- Process management via `supervisord.conf` (7+ programs)

### Multi-Tenancy
- Organizations are the root tenant container
- Projects are scoped to organizations
- Tenant context passed via `X-Organization-ID` header
- Role-based access through memberships app

### Async Task Processing
- **Broker**: RabbitMQ
- **Workers**: Celery with multiple specialized workers (train, create_version, annotation_audit, activity)
- **Scheduler**: Celery Beat for periodic tasks (metrics aggregation, cleanup, health checks)
- **Task routing**: Dynamic queue routing based on task name prefix (`queue:task`)
- Config in `api/config/celery_config.py` and `event_api/config/`

### FastAPI Router Auto-Discovery
Routers in `api/routers/` and `event_api/routers/` are auto-imported and mounted. To add a new API endpoint, create a new router module in the appropriate directory.

### Authentication
- Custom email-based user model (extends AbstractUser) in `users/`
- JWT Bearer tokens for FastAPI endpoints
- OAuth2 support for Microsoft and Google
- API key auth via `api_keys/` app
- Auth dependencies in `api/dependencies.py`

### Storage
Uses pluggable `perceptra-storage` package supporting S3, GCS, Azure Blob, and local backends.

### ML Training
- Strategy pattern with ComputeProfile selection
- Providers: Platform GPU, AWS SageMaker, GCP Vertex AI, Kubernetes
- Trainer adapters in `training/trainers/` (YOLO, RT-DETR, custom)
- Automatic provider fallback on failure

## CI/CD

- **versioning.yml**: Auto semantic version tagging on push to `main` (commit prefix `feat:` = minor bump, else patch)
- **docker-build.yml**: Builds and pushes to Docker Hub (`tannousgeagea/perceptrahub:{tag}`)

## Key Configuration

- Django settings: `perceptra_hub/perceptra_hub/settings.py`
- Database: PostgreSQL (configured via `DATABASE_*` env vars)
- Cache: Redis (via `REDIS_HOST`, `REDIS_PORT`)
- Message broker: RabbitMQ (via `RABBITMQ_*` env vars)
- No linting/formatting tools are currently configured
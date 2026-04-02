# Perceptra Python SDK

Official Python SDK for the [Perceptra Hub](https://github.com/tannousgeagea/perceptra-hub) Computer Vision MLOps platform.

## Installation

```bash
pip install perceptra
```

## Quick Start

```python
import perceptra

# Initialize with your API key
client = perceptra.Perceptra(api_key="ph_live_abc123...")

# Or set PERCEPTRA_API_KEY environment variable
client = perceptra.Perceptra()

# Create a project
project = client.projects.create(
    name="Hard Hat Detection",
    project_type_id=1,
)

# Upload images
from pathlib import Path
for path in Path("./images").glob("*.jpg"):
    client.images.upload(file=path, project_id=project.project_id)

# Add annotations
client.annotations.create(
    project_id=project.project_id,
    image_id="...",
    annotation_type="bbox",
    annotation_class_id=0,
    data=[0.1, 0.2, 0.5, 0.8],
)

# Split dataset
client.projects.split_dataset(project.project_id, 0.7, 0.2, 0.1)

# Create a dataset version
version = client.versions.create(project.project_id, "v1.0")

# Create and train a model
model = client.models.create(
    project_id=project.project_id,
    name="YOLOv8 Hard Hat",
    task="object-detection",
    framework="yolo",
)

result = client.models.train(
    model_id=model.id,
    dataset_version_id=version.id,
    config={"epochs": 100, "batch_size": 16},
)

# Stream training logs
for line in client.training.stream_logs(result.training_session_id):
    print(line, end="")
```

## Async Support

```python
import asyncio
import perceptra

async def main():
    async with perceptra.AsyncPerceptra(api_key="ph_live_abc123...") as client:
        projects = await client.projects.list()
        for p in projects:
            print(p["name"])

asyncio.run(main())
```

## Error Handling

```python
from perceptra import NotFoundError, RateLimitError, AuthenticationError

try:
    project = client.projects.retrieve("nonexistent-id")
except NotFoundError:
    print("Project not found")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except AuthenticationError:
    print("Invalid API key")
```

## Available Resources

| Resource | Description |
|----------|-------------|
| `client.projects` | Project CRUD, image management, dataset splitting |
| `client.images` | Image upload, listing, bulk operations |
| `client.annotations` | Annotation CRUD, batch operations |
| `client.models` | ML model CRUD, training triggers |
| `client.training` | Training session monitoring, log streaming |
| `client.versions` | Dataset version management, export |
| `client.organizations` | Organization details, members |
| `client.jobs` | Annotation job management |
| `client.classes` | Annotation class management |
| `client.tags` | Image tag management |
| `client.api_keys` | API key management, rotation |
| `client.storage` | Storage profile management |

## Configuration

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| `api_key` | — | `PERCEPTRA_API_KEY` |
| `base_url` | `http://localhost:29082` | `PERCEPTRA_BASE_URL` |
| `timeout` | `30.0` | — |
| `max_retries` | `3` | — |

## License

MIT

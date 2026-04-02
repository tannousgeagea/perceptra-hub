# Plan: Perceptra Python SDK

## Context

Perceptra Hub needs a standalone Python SDK package that lets users manage their CV MLOps platform programmatically using API key authentication. The SDK must be pip-installable, fully typed, support sync + async, and provide a resource-oriented API covering all 32 router groups (189 endpoints). API key auth (built in the previous task) makes organization context implicit — SDK users never need JWT or `X-Organization-ID` headers.

---

## Package Layout

```
perceptra-sdk/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── perceptra/
│       ├── __init__.py              # Public API: Perceptra, AsyncPerceptra, exceptions
│       ├── _version.py              # __version__
│       ├── _client.py               # Perceptra (sync client)
│       ├── _async_client.py         # AsyncPerceptra (async client)
│       ├── _base_client.py          # Shared config, resource wiring
│       ├── _constants.py            # DEFAULT_BASE_URL, API_VERSION, USER_AGENT
│       ├── _exceptions.py           # Exception hierarchy
│       ├── _files.py                # File upload normalization (path/bytes/fileobj)
│       ├── _pagination.py           # SyncPage, AsyncPage, auto-pagination iterators
│       ├── _utils.py                # strip_none, flatten_query_params
│       ├── py.typed                 # PEP 561 marker
│       │
│       ├── _transport/
│       │   ├── __init__.py
│       │   ├── _base.py             # BaseTransport ABC
│       │   ├── _sync.py             # SyncTransport (httpx.Client)
│       │   ├── _async.py            # AsyncTransport (httpx.AsyncClient)
│       │   ├── _retry.py            # RetryPolicy, exponential backoff
│       │   ├── _auth.py             # APIKeyAuth (httpx.Auth subclass)
│       │   └── _sse.py              # SSE decoder for training log streaming
│       │
│       ├── resources/               # One file per resource group
│       │   ├── __init__.py
│       │   ├── _base.py             # SyncAPIResource, AsyncAPIResource
│       │   ├── projects.py          # Projects, AsyncProjects
│       │   ├── images.py            # Images, AsyncImages
│       │   ├── annotations.py       # Annotations, AsyncAnnotations
│       │   ├── models.py            # Models, AsyncModels
│       │   ├── training.py          # Training, AsyncTraining
│       │   ├── versions.py          # Versions, AsyncVersions
│       │   ├── organizations.py     # Organizations, AsyncOrganizations
│       │   ├── jobs.py              # Jobs, AsyncJobs
│       │   ├── classes.py           # Classes, AsyncClasses
│       │   ├── tags.py              # Tags, AsyncTags
│       │   ├── api_keys.py          # APIKeys, AsyncAPIKeys
│       │   ├── storage.py           # Storage, AsyncStorage
│       │   ├── compute.py           # Compute, AsyncCompute
│       │   ├── evaluation.py        # Evaluation, AsyncEvaluation
│       │   └── analytics.py         # Analytics, AsyncAnalytics
│       │
│       └── types/                   # Pydantic response/request models
│           ├── __init__.py
│           ├── project.py
│           ├── image.py
│           ├── annotation.py
│           ├── model.py
│           ├── training.py
│           ├── version.py
│           ├── organization.py
│           ├── job.py
│           ├── api_key.py
│           ├── storage.py
│           ├── compute.py
│           ├── evaluation.py
│           └── shared.py            # Pagination, enums, common types
```

---

## Client Initialization

```python
import perceptra

# Sync (most common)
client = perceptra.Perceptra(
    api_key="ph_live_abc123...",            # or PERCEPTRA_API_KEY env var
    base_url="https://hub.example.com",     # default: http://localhost:29082
    timeout=30.0,                           # seconds
    max_retries=3,                          # retries on 429/5xx
)

# Async
async with perceptra.AsyncPerceptra(api_key="ph_live_...") as client:
    projects = await client.projects.list()
```

Design:
- `api_key` read from arg or `PERCEPTRA_API_KEY` env var; validated on init (`ph_` prefix)
- `__repr__` masks key: `Perceptra(api_key="ph_live_ab...****")`
- Client is a context manager that closes httpx connection pool
- All resources instantiated as attributes: `self.projects = Projects(self)`

---

## Transport Layer (`_transport/`)

### Auth (`_auth.py`)
`httpx.Auth` subclass injecting `X-API-Key` header on every request. No org headers needed — the server resolves organization from the key.

### Retry (`_retry.py`)
- Retries on 429, 500, 502, 503, 504
- Exponential backoff: `0.5 * 2^attempt` seconds, jittered, capped at 8s
- Honors `Retry-After` header from 429 responses
- POST retries only on 429/5xx (not 4xx)

### Core `_request()` method (on client)
```python
def _request(self, method, path, *, body=None, query=None, files=None, cast_to, stream=False) -> T:
```
1. Builds URL: `{base_url}/api/v1{path}`
2. Strips None from query params
3. Sends JSON body or multipart (if files)
4. Maps error status codes to typed exceptions
5. Deserializes response JSON into `cast_to` Pydantic model

### SSE (`_sse.py`)
For `client.training.stream_logs()` — opens streaming response, yields lines as `Iterator[str]` / `AsyncIterator[str]`.

---

## Error Hierarchy (`_exceptions.py`)

```
PerceptraError (base)
├── APIError (all HTTP errors)
│   ├── AuthenticationError (401)
│   ├── PermissionDeniedError (403)
│   ├── NotFoundError (404)
│   ├── ConflictError (409)
│   ├── UnprocessableEntityError (422)
│   ├── RateLimitError (429)          # .retry_after: float | None
│   └── InternalServerError (5xx)
├── APIConnectionError
└── APITimeoutError
```

`APIError` carries `status_code`, `message` (from `response["detail"]`), `body`, `headers`.

---

## Resource Methods (Core)

### `client.projects`
| Method | HTTP | Path |
|--------|------|------|
| `.create(name, project_type_id, ...)` | POST | `/projects/add` |
| `.list(*, skip, limit, is_active)` | GET | `/projects` |
| `.retrieve(project_id)` | GET | `/projects/{id}` |
| `.delete(project_id)` | DELETE | `/projects/{id}` |
| `.list_images(project_id, *, skip, limit, status, search, split)` | GET | `/projects/{id}/images` |
| `.add_images(project_id, image_ids, *, auto_assign_job)` | POST | `/projects/{id}/images` |
| `.split_dataset(project_id, train_ratio, val_ratio, test_ratio)` | POST | `/projects/{id}/images/split` |

### `client.images`
| Method | HTTP | Path |
|--------|------|------|
| `.upload(file, *, name, project_id, tags, storage_profile_id)` | POST | `/images/upload` |
| `.list(*, skip, limit)` | GET | `/images` |
| `.bulk_delete(image_ids)` | POST | `/images/bulk-delete` |

`file` accepts `str | Path | bytes | BinaryIO`. The `_files.py` helper normalizes to `httpx` multipart format, infers filename and content-type.

### `client.annotations`
| Method | HTTP | Path |
|--------|------|------|
| `.create(project_id, image_id, ...)` | POST | `/projects/{pid}/images/{iid}/annotations` |
| `.batch_create(project_id, items)` | POST | `/projects/{pid}/annotations/batch` |
| `.list(project_id, image_id)` | GET | `/projects/{pid}/images/{iid}/annotations` |
| `.update(project_id, image_id, annotation_id, ...)` | PATCH | `.../{aid}` |
| `.delete(project_id, image_id, annotation_id)` | DELETE | `.../{aid}` |

### `client.models`
| Method | HTTP | Path |
|--------|------|------|
| `.create(project_id, name, task, framework, ...)` | POST | `/models/projects/{pid}/models` |
| `.list(project_id)` | GET | `/models/projects/{pid}/models` |
| `.retrieve(model_id)` | GET | `/models/{id}` |
| `.update(model_id, ...)` | PUT | `/models/{id}` |
| `.delete(model_id)` | DELETE | `/models/{id}` |
| `.train(model_id, dataset_version_id, ...)` | POST | `/models/{id}/train` |
| `.duplicate(model_id, ...)` | POST | `/models/{id}/duplicate` |

### `client.training`
| Method | HTTP | Path |
|--------|------|------|
| `.list(*, project_id, model_id)` | GET | `/training-sessions` |
| `.retrieve(session_id)` | GET | `/training-sessions/{id}` |
| `.stream_logs(session_id)` | GET (SSE) | `/training-sessions/{id}/logs` |

### `client.versions`
| Method | HTTP | Path |
|--------|------|------|
| `.create(project_id, version_name, ...)` | POST | `/projects/{pid}/versions` |
| `.list(project_id)` | GET | `/projects/{pid}/versions` |
| `.retrieve(project_id, version_id)` | GET | `/projects/{pid}/versions/{vid}` |
| `.export(project_id, version_id, ...)` | POST | `/projects/{pid}/versions/{vid}/export` |

### `client.organizations`
| Method | HTTP | Path |
|--------|------|------|
| `.retrieve()` | GET | `/organizations/details` |
| `.list_members()` | GET | `/organizations/members` |
| `.list_projects()` | GET | `/organizations/projects` |

### `client.jobs`, `client.classes`, `client.tags`, `client.api_keys`, `client.storage`, `client.compute`, `client.evaluation`, `client.analytics`
Follow the same pattern — CRUD methods mapping to their respective API endpoints.

---

## Type Definitions (`types/`)

All response types use Pydantic `BaseModel` with `ConfigDict(extra="allow")` for forward compatibility. Request params use keyword arguments on resource methods (not separate request models).

Example (`types/project.py`):
```python
class Project(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    project_type: dict
    visibility: dict
    is_active: bool
    created_at: str
    last_edited: str
```

Source schemas to mirror:
- `perceptra_hub/api/routers/projects/schemas.py`
- `perceptra_hub/api/routers/ml_models/schemas.py`
- `perceptra_hub/api/routers/annotations/schemas.py`
- `perceptra_hub/api/routers/api_keys/schemas.py`
- `perceptra_hub/api/routers/storage/schemas.py`

---

## Pagination (`_pagination.py`)

```python
# Manual
page = client.projects.list(skip=0, limit=50)
page.data    # list[ProjectListItem]
page.total   # int

# Auto-pagination iterator
for project in client.projects.list(auto_paginate=True):
    print(project.name)
```

`SyncPage[T]` wraps response with `.data`, `.total`, `.has_more`. Auto-pagination yields items across pages (incrementing `skip` by `limit` until exhausted).

---

## File Uploads (`_files.py`)

```python
# All accepted
client.images.upload(file="./photo.jpg")           # str/Path
client.images.upload(file=raw_bytes, name="x.png")  # bytes
client.images.upload(file=open("x.jpg", "rb"))       # file-like
```

Normalizes to httpx multipart: `files={"file": (filename, data, content_type)}`, `data={...metadata...}`.

---

## `pyproject.toml`

```toml
[project]
name = "perceptra"
version = "0.1.0"
description = "Official Python SDK for Perceptra Hub"
requires-python = ">=3.9"
dependencies = [
    "httpx>=0.25.0,<1.0",
    "pydantic>=2.0.0,<3.0",
    "anyio>=3.5.0,<5.0",
    "typing-extensions>=4.7.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "respx", "ruff", "mypy"]
```

Build backend: `hatchling`. Package in `src/perceptra/`.

---

## Implementation Phases

| Phase | What | Files |
|-------|------|-------|
| **1. Foundation** | Exceptions, transport (auth, sync, async, retry), base client, `__init__.py` | `_exceptions.py`, `_transport/*`, `_base_client.py`, `_client.py`, `_async_client.py`, `_constants.py` |
| **2. Core resources** | Projects, Images, Annotations + their types | `resources/{projects,images,annotations}.py`, `types/{project,image,annotation}.py`, `_files.py` |
| **3. ML resources** | Models, Training, Versions + SSE streaming | `resources/{models,training,versions}.py`, `types/{model,training,version}.py`, `_transport/_sse.py` |
| **4. Supporting resources** | Organizations, Jobs, Classes, Tags, API Keys, Storage, Compute, Eval, Analytics | Remaining `resources/*.py` and `types/*.py` |
| **5. Pagination** | Page types, auto-pagination iterators | `_pagination.py` |
| **6. Packaging** | pyproject.toml, README, py.typed, LICENSE | Root files |

---

## Usage Example (End-to-End)

```python
import perceptra

client = perceptra.Perceptra(api_key="ph_live_abc123...")

# Create project
project = client.projects.create(
    name="Hard Hat Detection",
    project_type_id=1,
)

# Upload images
for path in Path("./images").glob("*.jpg"):
    client.images.upload(file=path, project_id=project.project_id)

# Annotate
client.annotations.create(
    project_id=project.project_id,
    image_id="...",
    annotation_type="bbox",
    annotation_class_id=0,
    data=[0.1, 0.2, 0.5, 0.8],
)

# Split + version
client.projects.split_dataset(project.project_id, 0.7, 0.2, 0.1)
version = client.versions.create(project.project_id, "v1.0")

# Train
model = client.models.create(project.project_id, "YOLOv8", "object-detection", "yolo")
result = client.models.train(model.id, version.id, config={"epochs": 100})

# Stream logs
for line in client.training.stream_logs(result.training_session_id):
    print(line, end="")
```

---

## Verification

1. **Unit tests**: Mock HTTP with `respx`, test each resource method, error mapping, retry logic, pagination
2. **Type checking**: `mypy --strict` on the package
3. **Integration test**: Against a running Perceptra Hub instance — create project, upload image, annotate, train (can be a manual smoke test initially)
4. **Install test**: `pip install -e .` from the SDK root, import and instantiate client
5. **Lint**: `ruff check src/`
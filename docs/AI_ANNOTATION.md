# Plan: Production-Ready AI Annotation Inference Architecture

## Context

The SAM-based annotation suggestion system is fully scaffolded (endpoints, services, frontend, Celery, Redis caching) but **never actually runs real inference**. The `perceptra-seg` package is already cloned at `/home/tannous/Desktop/malumetrix/perceptra-seg/` and has a production-ready `service/` FastAPI layer. The issues are:

1. `dependencies.py` always returns `MockSegmentationService()` — intentionally used for testing without GPU
2. `services.py` inference methods (`run_sam_point`, `run_sam_auto`) are empty stubs
3. `SegmentationService` loads GPU models inside the API worker process — blocks async workers, requires local GPU
4. `perceptra-seg` service is missing 3 endpoints the platform needs: `/v1/segment/text`, `/v1/segment/exemplar`, `/v1/segment/auto`
5. No docker-compose wiring between platform and inference service
6. No async HTTP client in the platform to call the inference service

**Invariants to preserve:**
- All business logic (sessions, Redis caching, accept/reject, annotation creation) stays 100% in `perceptra-hub`
- `perceptra-seg` service is stateless — pure inference, no DB, no session state
- MockSegmentationService remains available as a zero-config dev fallback when `SEG_INFERENCE_URL` is unset

---

## Architecture

```
Browser
  └──► FastAPI (perceptra-hub, runs anywhere — no GPU needed)
           └──► SuggestionService     ← sessions, cache, accept/reject, annotations (unchanged)
           └──► SegInferenceClient    ← NEW: async httpx, coord conversion, error handling
                       │
                  async HTTP (httpx)
                       │
           ┌───────────▼────────────────────────┐
           │  perceptra-seg/service/  (existing) │  ← Running on GPU server or same host
           │                                     │
           │  GET  /v1/healthz        (exists)   │
           │  POST /v1/segment/points (exists)   │
           │  POST /v1/segment/box    (exists)   │
           │  POST /v1/segment/text   (ADD)      │
           │  POST /v1/segment/exemplar (ADD)    │
           │  POST /v1/segment/auto   (ADD)      │
           └─────────────────────────────────────┘
```

**Key coordinate note:** The perceptra-seg API speaks **pixel coordinates** (absolute ints). The platform uses **normalized coordinates** (0.0–1.0). The `SegInferenceClient` owns this conversion — no other layer changes.

---

## Coordinate & Image Contract

| Boundary | Format |
|---|---|
| Frontend → Platform API | Normalized 0–1 |
| Platform `SegmentationService` / `SegInferenceClient` interface | Normalized 0–1 |
| `SegInferenceClient` → perceptra-seg HTTP | Pixel coordinates (absolute int) |
| Image transport | numpy → PIL PNG → base64 string (already the perceptra-seg contract) |

---

## Part 1 — Changes to `perceptra-seg/`

### 1a. Add 3 Missing Endpoints to `service/routes.py`

The existing `routes.py` has `/v1/segment/box`, `/v1/segment/points`, `/v1/segment`. Add:

**`POST /v1/segment/text`** (SAM3 only)
```python
class SegmentTextRequest(BaseModel):
    image: str                          # base64 PNG/JPEG
    text: str                           # natural language prompt
    output_formats: list[str] = ["rle", "polygons"]

# Returns List[SegmentationResponse]
# Calls: segmentor.segment_from_text(image_np, text, output_formats=...)
```

**`POST /v1/segment/exemplar`** (SAM3 only)
```python
class SegmentExemplarRequest(BaseModel):
    image: str
    exemplar_box: list[int]             # [x1, y1, x2, y2] pixel coords
    output_formats: list[str] = ["rle", "polygons"]

# Returns List[SegmentationResponse]
# Calls: segmentor.segment_from_exemplar_box(image_np, pixel_box, output_formats=...)
```

**`POST /v1/segment/auto`**
```python
class SegmentAutoRequest(BaseModel):
    image: str
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_area: float = 0.001             # normalized — filter tiny masks
    output_formats: list[str] = ["rle", "polygons"]

# Returns List[SegmentationResponse]
# Calls: segmentor.segment_batch(image_np, ...) or backend's generate_all()
# min_area converted to pixels for filtering: area < min_area * H * W → skip
```

### 1b. Enhance `GET /v1/healthz`

Current response: `{"status": "ok"}`

Enhance to return model info (useful for the platform's health routing):
```python
{
  "status": "ok",
  "model_name": "sam_v2",
  "device": "cuda",
  "precision": "fp16",
  "gpu_memory_used_mb": 3812,   # None if CPU
  "model_loaded": true
}
```

This lets the platform validate the configured model matches what's running.

### 1c. Add `docker-compose.yml` to `perceptra-seg/`

```yaml
# perceptra-seg/docker-compose.yml
services:
  perceptra-seg:
    build:
      context: .
      dockerfile: Dockerfile.gpu     # or Dockerfile for CPU
    environment:
      - SEGMENTOR_MODEL_NAME=${SEG_MODEL:-sam_v2}
      - SEGMENTOR_RUNTIME_DEVICE=${SEG_DEVICE:-cuda}
      - SEGMENTOR_RUNTIME_PRECISION=${SEG_PRECISION:-fp16}
      - SEGMENTOR_SERVER_API_KEYS=${SEG_API_KEYS:-}
    ports:
      - "29086:8080"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/v1/healthz', timeout=5)"]
      interval: 30s
      timeout: 10s
      start_period: 90s
      retries: 3
```

**Critical files to modify in `perceptra-seg/`:**

| File | Change |
|---|---|
| `service/routes.py` | Add 3 endpoints + their request schemas + enhanced /healthz |
| `perceptra-seg/docker-compose.yml` | Create |

---

## Part 2 — Changes to `perceptra-hub/`

### 2a. Create `inference_client.py`

**File:** `perceptra-hub/perceptra_hub/api/routers/suggestions/inference_client.py`

This is the only place that knows about HTTP transport and coordinate conversion. Implements the same sync interface as `MockSegmentationService` but async.

```python
import os, base64, io
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image as PILImage
import httpx
from fastapi import HTTPException

from .segmentation_service import SegmentationOutput

class SegInferenceClient:
    """
    Async HTTP client for perceptra-seg inference service.
    Converts normalized platform coords ↔ pixel coords for the service API.
    Drop-in interface replacement for SegmentationService / MockSegmentationService.
    """

    def __init__(self, base_url: str, timeout_s: float = 10.0, api_key: str = ""):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            headers=headers,
        )

    # ---- Image encoding ----

    def _encode_image(self, image: np.ndarray) -> str:
        """numpy RGB → base64 PNG string (what perceptra-seg expects)"""
        buf = io.BytesIO()
        PILImage.fromarray(image.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ---- Coordinate helpers ----

    def _norm_to_pixel_box(self, box, w, h):
        """Normalized (x,y,bw,bh) → pixel [x1,y1,x2,y2]"""
        x, y, bw, bh = box
        return [int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)]

    def _norm_to_pixel_points(self, points, w, h):
        """[(x_norm, y_norm, label), ...] → [{"x": px, "y": py, "label": l}, ...]"""
        return [{"x": int(px * w), "y": int(py * h), "label": lbl} for px, py, lbl in points]

    def _pixel_bbox_to_norm(self, bbox, w, h) -> Tuple:
        """pixel [x1,y1,x2,y2] → normalized (x,y,bw,bh)"""
        x1, y1, x2, y2 = bbox
        return (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)

    def _response_to_output(self, resp: dict, w: int, h: int) -> SegmentationOutput:
        bbox_norm = self._pixel_bbox_to_norm(resp["bbox"], w, h) if resp.get("bbox") else (0, 0, 0, 0)
        return SegmentationOutput(
            bbox=bbox_norm,
            mask_rle=resp.get("rle"),
            polygons=resp.get("polygons"),
            confidence=resp.get("score", 0.0),
        )

    def _handle_error(self, e: Exception):
        if isinstance(e, httpx.TimeoutException):
            raise HTTPException(504, "Inference server timed out")
        if isinstance(e, httpx.ConnectError):
            raise HTTPException(503, "Inference server unreachable — check SEG_INFERENCE_URL")
        raise HTTPException(502, f"Inference service error: {e}")

    # ---- Public interface (mirrors MockSegmentationService) ----

    async def segment_from_points(self, image: np.ndarray, points, model, device, precision) -> SegmentationOutput:
        h, w = image.shape[:2]
        try:
            r = await self._client.post("/v1/segment/points", json={
                "image": self._encode_image(image),
                "points": self._norm_to_pixel_points(points, w, h),
                "output_formats": ["rle", "polygons"],
            })
            r.raise_for_status()
        except Exception as e:
            self._handle_error(e)
        return self._response_to_output(r.json(), w, h)

    async def segment_from_box(self, image, box, model, device, precision) -> SegmentationOutput:
        h, w = image.shape[:2]
        try:
            r = await self._client.post("/v1/segment/box", json={
                "image": self._encode_image(image),
                "box": self._norm_to_pixel_box(box, w, h),
                "output_formats": ["rle", "polygons"],
            })
            r.raise_for_status()
        except Exception as e:
            self._handle_error(e)
        return self._response_to_output(r.json(), w, h)

    async def segment_from_text(self, image, text, model, device, precision) -> List[SegmentationOutput]:
        h, w = image.shape[:2]
        try:
            r = await self._client.post("/v1/segment/text", json={
                "image": self._encode_image(image),
                "text": text,
                "output_formats": ["rle", "polygons"],
            })
            r.raise_for_status()
        except Exception as e:
            self._handle_error(e)
        return [self._response_to_output(item, w, h) for item in r.json()]

    async def segment_from_exemplar(self, image, exemplar_box, model, device, precision) -> List[SegmentationOutput]:
        h, w = image.shape[:2]
        try:
            r = await self._client.post("/v1/segment/exemplar", json={
                "image": self._encode_image(image),
                "exemplar_box": self._norm_to_pixel_box(exemplar_box, w, h),
                "output_formats": ["rle", "polygons"],
            })
            r.raise_for_status()
        except Exception as e:
            self._handle_error(e)
        return [self._response_to_output(item, w, h) for item in r.json()]

    async def segment_auto(self, image, config: dict, model, device, precision) -> List[SegmentationOutput]:
        """Used by Celery task only — called via sync wrapper."""
        h, w = image.shape[:2]
        try:
            r = await self._client.post("/v1/segment/auto", json={
                "image": self._encode_image(image),
                **config,
                "output_formats": ["rle", "polygons"],
            }, timeout=120.0)
            r.raise_for_status()
        except Exception as e:
            self._handle_error(e)
        return [self._response_to_output(item, w, h) for item in r.json()]

    async def health_check(self) -> dict:
        r = await self._client.get("/v1/healthz")
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._client.aclose()


class SegInferenceClientSync:
    """Synchronous variant for use inside Celery tasks."""
    def __init__(self, base_url: str, timeout_s: float = 120.0, api_key: str = ""):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s, headers=headers)

    # Same _encode_image, _norm_to_pixel_*, _pixel_bbox_to_norm, _response_to_output helpers
    # Same segment_auto() but synchronous (no await)
```

### 2b. Fix `dependencies.py`

**File:** `perceptra-hub/perceptra_hub/api/routers/suggestions/dependencies.py`

```python
import os
from .inference_client import SegInferenceClient
from .segmentation_service_mock import MockSegmentationService

def get_segmentation_service():
    url = os.getenv("SEG_INFERENCE_URL", "").strip()
    if not url:
        return MockSegmentationService()
    api_key = os.getenv("SEG_INFERENCE_API_KEY", "")
    timeout = float(os.getenv("SEG_INFERENCE_TIMEOUT", "10"))
    return SegInferenceClient(base_url=url, timeout_s=timeout, api_key=api_key)
```

### 2c. Make `MockSegmentationService` Methods `async`

**File:** `perceptra-hub/perceptra_hub/api/routers/suggestions/segmentation_service_mock.py`

All public methods change from `def` to `async def`:
- `async def segment_from_points(...)` → `return self._create_output()`
- `async def segment_from_box(...)` → same
- `async def segment_batch_boxes(...)` → same
- `async def segment_from_text(...)` → same
- `async def segment_from_exemplar(...)` → same
- `async def segment_text_and_box(...)` → same

No logic changes — just `async` prefix enables `await` in endpoints.

### 2d. Fix 4 Interactive Endpoint Files — Add `await`

**Files:** `queries/suggestion_from_point.py:44`, `queries/suggestion_from_box.py`, `queries/suggestion_from_text.py`, `queries/suggestion_from_examplar.py`

Single change per file — add `await`:

```python
# Before (suggestion_from_point.py line 44):
output = seg_svc.segment_from_points(image, points, ...)

# After:
output = await seg_svc.segment_from_points(image, points, ...)
```

Same pattern for box, text, exemplar. No other changes in these files.

### 2e. Create Celery Task for Auto-Segment

**New file:** `perceptra-hub/perceptra_hub/api/tasks/seg_inference.py`

```python
import os, uuid as uuid_lib, json
from celery import shared_task
from django.core.cache import cache

@shared_task(bind=True, name='seg_inference:auto_segment', max_retries=3, queue='seg_inference',
             autoretry_for=(Exception,), retry_backoff=True)
def auto_segment_task(self, session_id: str, image_id: int, model: str, device: str,
                       precision: str, config: dict):
    from api.routers.suggestions.inference_client import SegInferenceClientSync
    from api.routers.suggestions.services import SuggestionService
    from api.routers.suggestions.schemas import Suggestion, BoundingBox

    # 1. Load image (sync Django ORM)
    from projects.models import ProjectImage
    from PIL import Image as PILImage
    import numpy as np
    project_image = ProjectImage.objects.select_related('image__storage_profile').get(id=image_id)
    # ... load image to np.ndarray (reuse same logic as services.py:load_image)

    # 2. Call inference service
    client = SegInferenceClientSync(
        base_url=os.environ["SEG_INFERENCE_URL"],
        api_key=os.getenv("SEG_INFERENCE_API_KEY", ""),
        timeout_s=120.0,
    )
    outputs = client.segment_auto(image_np, config, model, device, precision)

    # 3. Filter and convert to Suggestion objects
    min_area = config.get("min_area", 0.001)
    suggestions = [
        Suggestion(
            suggestion_id=str(uuid_lib.uuid4()),
            bbox=BoundingBox(x=o.bbox[0], y=o.bbox[1], width=o.bbox[2], height=o.bbox[3]),
            mask_rle=o.mask_rle, polygons=o.polygons, confidence=o.confidence,
            type="auto", status="pending",
        )
        for o in outputs
        if (o.bbox[2] * o.bbox[3]) >= min_area
    ]

    # 4. Store in Redis (same key pattern as SuggestionService)
    cache_key = f"suggestions:{session_id}"
    cache.set(cache_key, json.dumps([s.model_dump() for s in suggestions]), 3600)

    # 5. Update session count in DB
    from annotations.models import SuggestionSession
    SuggestionSession.objects.filter(suggestion_id=session_id).update(
        suggestions_generated=len(suggestions)
    )
```

### 2f. Fix `suggestion_auto_segment.py` — Dispatch to Celery

**File:** `perceptra-hub/perceptra_hub/api/routers/suggestions/queries/suggestion_auto_segment.py`

Replace `background_tasks.add_task(svc.run_sam_auto, ...)` with Celery dispatch:

```python
from api.tasks.seg_inference import auto_segment_task

# In the endpoint, after creating the session:
session = await sug_svc.get_session(request.session_id)
auto_segment_task.delay(
    session_id=str(request.session_id),
    image_id=image_id,
    model=session.model_name,
    device=session.model_device,
    precision=session.model_precision,
    config={"points_per_side": request.points_per_side, "min_area": request.min_area,
            "pred_iou_thresh": request.pred_iou_thresh}
)
```

### 2g. Update `celery_config.py` — Add Queue

**File:** `perceptra-hub/perceptra_hub/api/config/celery_config.py`

The existing `route_task()` function uses `queue:task_name` prefix routing — `seg_inference:auto_segment` will automatically route to `seg_inference` queue. No code change needed, just add the queue to the worker list.

### 2h. Add Worker to `supervisord.conf`

**File:** `perceptra-hub/supervisord.conf`

```ini
[program:seg_inference_worker]
command=celery -A api worker -l info -Q seg_inference -c 2 --without-gossip --without-mingle
directory=/perceptra_hub
stdout_logfile=/var/log/celery_seg_inference.log
stderr_logfile=/var/log/celery_seg_inference_err.log
autostart=true
autorestart=true
```

### 2i. Update `docker-compose.yml` and `.env`

**Root `docker-compose.yml`** — add perceptra-seg service:
```yaml
perceptra-seg:
  build:
    context: ./perceptra-seg
    dockerfile: Dockerfile.gpu        # Dockerfile for CPU-only servers
  environment:
    - SEGMENTOR_MODEL_NAME=${SEG_MODEL:-sam_v2}
    - SEGMENTOR_RUNTIME_DEVICE=${SEG_DEVICE:-cuda}
    - SEGMENTOR_RUNTIME_PRECISION=${SEG_PRECISION:-fp16}
    - SEGMENTOR_SERVER_API_KEYS=${SEG_API_KEYS:-}
  ports:
    - "29086:8080"
  networks:
    - perceptra-net
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**`perceptra-hub/.env`** — add:
```
SEG_INFERENCE_URL=http://perceptra-seg:8080
SEG_INFERENCE_TIMEOUT=10
SEG_INFERENCE_API_KEY=
SEG_MODEL=sam_v2
SEG_DEVICE=cuda
SEG_PRECISION=fp16
```

### 2j. Remove Dead Stubs from `services.py`

**File:** `perceptra-hub/perceptra_hub/api/routers/suggestions/services.py`

Remove `run_sam_point` and `run_sam_auto` — these stubs are vestigial; inference now happens in endpoint layer (interactive) and Celery task (auto). The remaining methods in `SuggestionService` are correct and already implemented.

---

## Complete File Change Summary

### `perceptra-seg/` changes:
| File | Action | What |
|---|---|---|
| `service/routes.py` | Modify | Add `/v1/segment/text`, `/v1/segment/exemplar`, `/v1/segment/auto`; enhance `/v1/healthz` |
| `docker-compose.yml` | Create | Service definition with GPU support |

### `perceptra-hub/` changes:
| File | Action | What |
|---|---|---|
| `api/routers/suggestions/inference_client.py` | Create | Async + sync HTTP clients with coord conversion |
| `api/routers/suggestions/dependencies.py` | Modify | Env-based routing: real client vs mock |
| `api/routers/suggestions/segmentation_service_mock.py` | Modify | Add `async` to all public methods |
| `api/routers/suggestions/queries/suggestion_from_point.py` | Modify | `await seg_svc.segment_from_points(...)` |
| `api/routers/suggestions/queries/suggestion_from_box.py` | Modify | `await seg_svc.segment_from_box(...)` |
| `api/routers/suggestions/queries/suggestion_from_text.py` | Modify | `await seg_svc.segment_from_text(...)` |
| `api/routers/suggestions/queries/suggestion_from_examplar.py` | Modify | `await seg_svc.segment_from_exemplar(...)` |
| `api/routers/suggestions/queries/suggestion_auto_segment.py` | Modify | Celery task dispatch instead of BackgroundTasks |
| `api/routers/suggestions/services.py` | Modify | Remove `run_sam_point` and `run_sam_auto` stubs |
| `api/tasks/seg_inference.py` | Create | Celery auto-segment task |
| `supervisord.conf` | Modify | Add `seg_inference_worker` program |
| `docker-compose.yml` | Modify | Add `perceptra-seg` service |
| `.env` | Modify | Add `SEG_INFERENCE_URL`, `SEG_INFERENCE_TIMEOUT`, etc. |

---

## Verification Plan

1. **Dev (no GPU):** Keep `SEG_INFERENCE_URL` unset → `MockSegmentationService` returned → all interactive endpoints return synthetic suggestions. Confirm no regression.

2. **With inference server:** Start `docker-compose up perceptra-seg` → hit `GET http://localhost:29086/v1/healthz` → expect `{"status":"ok","model_loaded":true}`. Open annotation tool → click point prompt → real SAM mask returned.

3. **Auto-segment:** `POST /suggestions/sam/auto` → 202 + session_id. Watch `seg_inference_worker` Celery logs. Poll `GET /suggestions/{session_id}` → status `ready`, suggestions populated.

4. **Failure path:** Stop perceptra-seg container → click point tool → frontend receives 503 with message "Inference server unreachable" instead of silent failure or generic 500.

5. **CPU deployment:** Set `SEG_DEVICE=cpu` → confirm inference completes (slower) without crash.
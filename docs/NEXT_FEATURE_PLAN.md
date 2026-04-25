# Perceptra: Next Important Features

## Context

The prior plan (CV DevOps loop) is fully implemented: `perceptra-inference` service, deploy/undeploy API, auto-annotation pipeline, retraining policies, champion/challenger evaluation, and all frontend wiring for ModelDetail tabs and ModelsList badges.

**Platform assessment (April 2026):**
- Backend ~85–90% complete; 32 auto-discovered routers, all core domains production-quality
- Frontend ~65% wired: Auto-Annotate, Evaluation Dashboard, Inference page are still 100% mock data
- Real-time: no WebSocket/SSE anywhere — training jobs are a black-box UX
- Active learning: heuristic-only (entropy/confidence), no embedding-based diversity
- Explainability: not started
- Data drift detection: not started
- Tests: ~66 LOC total

**The next step:** Surface what's already built (wire the mocks), then add the three features that uniquely differentiate Perceptra from Roboflow / Scale AI / LandingLens.

---

## Priority 1 — Fix Frontend Mocks (1 week)

Unblocks everything else. The backends exist; only the frontend calls are wrong.

### 1a. Auto-Annotate

**File to rewrite:** `perceptra-console/src/hooks/useAutoAnnotate.ts`

- Replace `MOCK_MODELS` array with `useQuery` → `GET /api/v1/models/projects/{projectId}/models`, filter to `has_production_version: true`
- Replace `setTimeout` tick loop in `startProcessing` with `apiFetch` → `POST /api/v1/projects/{projectId}/auto-annotate` (body: `{ model_version_id, image_ids, confidence_threshold }`) which returns `{ task_id, status }`
- Wire returned `task_id` to existing `useProgress` polling hook (`GET /api/v1/progress/{task_id}`)
- Drop the pause/resume simulation (Celery tasks don't support it); hide those buttons or replace with Cancel

### 1b. Evaluation Dashboard

**File to rewrite:** `perceptra-console/src/hooks/useEvaluationData.ts`

Replace all seven `generateMock*()` functions with real `apiFetch` calls:

| Mock function | Real endpoint |
|---|---|
| `generateMockSummary` | `GET /api/v1/evaluation/projects/{id}/summary` (EvaluationQueryBuilder exists) |
| `generateMockClassMetrics` | `GET /api/v1/evaluation/projects/{id}/class-metrics` |
| `generateMockTrends` | `GET /api/v1/temporal/projects/{id}/trends?days={days}` |
| `generateMockAlerts` | `GET /api/v1/temporal/projects/{id}/alerts` (need thin endpoint over `MetricAlert`) |
| `generateMockThresholds` | `GET /api/v1/temporal/projects/{id}/thresholds` |
| `generatePriorityQueue` | `GET /api/v1/active-learning/projects/{id}/suggest?strategy=uncertainty&limit=20` |

**New backend file needed:** `perceptra-hub/perceptra_hub/api/routers/temporal/queries/alerts.py`
- `GET /temporal/projects/{id}/alerts` — list `MetricAlert` records, filter by `is_acknowledged`
- `POST /temporal/projects/{id}/alerts/{alert_id}/acknowledge`
- `GET /temporal/projects/{id}/thresholds` / `PUT` — expose `AlertRuleEngine.DEFAULT_THRESHOLDS`

### 1c. Inference Model Selector

**File:** `perceptra-console/src/components/inference/ModelSelector.tsx`
- Remove `mockModels` array
- Accept `projectId` prop, call `useProjectModels(projectId)`, filter to `has_production_version: true`
- The selected item must carry `version_id` (UUID of the deployed ModelVersion)

**File:** `perceptra-console/src/hooks/useInference.tsx`
- Replace `/infer/18` hardcode with `/infer/{selectedVersion.version_id}`
- Replace bare `fetch()` with `apiFetch()` (adds auth token — the endpoint uses `get_current_user`)

### 1d. Housekeeping (apply throughout)
- `useClasses.tsx`: remove `setTimeout(1000ms)` mock delay; call `GET /api/v1/classes` directly
- Remove 60+ `console.log` statements from production code (find with `grep -r "console\.log"`)
- `useProgress.ts`: remove mock short-circuit when `VITE_NODE_ENV === 'development'`

---

## Priority 2 — Real-Time Job Progress via SSE (1–2 weeks)

**Why unique:** All competitors show a spinner. Perceptra can show live epoch-by-epoch metrics streaming during training.

### How it works

`track_progress()` already writes `{percentage, status, message, isComplete}` JSON to Redis under the task ID key. An SSE endpoint just needs to poll that key and stream it.

**New file:** `perceptra-hub/perceptra_hub/api/routers/progress/queries/stream_progress.py`

```
GET /api/v1/tasks/{task_id}/stream?token={access_token}
Content-Type: text/event-stream
```

- `EventSource` API does not support custom headers → accept token as query param, validate via `decode_access_token()`
- Poll `cache.get(task_id)` via `sync_to_async` every 500ms inside an async generator
- `StreamingResponse(generator(), media_type="text/event-stream")` with `X-Accel-Buffering: no` header (required for nginx to not buffer SSE)
- Send a heartbeat comment line `": heartbeat\n\n"` every 15 seconds to keep the connection alive through proxies
- Stop when `isComplete=True` or after 1 hour max

**New frontend hook:** `perceptra-console/src/hooks/useTaskStream.ts`

```typescript
// Opens EventSource to /api/v1/tasks/{taskId}/stream?token={accessToken}
// Returns { progress, status, message, isComplete, error }
// Closes EventSource on unmount or when isComplete=true
```

Get token via `authStorage.getAccessToken()` from `src/services/authService.ts`.

**Wire to:**
- Auto-Annotate: replace polling in `useAutoAnnotate.ts` after Priority 1
- Training sessions: replace 3-second interval polling in `TrainingSessionDetails.tsx`
- Similarity scans: `ScanHistory.tsx` / `ScanResults.tsx`

---

## Priority 3 — Embedding-Based Active Learning (2–3 weeks)

**Why unique:** Competitors offer uncertainty sampling (confidence thresholding). Embedding-based diversity (greedy k-center coreset) ensures you label structurally different images, not just uncertain ones. The combined score is genuinely novel in a no-code MLOps tool.

### New Django model

**`perceptra-hub/perceptra_hub/common_utils/embeddings/models.py`** (or a new `embeddings` app):

```python
class ImageEmbedding(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name='embeddings')
    model_name = models.CharField(max_length=50, default='clip-vit-base-patch32')
    embedding = models.JSONField()  # 512 floats
    computed_at = models.DateTimeField(auto_now=True)
    class Meta:
        unique_together = [('image', 'model_name')]
        indexes = [models.Index(fields=['image', 'model_name'])]
```

Migration needed: `perceptra-hub/perceptra_hub/embeddings/migrations/0001_initial.py`

### New Celery task

**`perceptra-hub/perceptra_hub/api/tasks/compute_embeddings.py`**

```python
@shared_task(bind=True, queue='embeddings', max_retries=1, soft_time_limit=3600)
def compute_embeddings_for_project(self, project_id, image_ids=None, model_name='clip-vit-base-patch32')
```

- Load images in batches of 32 via existing storage adapter
- Use `transformers.CLIPProcessor` + `CLIPModel.get_image_features()` (CPU, ~200ms/image)
- Bulk upsert `ImageEmbedding` records
- Call `track_progress(task_id, ...)` per batch → SSE-streamable

Add `Queue("embeddings")` to `CELERY_TASK_QUEUES` in `api/config/celery_config.py`. Add `embeddings_worker` to `supervisord.conf` (concurrency 2).

**New endpoint:** `POST /api/v1/active-learning/projects/{id}/compute-embeddings` → returns `{task_id}` (same pattern as `auto_annotate.py`)

### New diversity utility

**`perceptra-hub/perceptra_hub/common_utils/active_learning/diversity.py`** (new):

Greedy k-center coreset selection:
1. Normalize all pool embeddings (cosine distance)
2. Initialize with the unlabeled image furthest from any labeled image
3. Greedily select images maximizing minimum distance to already-selected set
4. Returns `list[int]` of indices into the pool

### Modify existing endpoint

**`perceptra-hub/perceptra_hub/api/routers/active_learning/queries/active_learning.py`**

The existing `diversity_score = 0.5` placeholder is the exact insertion point. Replace with:
- Query `ImageEmbedding` for the batch
- Call `greedy_kcenter(pool_embeddings, labeled_embeddings, n=batch_size)`
- `final_score = 0.5 * uncertainty_score + 0.5 * diversity_score`
- Add `strategy` query param: `uncertainty` | `diversity` | `hybrid` (default)

### Frontend

**`perceptra-console/src/hooks/useEvaluationData.ts`** (already updated in Priority 1):
- Change `generatePriorityQueue` replacement to call `GET /api/v1/active-learning/projects/{id}/suggest?strategy=hybrid`

**`perceptra-console/src/components/evaluation/ActiveLearning.tsx`** (or equivalent):
- Add "Initialize Embeddings" button → triggers `POST .../compute-embeddings`, shows `useTaskStream` progress bar
- Add strategy toggle: "Uncertainty" / "Balanced" / "Diversity"

---

## Priority 4 — Model Explainability (GradCAM) in Annotation Tool (2–3 weeks)

**Why unique:** No CV MLOps tool shows "why the model predicted this" inline in the annotation workflow. This directly closes the loop between AI predictions and human understanding.

### New backend endpoint

**`perceptra-hub/perceptra_hub/api/routers/inference/queries/explain.py`**

```
POST /api/v1/model-versions/{version_id}/explain
Body: { image_id: int, bbox: [x_min, y_min, x_max, y_max], class_name: string }
Response: { heatmap_b64: string, overlay_b64: string, class_name: string, confidence: float }
```

**New helper:** `perceptra-hub/perceptra_hub/common_utils/explainability/gradcam.py`

```python
def compute_gradcam(checkpoint_path: str, image_array: np.ndarray, bbox: list, class_idx: int) -> dict
```

- Load `.pt` checkpoint from `ModelVersion.checkpoint_key` via storage adapter (not ONNX — ONNX has no gradient support)
- Use `pytorch-grad-cam` library (`GradCAMPlusPlus` for better bbox localization)
- Target layer: last backbone feature layer (YOLOv8: `model.model[-3]` detection head input)
- Crop image to `bbox` + 20% margin, run forward pass, extract CAM, resize to bbox dimensions
- Return raw heatmap array + pre-composited PNG (jet colormap, 60% alpha) as base64 strings
- Add dependency: `grad-cam>=1.5.0` to `pyproject.toml`

For latency (2–5 seconds CPU), cache results: `cache.set(f"gradcam:{version_id}:{image_id}:{bbox_hash}", result, timeout=3600)`.

### Frontend

**`perceptra-console/src/pages/annotate-tool/AnnotationTool.tsx`** (or wherever the annotation canvas lives):
- Add "Explain" button to annotation toolbar, visible only when a prediction annotation (`annotation_source='prediction'`) is selected
- On click: `POST /api/v1/model-versions/{selectedVersion.id}/explain` with selected bbox
- Render returned `overlay_b64` as a semi-transparent `<img>` layer over the canvas in the bbox region

**New hook:** `perceptra-console/src/hooks/useExplainability.ts`

```typescript
export function useExplainability(modelVersionId: string) {
  return useMutation({
    mutationFn: ({ imageId, bbox, className }: ExplainRequest) =>
      apiFetch(`/api/v1/model-versions/${modelVersionId}/explain`, { method: 'POST', body: JSON.stringify({ image_id: imageId, bbox, class_name: className }) })
  });
}
```

**New type:** `perceptra-console/src/types/explainability.ts`

```typescript
export interface ExplainRequest { imageId: number; bbox: [number, number, number, number]; className: string; }
export interface ExplainResponse { heatmap_b64: string; overlay_b64: string; class_name: string; confidence: number; }
```

**Start with YOLOv8 only** — add framework guard in `compute_gradcam()` that raises `HTTP 422` for non-YOLO frameworks.

---

## Priority 5 — Data Drift Detection (2–3 weeks, requires P3 embeddings)

**Why unique:** Perceptra already knows both training data (via dataset versions) and production data (via auto-annotated images). Connecting these two for automatic drift alerts closes the monitoring loop without any external tool.

### New Django app: `drift`

**`perceptra-hub/perceptra_hub/drift/models.py`**

```python
class DriftBaseline(models.Model):
    model_version = models.OneToOneField(ModelVersion, on_delete=models.CASCADE, related_name='drift_baseline')
    mean_embedding = models.JSONField()           # list[float] length 512
    covariance_diagonal = models.JSONField()      # diagonal only — 512 floats (full 512x512 is too large)
    sample_count = models.IntegerField()
    computed_at = models.DateTimeField(auto_now=True)
    class Meta: db_table = 'drift_baseline'

class DriftReport(models.Model):
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='drift_reports')
    baseline = models.ForeignKey(DriftBaseline, on_delete=models.PROTECT)
    mmd_score = models.FloatField()
    drift_detected = models.BooleanField()
    threshold = models.FloatField(default=0.05)
    production_sample_count = models.IntegerField()
    lookback_days = models.IntegerField(default=7)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    class Meta:
        db_table = 'drift_report'
        ordering = ['-created_at']
        indexes = [models.Index(fields=['model_version', 'created_at'])]
```

Migration: `perceptra-hub/perceptra_hub/drift/migrations/0001_initial.py`

### Utility

**`perceptra-hub/perceptra_hub/common_utils/drift/mmd.py`** (new):
- Unbiased Maximum Mean Discrepancy with RBF kernel
- Use linear-time MMD estimator (random Fourier features) for N > 1000; otherwise exact
- Subsample both distributions to max 1000 images if larger

### Celery Beat task

**`perceptra-hub/perceptra_hub/api/tasks/drift_detection.py`** (new):

```python
@shared_task(name='drift.check_all_models')
def check_all_models_for_drift():
    # For each active production ModelVersion with a DriftBaseline:
    # 1. Query ImageEmbedding for production images uploaded in last 7 days
    # 2. Compute MMD between production embeddings and baseline
    # 3. Create DriftReport; if drift_detected, fire existing alert notification pipeline
```

Add to `CELERY_BEAT_SCHEDULE` in `api/config/celery_config.py`:
```python
'check-drift': {'task': 'drift.check_all_models', 'schedule': crontab(hour=3, minute=30), 'options': {'queue': 'activity'}}
```

### API endpoints

**`perceptra-hub/perceptra_hub/api/routers/drift/queries/drift.py`** (new router, auto-discovered):
```
POST /drift/model-versions/{version_id}/baseline   → compute+store DriftBaseline from training dataset embeddings
GET  /drift/model-versions/{version_id}/reports    → paginated DriftReport list
GET  /drift/model-versions/{version_id}/reports/latest
```

### Frontend

**`perceptra-console/src/pages/models/ModelDetail.tsx`**: Add "Data Drift" card below production info cards. Show `mmd_score` time series using Recharts (same pattern as `TemporalAnalysis.tsx`). Color-code: green < threshold, amber within 2×, red > 2×.

**`perceptra-console/src/hooks/useDriftReports.ts`** (new): `useQuery` wrapping `GET /api/v1/drift/model-versions/{versionId}/reports`.

**`perceptra-console/src/types/drift.ts`** (new):
```typescript
export interface DriftReport { id: number; mmd_score: number; drift_detected: boolean; threshold: number; production_sample_count: number; created_at: string; }
```

---

## Dependency Graph

```
P1 (Fix Mocks) — no dependencies, start immediately
  └──► P2 (SSE) — replaces P1's polling progress
        └──► P3 (Embeddings) — uses P2 for streaming compute progress
              └──► P5 (Drift) — requires ImageEmbedding from P3

P4 (GradCAM) — independent, only needs ModelVersion.checkpoint_key
```

---

## Verification Plan

### Priority 1
1. Open Auto-Annotate page: model selector should show real deployed models (not hardcoded list)
2. Start auto-annotation: progress bar should reflect real task (poll `GET /api/v1/progress/{task_id}`)
3. Evaluation Dashboard: charts should show real data from DB, not random-seeded values
4. Inference page: model selector loads real models; inference call returns real predictions

### Priority 2
1. `curl -N "http://localhost:29085/api/v1/tasks/{task_id}/stream?token={jwt}"` while a training job runs → confirm SSE events arrive with increasing `percentage`
2. Start training from UI, confirm training progress bar updates in real time without page refresh

### Priority 3
1. `POST /api/v1/active-learning/projects/{id}/compute-embeddings` → returns `task_id`
2. Poll or stream that task until complete; verify `ImageEmbedding` rows in DB
3. `GET /api/v1/active-learning/projects/{id}/suggest?strategy=hybrid` → confirm diversity scores ≠ 0.5
4. UI "Smart Sample" shows different images than uncertainty-only sampling

### Priority 4
1. In annotation tool, select a prediction annotation → "Explain" button appears
2. Click Explain → heatmap overlay renders within 5 seconds
3. `curl -X POST .../explain -F "image_id=1 bbox=[0.1,0.1,0.5,0.5] class_name=person"` → 200 with base64 PNG

### Priority 5
1. `POST .../drift/baseline` for a production model version → `DriftBaseline` row in DB
2. Run `check_all_models_for_drift.delay()` directly → `DriftReport` row created
3. Model Detail page: "Data Drift" section renders chart with `mmd_score` over time
4. Simulate high-drift data (all-black images): verify `drift_detected=True` and alert fired
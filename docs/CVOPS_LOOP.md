# CV DevOps Loop: Closing the Gap — Inference, Auto-Annotation & Continuous Training

## Context

Perceptra Hub has a complete annotation pipeline, production-grade training orchestration (multi-provider: GPU, on-prem agents, SageMaker, Vertex AI, K8s), and a rich model registry. The loop is broken at inference: `REMOTE_INFERENCE_URL` is an unimplemented placeholder. Without a real inference service, trained models cannot serve predictions, new images cannot be auto-annotated, and there is no feedback signal to trigger re-training.

**The loop to close:** Annotate → Version → Train → **Deploy** → **Infer** → **Auto-annotate** → Review corrections → **Retrain trigger** → repeat.

---

## Gap Analysis (what's missing)

| Gap | Evidence |
|-----|---------|
| No inference service | `common_utils/inference/utils.py` proxies to `REMOTE_INFERENCE_URL` env var — no service exists |
| No deployment orchestration | `ModelVersion.deployment_status` field exists but nothing sets it |
| No auto-annotation from trained models | Only SAM (`perceptra-seg`) is used; `annotation_source='prediction'` + `model_version` fields exist but never written by model inference |
| No continuous retraining trigger | No policy model, no Beat task watching for annotation accumulation |
| No champion/challenger | After training completes, new version is never compared to current production |
| No inference agent | `agent/main.py` handles training only; no inference job path |

---

## Architecture: The Closed Loop

```
Upload Images
      │
      ▼
[EXISTS] Annotation Tool (manual + SAM AI-assist via perceptra-seg)
      │
      ▼
[EXISTS] Dataset Version (YOLO/COCO/VOC snapshot, train/val/test splits)
      │
      ▼
[EXISTS] Training Orchestration → ONNX + checkpoint stored in storage
      │
      ▼
[PHASE 1] InferenceOrchestrator.deploy() → perceptra-inference service loads ONNX
      │
      ▼
[PHASE 1] POST /api/v1/model-versions/{id}/deploy → deployment_status = 'production'
      │
      ▼
[PHASE 2] Auto-annotation Celery task → run_inference() → Annotation(source='prediction')
      │
      ▼
[EXISTS] Active learning scoring (uncertainty/diversity on stored predictions)
      │
      ▼
[EXISTS] Human review: correct predictions → AnnotationAudit (TP/FP/FN, IoU)
      │
      ▼
[PHASE 3] RetrainingPolicy Beat task: N corrections → new DatasetVersion → TrainingOrchestrator
      │
      ▼
[PHASE 4] Champion/challenger: new version evaluated vs. current production → promote if better
      │
      └──────────────────────────────────────────────────────► REPEAT
```

---

## Existing Code to Reuse (do not reinvent)

| What to reuse | Location |
|---------------|----------|
| `perceptra-seg/service/main.py` — `create_app()` with `app.state.models` dict | Mirror exactly for `perceptra-inference/service/main.py` |
| `perceptra-seg/perceptra_seg/backends/base.py` — `BaseSegmentationBackend` protocol | Mirror for `BaseDetectionBackend` |
| `training/orchestrator.py` — `TrainingOrchestrator` | Mirror class structure for `InferenceOrchestrator` |
| `compute/adapters.py` — `BaseComputeAdapter` + `OnPremiseAgentAdapter` | Mirror for `BaseInferenceAdapter` |
| `agent/main.py` — `AgentClient` + `JobExecutor` polling loop | Extend (not replace) for inference jobs |
| `evaluation/pipeline.py` — `ModelEvaluationPipeline`, `EvaluationReport`, `ModelPerformanceMetrics` | Reuse directly in champion/challenger task |
| `common_utils/inference/utils.py` — `run_inference()` | Already correct; only rename env var |
| `common_utils/progress/core.py` — `track_progress()` | Use in auto-annotation and evaluation tasks |
| `api/tasks/training/train_model.py` — upload helper + callback pattern | Copy pattern for inference artifact handling |

---

## Phase 1 — `perceptra-inference` Service + Deploy API (Weeks 1–3)

### 1a. New microservice: `perceptra-inference/`

Mirror `perceptra-seg/` structure exactly.

```
perceptra-inference/
  Dockerfile
  Dockerfile.gpu              # FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime + onnxruntime-gpu
  docker-compose.yml
  pyproject.toml
  perceptra_inference/
    config.py                 # InferenceConfig (pydantic-settings); INFERENCE_MODEL_NAMES, device, api_keys
    model_registry.py         # ModelRegistry: load_model(version_id, onnx_bytes, task), predict(), unload_model()
    backends/
      base.py                 # BaseDetectionBackend(Protocol): load(), predict(image, conf, max_det) → list[Prediction]
      onnx_yolo.py            # YOLODetectionBackend: onnxruntime.InferenceSession, pre/post-process, NMS
    models.py                 # Pydantic schemas: Prediction, BoundingBox, DetectionResult, LoadModelRequest
    exceptions.py             # ModelNotLoadedError, InferenceError
    utils/
      image_io.py             # load_image_from_bytes() — copy from perceptra_seg/utils/image_io.py
      nms.py                  # Non-maximum suppression
  service/
    main.py                   # create_app() — identical structure to perceptra-seg/service/main.py
                              # startup: load models from INFERENCE_MODEL_IDS env var (presigned URLs)
                              # app.state.models: dict[str, BaseDetectionBackend]
    routes.py                 # /v1/healthz, /v1/infer/{model_version_id}, /v1/models/load, DELETE /v1/models/{id}
    middleware.py             # Copy LoggingMiddleware from perceptra-seg
```

**`service/routes.py` key contracts:**
- `POST /v1/models/load` — body: `{version_id, storage_url, task}` → downloads ONNX, calls `registry.load_model()`
- `DELETE /v1/models/{version_id}` → `registry.unload_model()`
- `POST /v1/infer/{model_version_id}` — multipart: `file`, `confidence_threshold`, `max_detections` → `{"predictions": [...]}`
- `GET /v1/healthz` → `{"status": "ok", "loaded_models": [...]}`

**Root `docker-compose.yml` addition:**
```yaml
perceptra-inference:
  build: ./perceptra-inference
  ports: ["29087:8080"]
  environment:
    - INFERENCE_SERVICE_URL=http://perceptra-inference:8080
  deploy:
    resources:
      reservations:
        devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
```

### 1b. `InferenceOrchestrator` inside perceptra-hub

**New file:** `perceptra_hub/inferences/orchestrator.py`

```python
class InferenceOrchestrator:
    def deploy(self, model_version: ModelVersion, target: str = 'production') -> InferenceDeployment:
        # 1. Assert model_version.status == 'trained' and onnx_model_key is set
        # 2. Generate presigned download URL for onnx_model_key via storage profile
        # 3. POST {INFERENCE_SERVICE_URL}/v1/models/load with {version_id, storage_url, task}
        # 4. Create InferenceDeployment record
        # 5. model_version.deployment_status = target; save()
        # 6. If target == 'production': model_version.mark_as_production(user)

    def undeploy(self, model_version: ModelVersion) -> None:
        # 1. DELETE {INFERENCE_SERVICE_URL}/v1/models/{version_id}
        # 2. model_version.deployment_status = 'retired'; save()

    def _get_inference_url(self) -> str:
        return os.getenv('INFERENCE_SERVICE_URL', 'http://perceptra-inference:8080')
```

### 1c. New model: `InferenceDeployment` — add to `inferences/models.py`

```python
class InferenceDeployment(models.Model):
    model_version         = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='deployments')
    target_env            = models.CharField(choices=[('staging','Staging'),('production','Production')])
    inference_service_url = models.CharField(max_length=500)
    deployed_by           = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    deployed_at           = models.DateTimeField(auto_now_add=True)
    undeployed_at         = models.DateTimeField(null=True, blank=True)
    is_active             = models.BooleanField(default=True)
```

### 1d. Deploy/undeploy API endpoints

**New file:** `perceptra_hub/api/routers/inference/queries/deploy.py`
- `POST /api/v1/model-versions/{version_id}/deploy` — body: `{target_env}` → calls `InferenceOrchestrator.deploy()`
- `POST /api/v1/model-versions/{version_id}/undeploy` → calls `InferenceOrchestrator.undeploy()`

### 1e. Fix `REMOTE_INFERENCE_URL` → `INFERENCE_SERVICE_URL`

**Modify:** `perceptra_hub/common_utils/inference/utils.py`
- Rename env var `REMOTE_INFERENCE_URL` to `INFERENCE_SERVICE_URL`
- Change URL path from `/api/v1/infer/` to `/v1/infer/` (matches new service)
- No other logic changes needed — `run_inference()` already has the right shape

### 1f. New migration

`perceptra_hub/inferences/migrations/0002_inferencedeployment.py`

---

## Phase 2 — Auto-Annotation Pipeline (Weeks 4–5)

### 2a. New Celery task

**New file:** `perceptra_hub/api/tasks/auto_annotate.py`

```python
@shared_task(bind=True, queue='auto_annotate', max_retries=2)
def auto_annotate_images(self, project_id, image_ids, model_version_id, confidence_threshold=0.25, task_id=None):
    # For each image_id:
    # 1. Fetch Image + ProjectImage
    # 2. Call run_inference(image, model_version_id, confidence_threshold) → predictions
    # 3. For each prediction: look up AnnotationClass by class_label in project's AnnotationGroup
    #    Skip unknown classes (do NOT auto-create); log warning
    # 4. Bulk-create Annotation objects:
    #    annotation_source='prediction', model_version=version_id, confidence=score,
    #    reviewed=False, annotation_type=BoundingBox, data=[xmin,ymin,xmax,ymax]
    # 5. Mark ProjectImage.annotated = True
    # 6. track_progress(task_id, percent, ...)
```

### 2b. New API endpoint

**New file:** `perceptra_hub/api/routers/inference/queries/auto_annotate.py`

`POST /api/v1/projects/{project_id}/auto-annotate`
- Body: `{model_version_id, image_ids: list[int], confidence_threshold: float = 0.25}`
- Dispatches `auto_annotate_images.apply_async(queue='auto_annotate', task_id=x_request_id)`
- Returns: `{task_id, status: "queued"}`

### 2c. Celery queue + supervisord

**Modify `api/config/celery_config.py`:** Add `Queue("auto_annotate")` to `CELERY_TASK_QUEUES`

**Modify `supervisord.conf`:** Add program `auto_annotate` worker, concurrency 4, queue `auto_annotate`

### 2d. Inference agent extension (on-premise)

**New file:** `perceptra_hub/agent/inference_executor.py`

```python
class InferenceJobExecutor:
    # Receives job: {job_id, model_version_id, onnx_presigned_url, image_ids, confidence_threshold}
    # 1. Download ONNX to /tmp/{job_id}.onnx
    # 2. Load onnxruntime.InferenceSession
    # 3. For each image: download, run inference, POST predictions to
    #    POST /api/v1/projects/{project_id}/auto-annotate/results (new endpoint)
    # 4. Cleanup temp files
```

**Modify `agent/main.py`:**
- Add poll to `GET /api/v1/agents/poll/inference-job`
- Dispatch based on `job['type']`: `'training'` → existing `JobExecutor`, `'inference'` → `InferenceJobExecutor`

**New file:** `perceptra_hub/api/routers/compute/queries/agent_inference.py`
- `GET /api/v1/agents/poll/inference-job` — returns pending inference job for agent's org
- `POST /api/v1/agents/inference-jobs/{job_id}/results` — receives prediction results

---

## Phase 3 — Continuous Retraining Triggers (Weeks 6–7)

### 3a. New model: `RetrainingPolicy` — add to `training/models.py`

```python
class RetrainingPolicy(models.Model):
    policy_id                   = models.CharField(unique=True)
    model                       = models.ForeignKey(Model, related_name='retraining_policies')
    is_active                   = models.BooleanField(default=True)
    trigger_type                = models.CharField(choices=['annotation_count','correction_rate','time_elapsed','combined'])

    # Thresholds
    min_new_annotations         = models.PositiveIntegerField(default=100)
    min_correction_rate         = models.FloatField(null=True)   # FP+FN rate threshold
    max_days_since_training     = models.PositiveIntegerField(null=True)
    lookback_days               = models.PositiveIntegerField(default=30)

    # Action on trigger
    auto_create_dataset_version = models.BooleanField(default=True)
    auto_submit_training        = models.BooleanField(default=True)
    compute_profile             = models.ForeignKey(ComputeProfile, null=True)

    # Anti-spam
    min_hours_between_runs      = models.PositiveIntegerField(default=24)
    last_triggered_at           = models.DateTimeField(null=True)
```

### 3b. New service: `training/retraining_service.py`

```python
class RetrainingService:
    def evaluate_all_policies(self) -> list[dict]:
        # For each active RetrainingPolicy:
        # 1. Check cooldown (last_triggered_at + min_hours)
        # 2. Evaluate trigger condition (annotation count or correction rate)
        # 3. If triggered: _trigger_retraining(policy)

    def _check_annotation_count_trigger(self, policy) -> bool:
        # Count Annotation.objects.filter(
        #   project_image__project=policy.model.project,
        #   reviewed=True, is_deleted=False,
        #   reviewed_at__gte=cutoff
        # ).count() >= policy.min_new_annotations

    def _check_correction_rate_trigger(self, policy) -> bool:
        # Read AnnotationAudit for latest trained version
        # Compute (FP + FN) / total_predictions >= policy.min_correction_rate

    def _trigger_retraining(self, policy) -> None:
        # 1. Snapshot: Version.objects.create() + bulk VersionImage from current ProjectImages
        # 2. ModelVersion.objects.create(parent_version=current_production, status='draft')
        # 3. TrainingSession.objects.create()
        # 4. TrainingOrchestrator(model_version).submit_training(session, policy.compute_profile_id)
        # 5. policy.last_triggered_at = now(); policy.save()
```

### 3c. Celery Beat task

**New file:** `perceptra_hub/api/tasks/retraining.py`

```python
@shared_task(queue='activity')
def evaluate_retraining_policies():
    return RetrainingService().evaluate_all_policies()
```

**Modify `api/config/celery_config.py`:** Add to `CELERY_BEAT_SCHEDULE`:
```python
'evaluate-retraining-policies': {
    'task': 'api.tasks.retraining.evaluate_retraining_policies',
    'schedule': crontab(minute=0),   # Every hour
    'options': {'queue': 'activity'}
}
```

### 3d. Policy management API

**New file:** `perceptra_hub/api/routers/training/queries/retraining_policy.py`
- `POST /api/v1/models/{model_id}/retraining-policies`
- `GET /api/v1/models/{model_id}/retraining-policies`
- `PATCH /api/v1/models/{model_id}/retraining-policies/{policy_id}`
- `POST /api/v1/models/{model_id}/retraining-policies/{policy_id}/trigger` — manual trigger

### 3e. New migration

`perceptra_hub/training/migrations/000X_retrainingpolicy.py`

---

## Phase 4 — Champion/Challenger Evaluation (Weeks 8–9)

### 4a. New model: `ModelEvaluation` — add to `inferences/models.py`

```python
class ModelEvaluation(models.Model):
    evaluation_id          = models.CharField(unique=True)
    challenger             = models.ForeignKey(ModelVersion, related_name='challenger_evaluations')
    champion               = models.ForeignKey(ModelVersion, related_name='champion_evaluations', null=True)
    dataset_version        = models.ForeignKey(Version, on_delete=models.PROTECT)
    status                 = models.CharField(choices=['pending','running','completed','failed'])
    challenger_metrics     = models.JSONField(default=dict)   # precision, recall, f1, mAP50 from EvaluationReport
    champion_metrics       = models.JSONField(default=dict)
    primary_metric         = models.CharField(default='f1_score')
    improvement_delta      = models.FloatField(null=True)     # challenger - champion
    auto_promote_threshold = models.FloatField(default=0.02)
    recommendation         = models.CharField(choices=['promote','keep_champion','inconclusive'], null=True)
    auto_promoted          = models.BooleanField(default=False)
    triggered_by           = models.ForeignKey(User, null=True)
    created_at             = models.DateTimeField(auto_now_add=True)
    completed_at           = models.DateTimeField(null=True)
```

### 4b. New Celery task

**New file:** `perceptra_hub/api/tasks/champion_challenger.py`

```python
@shared_task(bind=True, queue='evaluation', max_retries=2)
def run_champion_challenger_evaluation(self, evaluation_id, task_id=None):
    # 1. Load ModelEvaluation, set status='running'
    # 2. Deploy challenger to staging: InferenceOrchestrator(challenger).deploy(target='staging')
    # 3. Run model validation on dataset_version.val images (reuse validate.py logic):
    #    - run_inference() on each val image → store in PredictionImageResult + PredictionOverlay
    # 4. Use ModelEvaluationPipeline (evaluation/pipeline.py) to compute EvaluationReport for challenger
    # 5. If champion exists and is_deployed: read its existing PredictionImageResult records
    #    (or run inference if no stored results) → compute EvaluationReport for champion
    # 6. Compute improvement_delta = challenger_f1 - champion_f1
    # 7. Set recommendation:
    #    delta >= threshold → 'promote'
    #    delta < -0.01 → 'keep_champion'
    #    else → 'inconclusive'
    # 8. If recommend == 'promote' and auto_promote policy set:
    #    InferenceOrchestrator(challenger).deploy(target='production')
    #    evaluation.auto_promoted = True
    # 9. evaluation.status='completed'; evaluation.completed_at=now(); save()
```

### 4c. Hook into training completion

**Modify `perceptra_hub/api/tasks/training/train_model.py`:**

At the end of step 10 (after `model_version.status = 'trained'; model_version.save()`), insert:

```python
champion = model_version.model.get_production_version()
eval_record = ModelEvaluation.objects.create(
    evaluation_id=str(uuid.uuid4()),
    challenger=model_version,
    champion=champion,
    dataset_version=model_version.dataset_version,
)
run_champion_challenger_evaluation.apply_async(
    kwargs={'evaluation_id': eval_record.evaluation_id},
    queue='evaluation', countdown=30
)
```

### 4d. Evaluation + promotion API

**New file:** `perceptra_hub/api/routers/inference/queries/evaluation.py`
- `GET /api/v1/model-evaluations/{evaluation_id}` — status + metrics
- `GET /api/v1/models/{model_id}/evaluations` — history
- `POST /api/v1/model-evaluations/{evaluation_id}/promote` — human override promotion

### 4e. Supervisord + queue

**Modify `supervisord.conf`:** Add `evaluation_worker` program (concurrency 2, queue `evaluation`)

**Modify `api/config/celery_config.py`:** Add `Queue("evaluation")` to `CELERY_TASK_QUEUES`

### 4f. New migration

`perceptra_hub/inferences/migrations/0003_modelevaluation.py`

---

## Complete File Manifest

### New service (perceptra-inference/)
| File | Purpose |
|------|---------|
| `perceptra-inference/Dockerfile.gpu` | GPU image with onnxruntime-gpu |
| `perceptra-inference/perceptra_inference/config.py` | InferenceConfig (pydantic-settings) |
| `perceptra-inference/perceptra_inference/model_registry.py` | In-memory model registry (LRU) |
| `perceptra-inference/perceptra_inference/backends/base.py` | BaseDetectionBackend protocol |
| `perceptra-inference/perceptra_inference/backends/onnx_yolo.py` | ONNX Runtime YOLO/RT-DETR backend |
| `perceptra-inference/perceptra_inference/models.py` | Pydantic schemas |
| `perceptra-inference/perceptra_inference/utils/image_io.py` | image loading (mirror perceptra-seg) |
| `perceptra-inference/perceptra_inference/utils/nms.py` | NMS helper |
| `perceptra-inference/service/main.py` | FastAPI app factory (mirror perceptra-seg) |
| `perceptra-inference/service/routes.py` | /v1/infer, /v1/models/load, /v1/healthz |
| `perceptra-inference/service/middleware.py` | LoggingMiddleware (copy from perceptra-seg) |

### New files in perceptra-hub/
| File | Phase |
|------|-------|
| `perceptra_hub/inferences/orchestrator.py` | 1 |
| `perceptra_hub/api/routers/inference/queries/deploy.py` | 1 |
| `perceptra_hub/api/tasks/auto_annotate.py` | 2 |
| `perceptra_hub/api/routers/inference/queries/auto_annotate.py` | 2 |
| `perceptra_hub/agent/inference_executor.py` | 2 |
| `perceptra_hub/api/routers/compute/queries/agent_inference.py` | 2 |
| `perceptra_hub/training/retraining_service.py` | 3 |
| `perceptra_hub/api/tasks/retraining.py` | 3 |
| `perceptra_hub/api/routers/training/queries/retraining_policy.py` | 3 |
| `perceptra_hub/api/tasks/champion_challenger.py` | 4 |
| `perceptra_hub/api/routers/inference/queries/evaluation.py` | 4 |

### Modified files in perceptra-hub/
| File | Change |
|------|--------|
| `perceptra_hub/inferences/models.py` | Add `InferenceDeployment` + `ModelEvaluation` |
| `perceptra_hub/training/models.py` | Add `RetrainingPolicy` |
| `perceptra_hub/common_utils/inference/utils.py` | Rename env var; fix URL path |
| `perceptra_hub/api/tasks/training/train_model.py` | Hook champion/challenger after step 10 |
| `perceptra_hub/agent/main.py` | Extend poll loop for inference jobs |
| `perceptra_hub/api/config/celery_config.py` | Add queues + Beat entry |
| `supervisord.conf` | Add auto_annotate + evaluation workers |
| `docker-compose.yml` | Add perceptra-inference service on port 29087 |

---

## Frontend Work (Console)

| Component | Change |
|-----------|--------|
| `ModelDetail.tsx` | Add "Deploy to Staging / Production" button → calls `POST /model-versions/{id}/deploy` |
| `ModelDetail.tsx` | Add "Evaluations" tab showing `ModelEvaluation` records with metrics diff and Promote button |
| `pages/models/ModelsList.tsx` | Show `deployment_status` badge on model cards |
| `pages/inference/Inference.tsx` | Already calls `/api/v1/infer/{modelId}` — works once service is live; verify |
| New page: `AutoAnnotate.tsx` | Trigger auto-annotation: select model version, image set, confidence → POST `/auto-annotate` |
| New page: `RetrainingPolicy.tsx` | CRUD for `RetrainingPolicy` per model; show last triggered + next threshold |

---

## Verification Plan

**Phase 1:**
1. Stand up `perceptra-inference` container; curl `/v1/healthz`
2. POST `/v1/models/load` with an ONNX file from storage; verify model loaded
3. POST `/v1/infer/{id}` with a test image; verify predictions returned
4. Call deploy API; verify `ModelVersion.deployment_status = 'production'`

**Phase 2:**
1. Deploy a model; POST `/api/v1/projects/{id}/auto-annotate` with 10 unannotated image IDs
2. Poll task progress; verify `Annotation(annotation_source='prediction')` records in DB
3. Open annotation tool; confirm prediction annotations visible and reviewable
4. Test on-premise agent: start agent, assign inference job, verify results posted back

**Phase 3:**
1. Set `RetrainingPolicy(min_new_annotations=5)`; review 5 annotations manually
2. Run `evaluate_retraining_policies` task directly (`task.delay()`)
3. Verify new `ModelVersion` + `TrainingSession` created with correct `parent_version`

**Phase 4:**
1. Wait for training to complete (or use a pre-trained ONNX)
2. Verify `ModelEvaluation` record created automatically after training
3. Inspect `challenger_metrics` vs `champion_metrics` in DB
4. Call `POST /model-evaluations/{id}/promote`; verify `ModelVersion.deployment_status = 'production'`
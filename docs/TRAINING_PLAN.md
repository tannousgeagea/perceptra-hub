# Plan: Training Pipeline Robustification + Model Size Selection

## Context

The on-premise training pipeline (YOLO, RF-DETR) has several production-breaking gaps: no OOM handling, a fundamental epoch-loop mismatch in YOLO, no worker-crash recovery, race conditions in the orchestrator, and no ability to resume interrupted training. Alongside this, users cannot currently select a model size variant (nano/small/medium/large/xlarge) when creating a model — the architecture variant is buried in a JSON blob with no UI exposure. This plan addresses both.

---

## Part 1: Training Pipeline Robustification

### Phase A — Critical / Production-Breaking Fixes

#### A1. YOLO Epoch-Loop Mismatch (yolo_trainer.py)
**Problem:** `train_epoch()` trains ALL epochs on the first call (ultralytics handles the loop internally), then returns cached metrics for all subsequent epoch calls. The base loop runs redundant `validate()` calls and wastes time.  
**Fix:** Override the full `train()` method in `YOLOTrainer` instead of relying on the base epoch loop. Call `on_train_start`, run ultralytics training once, emit `on_epoch_end` for each completed epoch from YOLO's callback, then call `on_train_end`. Remove the base-class epoch iteration entirely for this trainer.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/yolo_trainer.py`

#### A2. RF-DETR OOM in Backward Pass (rfdetr_trainer.py)
**Problem:** `scaler.backward()` and `scaler.step()` in `train_epoch()` (around lines 262–263) have no try-except. A CUDA OOM crashes the entire worker process.  
**Fix:** Wrap the forward + backward + step block in a `try/except torch.cuda.OutOfMemoryError` block. On OOM: log the error, call `on_train_error` callback, set session status to `failed` with a clear error message, and re-raise so the Celery task captures it cleanly.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/rfdetr_trainer.py`

#### A3. data.yaml Side-Effect (yolo_trainer.py)
**Problem:** `prepare_dataset()` rewrites the user's original `data.yaml` in-place (lines 119–121). If training is interrupted, the file is left corrupted.  
**Fix:** Write a derived `data_training.yaml` to `output_dir` instead of modifying the original. Pass the derived path to ultralytics. The original user file is never touched.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/yolo_trainer.py`

#### A4. Worker-Crash Recovery (train_model.py)
**Problem:** If the Celery worker crashes mid-training, the `TrainingSession` stays in `running` state forever. No cleanup of `/tmp/training/{job_id}`.  
**Fix:**
- Add `acks_late=True` and `reject_on_worker_lost=True` to the `@shared_task` decorator so tasks are re-queued on worker crash.
- Add a `soft_time_limit` and `time_limit` (e.g., 6h/7h) to auto-terminate runaway tasks.
- Wrap the full task body in a final `except WorkerLostError` / `SoftTimeLimitExceeded` handler that sets session status to `failed` with message "Worker terminated unexpectedly".
- Add cleanup of `/tmp/training/{job_id}` in a `finally` block.  
**File:** `perceptra-hub/perceptra_hub/api/tasks/training/train_model.py`

#### A5. Orchestrator Race Condition on Capacity (orchestrator.py)
**Problem:** `_check_provider_availability()` counts active jobs without a database lock. Concurrent requests can both pass the capacity check simultaneously.  
**Fix:** When checking platform GPU provider capacity, use `select_for_update()` on the active job count query, or use a Redis-based atomic counter keyed by `compute_profile_id`.  
**File:** `perceptra-hub/perceptra_hub/training/orchestrator.py`

#### A6. Submission Outside Transaction (orchestrator.py)
**Problem:** `TrainingJob.objects.create()` is inside `transaction.atomic()` but the actual provider submission (line 100) is outside it. On retry, a duplicate job record is created.  
**Fix:** Assign an idempotency key (`external_job_id = str(uuid4())`) before the transaction, and use `get_or_create(external_job_id=...)` to prevent duplicate submissions. Move the provider call inside the atomic block or use a two-phase pattern (record PENDING → submit → update to SUBMITTED).  
**File:** `perceptra-hub/perceptra_hub/training/orchestrator.py`

---

### Phase B — High Priority Improvements

#### B1. Add `on_train_error` Callback (base.py + all trainers)
**Fix:** Add `on_train_error(epoch: int, error: Exception)` to `TrainingCallbacks`. Call it in the Celery task's exception handler and from any try-except in trainers. The `PlatformTrainingCallbacks` implementation should set session status to `failed`, persist `error_message` and `error_traceback`.  
**Files:** `perceptra-hub/perceptra_hub/training/trainers/base.py`, `train_model.py`

#### B2. Implement Early Stopping (base.py)
**Problem:** `patience` config exists and `_check_early_stopping()` is stubbed but does nothing.  
**Fix:** Track `best_val_metric` and a `patience_counter` in the base training loop. If validation metric does not improve for `patience` consecutive epochs, set `self._stop_training = True` and break the loop. Call `on_train_end` with `stop_reason="early_stopping"`.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/base.py`

#### B3. Gradient Clipping for RF-DETR (rfdetr_trainer.py)
**Fix:** Add `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)` after `scaler.unscale_()` and before `scaler.step()`. Expose `grad_clip_norm` as a config parameter (default `0.1`).  
**File:** `perceptra-hub/perceptra_hub/training/trainers/rfdetr_trainer.py`

#### B4. LR Warmup for RF-DETR (rfdetr_trainer.py)
**Fix:** Add a `LinearLR` warmup scheduler for the first `warmup_epochs` (default 3). Chain with cosine/step using `SequentialLR`. Expose `warmup_epochs` in config.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/rfdetr_trainer.py`

#### B5. Checkpoint Resume for RF-DETR (rfdetr_trainer.py)
**Problem:** Checkpoint structure is well-defined (model + optimizer + scheduler + epoch) but no code uses `checkpoint_path` to resume.  
**Fix:** In `create_model()`, if `config.checkpoint_path` is set, load model state and also restore `optimizer.load_state_dict()`, `scheduler.load_state_dict()`, and set `self.start_epoch = checkpoint['epoch'] + 1`. Pass `start_epoch` to the base loop.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/rfdetr_trainer.py`

---

### Phase C — Stability & Maintenance

#### C1. Config Validation at Factory (factory.py)
**Fix:** Add a `validate_config(task, framework, config_dict)` function that checks required keys (`epochs`, `batch_size`, `learning_rate`, `image_size`, `output_dir`) and raises `ValueError` with a clear message before the trainer is created.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/factory.py`

#### C2. Checkpoint Cleanup Policy (rfdetr_trainer.py)
**Fix:** In `save_checkpoint()`, after saving the current epoch checkpoint, delete epoch checkpoints older than `keep_last_n=3` (configurable). Always preserve `best.pt`.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/rfdetr_trainer.py`

#### C3. Metric Extraction Hardening (yolo_trainer.py)
**Fix:** In `_extract_metrics_from_results()`, after extraction, apply NaN/Inf filtering: `{k: v for k, v in metrics.items() if isinstance(v, float) and math.isfinite(v)}`. Log a warning if any values were dropped.  
**File:** `perceptra-hub/perceptra_hub/training/trainers/yolo_trainer.py`

#### C4. Broader Retry Logic (train_model.py)
**Fix:** Replace the string-based retry check with a proper exception type list: `(torch.cuda.OutOfMemoryError, ConnectionError, TimeoutError)`. Retry up to 3 times with exponential backoff.  
**File:** `perceptra-hub/perceptra_hub/api/tasks/training/train_model.py`

---

## Part 2: Model Size Selection Feature

### Backend Changes

#### 2.1 Add `model_size` Field to ModelVersion (models.py)
**Fix:** Add `model_size = models.CharField(max_length=20, blank=True, default="")` to `ModelVersion`. This makes size queryable and visible in the Django admin without digging through JSON.  
Write and apply a migration.  
**File:** `perceptra-hub/perceptra_hub/training/models.py`  
**Migration:** `perceptra-hub/perceptra_hub/training/migrations/`

#### 2.2 Accept `model_size` in Model Create API
**Fix:** Add `model_size: str = ""` to the model creation request schema. On create, persist it to `ModelVersion.model_size` and also include it in `model_params` so trainers receive it: `config['model_params']['model_size'] = model_size`.  
**File:** `perceptra-hub/perceptra_hub/api/routers/models/` (create model endpoint)

#### 2.3 Return `model_size` in API Responses
**Fix:** Include `model_size` in the `ModelVersionOut` Pydantic schema. The frontend already fetches versions; no new endpoint needed.  
**File:** `perceptra-hub/perceptra_hub/api/routers/models/` (model detail + version serializer)

#### 2.4 Trainer Size Mapping (factory.py or yolo_trainer.py)
Ensure the following size→variant mappings are canonical:
- **ultralytics**: `nano→n`, `small→s`, `medium→m`, `large→l`, `xlarge→x`
- **rfdetr/pytorch**: `small→rfdetr_base`, `large→rfdetr_large`
- **generic**: pass through as-is to `model_params`

Add this mapping in `factory.py` `get_trainer()` so it normalizes before passing to trainers.

---

### Frontend Changes

#### 2.5 Add `ModelSize` Type (types/models.ts)
```typescript
export type ModelSize = 'nano' | 'small' | 'medium' | 'large' | 'xlarge';

// Add to ModelFormData:
modelSize: ModelSize | "";

// Add to ModelCreateRequest:
modelSize: ModelSize | "";
```
**File:** `perceptra-console/src/types/models.ts`

#### 2.6 New Step: `ModelSizeSelection.tsx`
Create `perceptra-console/src/components/models/create/ModelSizeSelection.tsx`.

The component renders **5 size cards** (framework-aware filtering: RF-DETR shows only small/large):

| Size | Icon | Badge | Speed | Accuracy | Use Case |
|------|------|-------|-------|----------|----------|
| Nano | ⚡ | YOLO n | Fastest | Low | Edge/mobile, real-time |
| Small | 🚀 | YOLO s | Fast | Good | Balanced production |
| Medium | ⚖️ | YOLO m | Moderate | High | Recommended default |
| Large | 🎯 | YOLO l | Slow | Very High | High-accuracy tasks |
| XLarge | 🏆 | YOLO x | Slowest | Highest | Max accuracy, cloud GPU |

Each card is a clickable tile with visual speed/accuracy bars. Selected state uses `ring-2 ring-primary`. Medium is pre-selected by default.

#### 2.7 Insert Size Step into Wizard (CreateModel.tsx)
- Re-number steps: 1=Basic Info, 2=Task & Framework, **3=Model Size**, 4=Configuration, 5=Review
- Add validation for step 3: `formData.modelSize !== ""`
- Show/hide size options based on `formData.framework` (ultralytics shows 5; pytorch shows 3: small/medium/large)
**File:** `perceptra-console/src/pages/models/CreateModel.tsx`

#### 2.8 Show Size in ModelReview Step (ModelReview.tsx)
Add a "Model Size" row in the review summary showing the selected size and its description.  
**File:** `perceptra-console/src/components/models/create/ModelReview.tsx`

#### 2.9 Include `modelSize` in API Call (useModels.tsx)
Pass `modelSize` from `ModelFormData` into the `ModelCreateRequest` payload.  
**File:** `perceptra-console/src/hooks/useModels.tsx`

#### 2.10 Display `model_size` Badge in Versions List (ModelVersionList.tsx)
Add a small size pill (e.g., `M` for medium) next to the version number in the versions table — same indigo pill style used for session version badges.  
**File:** `perceptra-console/src/components/models/ModelVersionList.tsx`

---

## Files to Modify

### Backend
| File | Changes |
|------|---------|
| `training/trainers/yolo_trainer.py` | A1 (epoch loop override), A3 (data.yaml fix), C3 (metric NaN filter) |
| `training/trainers/rfdetr_trainer.py` | A2 (OOM handling), B3 (grad clip), B4 (LR warmup), B5 (checkpoint resume), C2 (checkpoint cleanup) |
| `training/trainers/base.py` | B1 (on_train_error callback), B2 (early stopping) |
| `training/trainers/factory.py` | C1 (config validation), 2.4 (size mapping) |
| `api/tasks/training/train_model.py` | A4 (worker crash recovery), B1 (error callback), C4 (retry logic) |
| `training/orchestrator.py` | A5 (capacity race condition), A6 (submission atomicity) |
| `training/models.py` | 2.1 (model_size field) |
| `api/routers/models/` (create + detail) | 2.2 (accept model_size), 2.3 (return model_size) |
| `training/migrations/` | New migration for model_size field |

### Frontend
| File | Changes |
|------|---------|
| `src/types/models.ts` | 2.5 (ModelSize type, ModelFormData update) |
| `src/components/models/create/ModelSizeSelection.tsx` | 2.6 (new component — create) |
| `src/pages/models/CreateModel.tsx` | 2.7 (insert size step) |
| `src/components/models/create/ModelReview.tsx` | 2.8 (show size in review) |
| `src/hooks/useModels.tsx` | 2.9 (include modelSize in payload) |
| `src/components/models/ModelVersionList.tsx` | 2.10 (size badge in table) |

---

## Implementation Order

1. **A1→A3** (YOLO fixes — don't break existing training)
2. **A2** (RF-DETR OOM — prevent crashes)
3. **A4→A6** (infrastructure: worker recovery + orchestrator atomicity)
4. **B1→B5** (stability improvements)
5. **C1→C4** (maintenance)
6. **2.1 + migration** (DB field first)
7. **2.2→2.4** (backend API)
8. **2.5→2.10** (frontend wizard + display)

---

## Verification

1. **Training crash test:** Start RF-DETR training with a small batch, then `kill -9` the worker process. Session must show `failed` with message "Worker terminated unexpectedly", not stay in `running`.
2. **OOM test:** Run RF-DETR with `batch_size=512` on a small GPU. Task must catch OOM, set session `failed`, not crash the worker.
3. **data.yaml preservation:** Start YOLO training, interrupt it. Original `data.yaml` must be unmodified; derived `data_training.yaml` should exist in `output_dir`.
4. **YOLO epoch callbacks:** Confirm epoch metrics appear in `TrainingCheckpoint` for each epoch (not just the final one) by querying the DB during training.
5. **Model size selection:** Create a model via UI selecting "Large". Verify `ModelVersion.model_size = "large"` in DB, and YOLO trainer receives `model_params.model_size = "l"`.
6. **Wizard step:** Navigate through all 5 steps; confirm step 3 blocks advancement if no size is selected; confirm review shows the selected size.
7. **Early stopping:** Train with `patience=2` on a dataset where loss plateaus. Confirm training stops before `max_epochs`.
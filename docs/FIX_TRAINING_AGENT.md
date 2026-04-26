# Plan: Fix Training Agent — Progress Callback, Result Access, Dataset Download & Artifact Upload

## Context

The on-premise training agent (`agent/main.py`) fails immediately on `trainer.set_progress_callback(progress_callback)` because neither `BaseTrainer` nor `YOLOTrainer` define that method. Beyond that, the agent has two more crash paths: it accesses `TrainingResult` as a dict (it's a dataclass), and both `download_dataset` and `upload_artifact` in `AgentClient` are unimplemented stubs. All four issues must be fixed before a training job can complete end-to-end.

---

## Issues to Fix

### Bug 1 — `set_progress_callback` not defined (immediate crash)
`BaseTrainer` has no `set_progress_callback`. Agent calls it at `agent/main.py:305`.

### Bug 2 — `TrainingResult` used as dict (crash after training)
`trainer.train()` returns a `TrainingResult` dataclass. Agent uses `result.get('checkpoint_path')`, `result.get('onnx_path')`, `result.get('metrics', {})` — all `AttributeError`. Key name is also wrong: `checkpoint_path` → `best_checkpoint_path`.

### Bug 3 — No per-epoch progress in `YOLOTrainer`
YOLO trains all epochs internally; the base training loop is bypassed. Even after Bug 1 is fixed, the callback will never fire without hooking into Ultralytics' `on_train_epoch_end` callback.

### Missing 4 — `download_dataset` not implemented
`AgentClient.download_dataset()` just logs a warning. The agent has `dataset_version_id` and `storage_profile_id` from the job payload — the platform must generate a presigned GET URL, and the agent must stream-download and extract the zip.

### Missing 5 — `upload_artifact` not implemented
`AgentClient.upload_artifact()` just logs a warning. The agent must request a presigned PUT URL from the platform per artifact (checkpoint, ONNX, logs), then HTTP PUT the file to that URL.

---

## Implementation Plan

### Step 1 — Add `set_progress_callback` + `_call_progress` to `BaseTrainer`

**File:** `perceptra_hub/training/trainers/base.py`

In `BaseTrainer.__init__`, initialize:
```python
self._progress_callback: Optional[Callable] = None
```

Add two methods after `__init__`:
```python
def set_progress_callback(self, callback: Callable) -> None:
    self._progress_callback = callback

def _call_progress(self, epoch: int, total_epochs: int, metrics: dict) -> None:
    if self._progress_callback:
        try:
            self._progress_callback(epoch, total_epochs, metrics)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")
```

In `BaseTrainer.train()`, after `self.callbacks.on_epoch_end(epoch, train_metrics)` (line 332), add:
```python
self._call_progress(epoch, self.config.epochs, train_metrics.to_dict())
```

This covers `RFDETRTrainer` automatically since it uses the base loop.

---

### Step 2 — Wire per-epoch progress into `YOLOTrainer`

**File:** `perceptra_hub/training/trainers/yolo_trainer.py`

In `YOLOTrainer.train_epoch()`, before `self.model.train(**train_args)`, register a closure as an Ultralytics epoch callback:

```python
def _on_epoch_end(yolo_trainer_obj):
    epoch = yolo_trainer_obj.epoch + 1   # ultralytics is 0-indexed
    total = yolo_trainer_obj.epochs
    metrics = {'train_loss': float(yolo_trainer_obj.loss) if hasattr(yolo_trainer_obj, 'loss') else 0.0}
    if hasattr(yolo_trainer_obj, 'metrics') and yolo_trainer_obj.metrics:
        metrics.update({k: float(v) for k, v in yolo_trainer_obj.metrics.items()})
    self._call_progress(epoch, total, metrics)

self.model.add_callback('on_train_epoch_end', _on_epoch_end)
```

---

### Step 3 — Fix `result.get(...)` dict access in `agent/main.py`

**File:** `perceptra_hub/agent/main.py:312-348`

Replace all three `result.get(...)` blocks with direct dataclass attribute access:

```python
# Checkpoint
if result.best_checkpoint_path and Path(result.best_checkpoint_path).exists():
    storage_key = f"organizations/{job['organization_id']}/models/{job['model_version_id']}/checkpoint.pt"
    if self.client.upload_artifact(Path(result.best_checkpoint_path), storage_key, job['storage_profile_id']):
        artifacts['checkpoint_key'] = storage_key

# ONNX
if result.onnx_path and Path(result.onnx_path).exists():
    storage_key = f"organizations/{job['organization_id']}/models/{job['model_version_id']}/model.onnx"
    if self.client.upload_artifact(Path(result.onnx_path), storage_key, job['storage_profile_id']):
        artifacts['onnx_key'] = storage_key

# Metrics
final_metrics = result.best_metrics.to_dict() if result.best_metrics else {}
return True, artifacts, final_metrics, None
```

---

### Step 4 — Implement dataset download (presigned URL pattern)

The established pattern in this codebase is to include presigned URLs directly in the job payload (same as `onnx_presigned_url` in inference jobs).

#### 4a — Server: generate dataset presigned URL at job assignment time

**File:** `perceptra_hub/compute/services/agent_manager.py` — `assign_job_to_agent()`

After building `job_assignment`, generate a 1-hour presigned GET URL for the dataset zip:

```python
dataset_version = training_job.training_session.model_version.dataset_version
storage_profile = training_job.training_session.model_version.storage_profile

try:
    from storage.services import get_storage_adapter_for_profile
    adapter = get_storage_adapter_for_profile(storage_profile)
    dataset_key = f"datasets/{dataset_version.version_id}/data.zip"
    presigned = adapter.generate_presigned_url(dataset_key, expiration=3600, method='GET')
    job_assignment['dataset_presigned_url'] = presigned.url
    job_assignment['dataset_key'] = dataset_key
except Exception as e:
    logger.warning(f"Could not generate dataset presigned URL: {e}")
    job_assignment['dataset_presigned_url'] = None
```

#### 4b — Server: add `dataset_presigned_url` to `PollJobResponse` schema

**File:** `perceptra_hub/api/routers/agent/schemas.py`

```python
class PollJobResponse(BaseModel):
    ...existing fields...
    dataset_presigned_url: Optional[str] = None
    dataset_key: Optional[str] = None
```

#### 4c — Agent: implement `AgentClient.download_dataset()`

**File:** `perceptra_hub/agent/main.py`

```python
def download_dataset(self, dataset_version_id: str, storage_profile_id: str, output_path: Path, presigned_url: Optional[str] = None) -> bool:
    if not presigned_url:
        logger.error('No dataset presigned URL provided')
        return False
    try:
        logger.info(f'Downloading dataset from presigned URL...')
        resp = requests.get(presigned_url, stream=True, timeout=300)
        resp.raise_for_status()
        zip_path = output_path / 'dataset.zip'
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f'Dataset downloaded ({zip_path.stat().st_size} bytes), extracting...')
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_path)
        zip_path.unlink()
        logger.info(f'Dataset ready at: {output_path}')
        return True
    except Exception as e:
        logger.error(f'Dataset download failed: {e}')
        return False
```

Update the call site in `JobExecutor.execute()` to pass the presigned URL:
```python
self.client.download_dataset(
    dataset_version_id=job['dataset_version_id'],
    storage_profile_id=job['storage_profile_id'],
    output_path=dataset_dir,
    presigned_url=job.get('dataset_presigned_url')
)
```

---

### Step 5 — Implement artifact upload (presigned PUT URL)

Since artifact keys are determined at runtime (during training), the agent requests a presigned PUT URL per artifact via a new agent-authenticated endpoint.

#### 5a — Server: add artifact upload URL endpoint

**File:** `perceptra_hub/api/routers/agent/queries/agent.py`

Add a new endpoint (agent-key authenticated):

```python
@router.post("/artifacts/upload-url")
async def get_artifact_upload_url(
    request: ArtifactUploadUrlRequest,
    agent=Depends(get_agent_from_key)
):
    """Generate a presigned PUT URL for artifact upload."""
    @sync_to_async
    def _generate():
        from storage.models import StorageProfile
        profile = StorageProfile.objects.get(storage_profile_id=request.storage_profile_id)
        from storage.services import get_storage_adapter_for_profile
        adapter = get_storage_adapter_for_profile(profile)
        presigned = adapter.generate_presigned_url(request.key, expiration=1800, method='PUT')
        return presigned.url
    url = await _generate()
    return {"upload_url": url, "key": request.key}
```

**File:** `perceptra_hub/api/routers/agent/schemas.py` — add:
```python
class ArtifactUploadUrlRequest(BaseModel):
    storage_profile_id: str
    key: str
    content_type: str = 'application/octet-stream'
```

#### 5b — Agent: implement `AgentClient.upload_artifact()`

**File:** `perceptra_hub/agent/main.py`

```python
def upload_artifact(self, file_path: Path, storage_key: str, storage_profile_id: str) -> bool:
    try:
        # Get presigned PUT URL from platform
        resp = requests.post(
            f'{self.api_url}/api/v1/agents/artifacts/upload-url',
            headers=self.headers,
            json={'storage_profile_id': storage_profile_id, 'key': storage_key},
            timeout=15
        )
        resp.raise_for_status()
        upload_url = resp.json()['upload_url']

        # Upload file to presigned URL
        logger.info(f'Uploading artifact: {storage_key} ({file_path.stat().st_size} bytes)')
        with open(file_path, 'rb') as f:
            put_resp = requests.put(upload_url, data=f, timeout=600)
            put_resp.raise_for_status()
        logger.info(f'Artifact uploaded: {storage_key}')
        return True
    except Exception as e:
        logger.error(f'Artifact upload failed ({storage_key}): {e}')
        return False
```

---

## Critical Files

| File | Change |
|------|--------|
| `perceptra_hub/training/trainers/base.py` | Add `_progress_callback`, `set_progress_callback`, `_call_progress`; call in training loop |
| `perceptra_hub/training/trainers/yolo_trainer.py` | Register `on_train_epoch_end` Ultralytics callback in `train_epoch` |
| `perceptra_hub/agent/main.py` | Fix `result.get()` → attribute access; implement `download_dataset`; implement `upload_artifact` |
| `perceptra_hub/compute/services/agent_manager.py` | Generate `dataset_presigned_url` at job assignment |
| `perceptra_hub/api/routers/agent/schemas.py` | Add `dataset_presigned_url` to `PollJobResponse`; add `ArtifactUploadUrlRequest` |
| `perceptra_hub/api/routers/agent/queries/agent.py` | Add `POST /artifacts/upload-url` endpoint (agent-key auth) |

---

## Verification

1. Assign a training job with a valid `dataset_version_id` — confirm `dataset_presigned_url` is in the job payload.
2. Run the agent — confirm dataset streams and extracts cleanly to `DATASETS_DIR/<id>/`.
3. Confirm no `AttributeError` on `set_progress_callback`.
4. Confirm per-epoch progress appears in agent logs: `Progress reported: running (X%)`.
5. After training, confirm checkpoint and ONNX are uploaded: agent logs show `Artifact uploaded: organizations/.../checkpoint.pt`.
6. Confirm `complete_job` is called with `success=True` and `artifacts` dict populated.
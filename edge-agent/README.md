# Malumetrix edge agent

Keeps the models on a device in step with Malumetrix. You run inference your
way; the agent gets the right model file onto the device, tells your service
about it, checks it works, and goes back to the previous version if it
doesn't. It needs outgoing HTTPS only (no open ports, works behind NAT and
proxies), uses only the Python standard library (3.8+), and keeps working
while the network is down.

## How it works

Every 30 seconds or so (the platform decides) the agent sends one heartbeat:
what runs and how healthy the device is. The answer says what *should* run.
When the two differ, the agent:

1. downloads the file through a short-lived link, resuming after a cut;
2. checks its size and SHA-256;
3. switches `models/<model_id>/current` to it in one step;
4. hands it to your service (a command, an HTTP call, or a Python callback);
5. waits for your health check;
6. reports `running`, or switches back and reports `rolled_back`.

Nothing is pushed to the device, so a device that was offline for a day just
catches up. The two previous versions stay on disk, so going back needs no
network.

## Set up a device

1. In Malumetrix, **Fleet → Add device**. You get a one-time code, valid 24 h.
2. On the device:

   ```sh
   pip install ./edge-agent          # or build the Docker image, below
   malumetrix-edge --url https://malumetrix.example.com enroll --code K7QM-4XPA-9TWE
   malumetrix-edge --url https://malumetrix.example.com run
   ```

The device key is stored in `/var/lib/malumetrix/credentials.json` (mode 0600).
Revoking the device in Fleet stops the key at once; models keep running.

## Plug it into your inference

Pick whichever fits. They can be combined, and set per model.

### 1. Sidecar + files (any language, any runtime)

Your service loads the file at a fixed path and reloads when told:

```
/var/lib/malumetrix/models/<model_id>/current/model.onnx
/var/lib/malumetrix/models/<model_id>/current/manifest.json
/var/lib/malumetrix/models/<model_id>/current.json     # same, as a file, if symlinks aren't possible
```

```json
{
  "url": "https://malumetrix.example.com",
  "accepts": ["onnx", "pt"],
  "reload_command": "docker restart detector",
  "health_url": "http://127.0.0.1:8000/healthz",
  "health_timeout_s": 60
}
```

`reload_command` can be anything: `systemctl reload detector`,
`kill -HUP $(cat /run/detector.pid)`, a script. It gets
`MALUMETRIX_MODEL_PATH`, `MALUMETRIX_MANIFEST_PATH`, `MALUMETRIX_MODEL_ID` and
`MALUMETRIX_VERSION_ID` in its environment. Or set `notify_url` and the agent
POSTs `{model_id, version_id, model_path, manifest_path}` to your service.

### 2. Sidecar + local API (any language)

Your service asks the agent and reports to it on `127.0.0.1:7070`:

| Call | What for |
|---|---|
| `GET /v1/models/<model_id>` | the current file, manifest and version |
| `GET /v1/models` | every model on the device |
| `POST /v1/health` | `{"fps": 24.8, "latency_ms": 31, "errors": 0}`, sent with the next heartbeat |
| `GET /v1/status` | connection to the platform, last error, models |

### 3. SDK (inside your Python application)

```python
from malumetrix_edge import EdgeClient

def load(update):                       # called in the agent's thread
    detector.swap(update.model_path, classes=update.manifest["classes"])
    return detector.self_test()         # False or an exception → rolled back

edge = EdgeClient(url="https://malumetrix.example.com", code="K7QM-4XPA-9TWE",
                  on_model_update=load, accepts=["onnx"])
edge.start()
first = edge.wait_for_model("<model_id>")
...
edge.report_health(fps=24.8, latency_ms=31, errors=0)
```

### 4. Your own client (any language)

The protocol is three JSON calls; their OpenAPI schema is at
`/docs` on the platform (tag *Fleet*, paths `/api/v1/edge/*`).

```
POST /api/v1/edge/enroll      {"code": "K7QM-4XPA-9TWE", "accepts": ["onnx"]}
                              → {"device_id", "device_key", "poll_seconds"}
POST /api/v1/edge/heartbeat   X-Device-Key: <key>
                              {"health": {...}, "models": [{"model_id", "version_id", "status"}],
                               "state_token": "<from last answer>"}
                              → {"unchanged": true} or {"models": [{"model_id", "version_id",
                                 "artifact": {"url", "sha256", "size_bytes", "format"}, "manifest"}]}
POST /api/v1/edge/rotate-key  X-Device-Key: <key>  → {"device_key"}
```

Report `status: "rolled_back"` with `failed_version_id` when a version
doesn't work; the platform stops offering it to the device and pauses the
rollout.

## The manifest

Every version comes with `manifest.json`, so your code doesn't hard-code what
retraining can change:

```json
{"model_id": "…", "version_name": "v8", "task": "object-detection", "format": "onnx",
 "classes": ["pallet", "forklift", "person"], "input_size": 640, "metrics": {"mAP50": 0.71}}
```

## Docker

See the comments in `Dockerfile` for a sidecar next to your inference
container, sharing the model volume.

## All settings

JSON file (`--config` or `MALUMETRIX_CONFIG`), overridden by environment:

| Setting | Environment | Default |
|---|---|---|
| `url` | `MALUMETRIX_URL` | (required) |
| `data_dir` | `MALUMETRIX_DATA_DIR` | `/var/lib/malumetrix` |
| `accepts` | `MALUMETRIX_ACCEPTS` (comma separated) | `onnx,pt` |
| `reload_command` | `MALUMETRIX_RELOAD_COMMAND` | none |
| `notify_url` | `MALUMETRIX_NOTIFY_URL` | none |
| `health_url` | `MALUMETRIX_HEALTH_URL` | none |
| `health_command` | `MALUMETRIX_HEALTH_COMMAND` | none |
| `health_timeout_s` | `MALUMETRIX_HEALTH_TIMEOUT` | 60 |
| `local_api` | `MALUMETRIX_LOCAL_API` (`off` disables) | `127.0.0.1:7070` |
| `keep_versions` | `MALUMETRIX_KEEP_VERSIONS` | 3 (at least 2) |
| `ca_bundle` | `MALUMETRIX_CA_BUNDLE` | system CAs |
| `models.<model_id>.*` | | per-model hooks |

Anything that can reach the local API can read model paths and report
health, so keep it on localhost unless your service runs in another
container on a private network.

## Not yet

Image and prediction samples (for active learning from the field) come in a
later version; `POST /v1/samples` answers 501 until then.

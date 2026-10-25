# External storage & ingestion by reference

Perceptra is a plug-and-play CVOps platform. It can **own** its storage, or **connect to storage owned by
another system** and work on those images in place. External systems push image *references* — a storage
profile plus a key — with metadata, tags and annotations. Perceptra builds URLs from the profile. It never
downloads, copies or deletes these images, and it knows nothing about the producing system.

```
producer (any system)                         Perceptra
──────────────────────────                    ──────────────────────────────────────────────
writes images to its bucket ─────────────────▶ external storage profile (read-only adapter)
POST /api/v1/ingest/images ─────────────────▶ Image(storage_profile, storage_key, external_id)
  {storage_profile, storage_key, width,          + tags, metadata, ProjectImage, annotations
   height, metadata, tags, annotations}        browser ◀── short-lived URL from the profile
PUT  /storage/profiles/{id}/credentials ─────▶ credential rotated in place (profile id unchanged)
```

## Storage profiles

| `access_mode` | Meaning |
|---|---|
| `managed` (default) | Perceptra writes here: uploads, dataset exports, model artifacts. One managed profile is the default. |
| `external_readonly` | Files belong to another system and are only referenced. The adapter is wrapped in `perceptra_storage.ReadOnlyStorageAdapter`, so uploads and deletes raise. The profile can never be the default. Deleting an image only unlinks it. The storage router's upload and delete endpoints return 403. |

**Backends:** `azure`, `s3` (with an optional `endpoint_url` for S3-compatible stores), `minio`, `gcs`, `local`.
An organization may have any number of external profiles. Images are unique per `(storage_profile, storage_key)`.

**Credentials**, created with `POST /api/v1/storage/profiles-with-credentials`:

| Backend | Credential `type` | Browser URLs |
|---|---|---|
| Azure | `account_key`, `connection_string` | short-lived SAS minted per blob |
| Azure | `service_principal` (`tenant_id`, `client_id`, `client_secret`; roles *Storage Blob Data Reader* + *Storage Blob Delegator*) | short-lived **user-delegation** SAS |
| Azure | `sas_token` (a read-only container SAS, `sp=r` `sr=c`, is enough for external; `l` optional — the connection test falls back to reading a probe blob that cannot exist, where 404 proves access) | cannot mint → **signed Perceptra proxy** `/api/v1/media/images/{image_id}?exp&sig` (the token never reaches a browser) |
| S3 / MinIO | `access_key` (+ optional `session_token`) | presigned URL |
| GCS | `gcs_service_account` (`service_account_info` JSON) | V4 signed URL |

Credentials are stored Fernet-encrypted (`EncryptedSecret`, `SECRET_ENCRYPTION_KEY`) or referenced in Vault, Azure
Key Vault or AWS Secrets Manager. They are never logged. Adapters are cached per process for
`STORAGE_ADAPTER_CACHE_TTL_S` (300 s), keyed by the profile's last change.

### Rotation

```
PUT /api/v1/storage/profiles/{storage_profile_id}/credentials     (org admin / owner, or an admin API key)
{"credentials": {"type": "sas_token", "sas_token": "sv=...&sp=rl&sig=..."}, "test_before_save": true}
```

1. The new credential is tested against the profile's config.
2. It replaces the stored secret in place.
3. The adapter cache is dropped.
4. The profile id never changes, so every registered image keeps resolving.

The storage owner can automate rotation with an admin API key. In the SDK this is
`client.storage.rotate_credentials(profile_id, {...})`; in the console, the key button on a profile card.

### Signed media URLs

`storage/signing.py` signs `path|exp` with HMAC-SHA256 using `MEDIA_URL_SIGNING_KEY`, which falls back to
`SECRET_KEY`. Set a real key in production; rotating it invalidates outstanding links.

- **External proxy:** always verified.
- **Local files** (`/api/v1/media/files/...`): URLs are always signed, but the signature is only enforced when
  `MEDIA_REQUIRE_SIGNED_URLS=true`. Turn it on once stored URLs created before signing are gone; local URLs are
  signed for `LOCAL_MEDIA_URL_TTL_S`, 7 days by default.

## Ingestion by reference

```
POST /api/v1/ingest/images            (API key: scope ingest:write, write permission; edit access to the project)
{
  "project_id": "uuid",                      // optional; annotations need it
  "options": {"auto_assign_job": false, "batch_id": null, "create_missing_classes": false, "verify_exists": false},
  "items": [{
    "external_id": "sample-uuid",            // idempotency key
    "storage_profile_name": "data-hub-dropzone",   // or storage_profile_id
    "storage_key": "images/t1/2026/10/sample-uuid.jpg",
    "width": 1920, "height": 1080,           // required: images are never downloaded
    "checksum": "sha256…", "file_size": 123456, "captured_at": "…",
    "metadata": {...}, "tags": ["tenant:t1", "delivery:D-7"],
    "mark_as_null": false, "review_priority": 7,
    "annotations": [{
      "external_id": "alarm:evt-1", "class_name": "impurity",
      "bbox_xyxyn": [0.1, 0.1, 0.3, 0.3], "polygon_xyn": [[…]],
      "source": "verified" | "prediction", "confidence": 0.82, "model_version": "v3", "metadata": {...}
    }]
  }]
}
```

- **Batches:** up to 500 items. Each item has its own savepoint: a bad item is reported per item and the rest
  still lands. The response gives counts plus a per-item `created | updated | unchanged | error`.
- **Idempotency:** an image is matched by `(organization, external_id)`, else by `(storage_profile, storage_key)`.
  An annotation is matched by `(project_image, external_id)`. Re-pushing the same payload is `unchanged`.
- **Annotations:** `annotations` is the producer's *full* set for that image. Ingested annotations no longer
  pushed are withdrawn (soft-deleted). Omit `annotations` to leave them untouched.
- **Human work always wins:** an annotation a person has edited, reviewed or deleted since the last push is never
  overwritten or withdrawn. It is reported in `annotations.conflicts`.
- **Label provenance:**
  - `prediction` → `annotation_source="prediction"`: model output for review, audited as TP/FP when people edit it.
  - `verified` → `annotation_source="imported"` with `reviewed=True`: ground truth confirmed by the producer.
    Exported with the manual labels, used as ground truth by the audit, never counted as a false negative, and
    counted by the retraining policy.
- **Tags:** the producer's tags are synced; tags people added stay.
- **Job assignment:** off by default (`job_assignment_status="excluded"`), on with `auto_assign_job`.
- **Hard negatives:** `mark_as_null` marks an image with nothing to label.

```
POST /api/v1/ingest/images/remove     {"external_ids": [...], "project_id": "uuid"?}
```

This deactivates the project links (or every link, without `project_id`) and withdraws the unedited ingested
annotations. Storage is never touched.

SDK: `client.ingest.register_images(items, project_id=…)`, which batches automatically, and
`client.ingest.remove_images(ids, project_id=…)`.

## Implementation map

| Concern | Code |
|---|---|
| Read-only wrapper, `can_presign`, GCS, Azure service principal | `perceptra-storage` 0.2.0 (`readonly.py`, `adapters/gcs.py`, `adapters/azure.py`) |
| Access mode, default rule | `storage/models.py` (`StorageAccessMode`, `StorageProfile.is_external`) |
| Adapter cache, writable-profile selection | `storage/services.py` (`get_storage_adapter_for_profile`, `get_writable_storage_profile`) |
| Rotation | `api/routers/storage/queries/storage_rotation.py` |
| Signed proxy | `storage/signing.py`, `api/routers/images/queries/media.py` |
| Ingestion | `ingest/services.py`, `api/routers/ingest/queries/ingest.py` |
| Console | `perceptra-console/src/components/storage/` (`CredentialFields`, `StorageSettings`, `StorageProfileCard`) |

Tests: `DATABASE_ENGINE=django.db.backends.sqlite3 python manage.py test ingest storage`. A settings shim with
`CACHE_BACKEND={"backend": "memory"}` and migrations disabled works without Redis or Postgres.

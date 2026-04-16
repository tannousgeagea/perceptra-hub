# Similarity Feature — Senior Engineering Assessment

## Context

The similarity/duplicate detection feature has been built with a well-structured model layer, service functions, a Celery task, and 10 FastAPI endpoints. This assessment identifies what is broken, what will not scale, and what meaningful features are missing — prioritized by severity.

---

## Critical Bugs / Blockers

### 1. No Database Migrations
`similarity/migrations/` contains only `__init__.py`. All five models exist in code but not in the database.
- **Fix:** `python manage.py makemigrations similarity && python manage.py migrate`

### 2. Race Condition in Scan Creation
`POST /scans` checks for an active scan then creates one, but there is no lock between the two steps. Two concurrent requests can both pass the check and create duplicate scans.
- **Fix:** Wrap the check + create in `select_for_update()` inside `transaction.atomic()` in `create_scan()` (`services/similarity_scan.py:33`).

### 3. Separate Celery App Conflicts with Main App
`similarity/celery_app.py` instantiates its own `Celery` app with its own broker config (`similarity/config/celery_config.py`). The main API app (`api/config/celery_config.py`) also defines queues. These two apps will **not** share workers or queues unless explicitly unified.
- **Fix:** Register `similarity.tasks` on the existing main Celery app via `autodiscover_tasks`, or move the similarity queue config into the existing `api/config/celery_config.py`. Delete `similarity/celery_app.py` and `similarity/config/`.

### 4. Missing `services/__init__.py`
`similarity/services/` has no `__init__.py`. Imports like `from similarity.services.similarity_scan import create_scan` will fail in strict environments.
- **Fix:** Add an empty `__init__.py`.

---

## Scalability Issues

### 5. O(n²) Greedy Clustering
`_greedy_cluster()` (`tasks/similarity_scan.py:255`) iterates every new image against every existing cluster representative. At 10k images this is ~50M comparisons in pure Python — minutes of CPU.
- **Fix:** Replace with a **BK-tree** (exact for Hamming space). A BK-tree brings lookup from O(n) per image to O(log n), making 10k images feasible in seconds. The `pybktree` package is a drop-in.

### 6. Sequential Storage Downloads for Hash Computation
`_compute_hash_from_storage()` is called one image at a time inside a for-loop (`tasks/similarity_scan.py:108-168`). For 1k+ images over S3/GCS this is network-bound and slow.
- **Fix:** Use `concurrent.futures.ThreadPoolExecutor` (or `asyncio.gather` if the storage adapter is async) to download and hash in parallel batches (e.g. 20 concurrent).

### 7. `GET /images/{image_id}/similar` Loads All Org Hashes Into Memory
`similarity.py:705` fetches every `ImageHash` for the entire org into a Python list, then iterates in Python to compare distances. An org with 100k images will OOM the worker.
- **Fix:** Push the distance computation into the database. For PostgreSQL, store the hash as a `bigint` and use a bit-count expression, or use a partial index. As a pragmatic intermediate step, add a hard cap (`LIMIT 50000`) and document the limitation.

### 8. Cluster Member Serialization — No Pagination
`_serialize_cluster(..., include_members=True)` (`similarity.py:158`) fetches all members with no limit. A cluster of 5k near-duplicates serializes the full list to the HTTP response.
- **Fix:** Add `member_limit` / `member_offset` parameters to `GET /scans/{scan_id}/results/{cluster_id}`.

---

## Missing Features (High Value)

### 9. Incremental / Delta Scanning
Every scan re-hashes all images even when most already have a cached `ImageHash`. The task correctly skips re-hashing (`tasks/similarity_scan.py:120-134`), but the **clustering is always full** — it cannot add new images to an existing scan result.
- **Add:** A `base_scan_id` parameter to `POST /scans` that re-uses existing clusters and only compares newly-added images.

### 10. Re-Cluster Without Re-Hashing
Users often want to tighten or loosen the threshold after reviewing results. Today they must run a full new scan (re-downloading/hashing everything).
- **Add:** `POST /scans/{scan_id}/recluster?threshold=0.90` — reuse existing `ImageHash` rows, skip download/compute phase, re-run `_greedy_cluster()` only.

### 11. Webhook / Notification on Scan Completion
There is no signal when a scan finishes. Users must poll `GET /scans/{scan_id}`.
- **Add:** Fire a `scan.completed` / `scan.failed` event via the existing event API (`event_api/`) or an org-level webhook.

### 12. Export Endpoint
No way to export scan results for reporting or offline review.
- **Add:** `GET /scans/{scan_id}/export?format=csv|json` — returns cluster list with representative image keys and duplicate counts.

### 13. Similarity Histogram in Stats
`GET /stats` returns aggregate counts but no distribution. Users cannot tell if their threshold is too loose (thousands of 3-member clusters) or too tight (very few matches) without manually reviewing clusters.
- **Add:** Return a `similarity_distribution` histogram (10 buckets from 0.5–1.0) in `GET /stats` using a DB `CASE` expression.

---

## Code Quality / Minor Issues

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 14 | Silent hash failures — no counter | `tasks/similarity_scan.py:148` | Increment `failed_hashes` counter, include in scan summary |
| 15 | `PROGRESS_BATCH=50` hardcoded | `tasks/similarity_scan.py:40` | Move to Django settings |
| 16 | Cluster stats can drift | `models.py:301` | Call `refresh_member_count()` in `SimilarityCluster.save()` as a guard |
| 17 | `_delete_duplicates` deletes storage before DB transaction commits | `services/similarity_scan.py:212` | Reverse order: commit DB first, then storage cleanup as a best-effort post-commit hook |
| 18 | `max_retries=0` — scan failure is permanent | `tasks/similarity_scan.py:51` | Add `autoretry_for=(SoftTimeLimitExceeded,)` with `countdown=60, max_retries=2` |
| 19 | Bulk action has no all-or-nothing mode | `services/similarity_scan.py:127` | Add `atomic=True` flag that wraps the loop in `transaction.atomic()` |
| 20 | No tests | `similarity/tests.py` | Add unit tests for `_greedy_cluster`, `_similarity_to_hamming`, and `_set_representative` at minimum |

---

## Recommended Priority Order

1. **Migrations** (blocker — nothing works without this)
2. **Celery app unification** (blocker — tasks won't route to workers)
3. **`services/__init__.py`** (import error)
4. **Race condition fix** (data integrity)
5. **BK-tree clustering** (performance — blocks production use at scale)
6. **Parallel hash computation** (performance)
7. **Incremental scanning** (UX — eliminates full re-runs)
8. **Re-cluster endpoint** (UX — threshold tuning without cost)
9. **Cluster member pagination** (stability)
10. **Export + histogram + webhook** (completeness)
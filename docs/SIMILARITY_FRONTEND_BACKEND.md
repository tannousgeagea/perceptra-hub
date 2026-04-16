# Plan: Wire Similarity Frontend to Backend

## Context

The similarity module is fully implemented on the backend (12 API endpoints under `/api/v1/similarity`). The frontend has two hook layers:

1. **`useSimilarity.ts`** — production-ready hooks calling real API endpoints (not yet used by UI)
2. **`useScanHistory.ts` + `useSimilarityScan.ts` + `lib/similarity.ts`** — mock layer currently powering all UI

**Two wiring targets:**
- **ScanHistory page** (`/similarity/scans`): simple hook swap; types already match backend
- **DataLake similarity panel**: full rewrite; requires data adapters + minor backend additions

---

## Backend Changes

**File: `perceptra_hub/api/routers/similarity/queries/similarity.py`**

### 1. Add `download_url` to `_serialize_image_stub` (line ~217)
```python
def _serialize_image_stub(image) -> dict:
    if image is None:
        return {}
    return {
        ...existing fields...,
        "download_url": image.get_download_url(),  # ADD — Image model method exists
    }
```
`Image.get_download_url()` is already defined at `images/models.py:129`.

### 2. Add `include_members` to `get_scan_results` (line ~486)
- Add `include_members: bool = Query(False)` parameter
- Add `select_related("representative__storage_profile")` to the cluster queryset (needed for `download_url`)
- When `include_members=True`, add `prefetch_related("members__image__storage_profile", "members__image")`
- Pass `include_members` to `_serialize_cluster(c, include_members=include_members)`

---

## Frontend Changes

### File 1: `src/types/similarity.ts`
- Add `download_url: string` to `ImageStub` (line ~219)
- Add `members?: ClusterMember[]` to `ClusterSummary` (optional — only present when fetched with `include_members=true`)

### File 2: `src/hooks/useSimilarity.ts`
- Add optional `include_members?: boolean` to `getScanResults` and `useScanResults`
- Append `?include_members=true` to the query URL when set

### File 3: `src/pages/ScanHistory.tsx`
Replace import:
```typescript
// OLD
import { useScanList, useScanStats } from '@/hooks/useScanHistory';
// NEW
import { useScans, useSimilarityStats } from '@/hooks/useSimilarity';
```
Adapt call signatures:
- `useScanList({ skip, limit, status, scope, algorithm })` → `useScans({ status, scope, algorithm }, { skip, limit })`
- `useScanStats()` → `useSimilarityStats()`
- `data.scans`, `data.total`, `stats.scans.running` — field names unchanged ✓

### File 4: `src/components/similarity/ScanRow.tsx`
Replace import:
```typescript
// OLD
import { useLiveScan, useCancelScan } from '@/hooks/useScanHistory';
// NEW
import { useScan, useCancelScan } from '@/hooks/useSimilarity';
```
Replace `useLiveScan(initialScan.scan_id, initialScan.status)` with `useScan(initialScan.scan_id, { poll: isLive })`.

Add `useEffect` to show completion toast (old `useLiveScan` had this built-in):
```typescript
useEffect(() => {
  const s = liveScan?.status;
  if (!s || s === 'running' || s === 'pending') return;
  if (s === 'completed') toast.success(`Scan finished — ${liveScan.clusters_found} clusters found`);
  if (s === 'failed') toast.error('Scan failed — see error log');
  queryClient.invalidateQueries({ queryKey: ['similarity'] });
}, [liveScan?.status]);
```

`useCancelScan` mutation call `cancelMutation.mutate(scan.scan_id)` — unchanged ✓

### File 5: `src/hooks/useSimilarityScan.ts` — **Full Rewrite**

New implementation uses real API, maintains same return interface for `DataLake.tsx` (no component changes needed).

**Key adapters:**
```typescript
function adaptScanToJob(scan: ScanSummary): ScanJob {
  return {
    scan_id: scan.scan_id,
    status: scan.status,
    total_images: scan.total_images,
    images_processed: scan.hashed_images,
    clusters_found: scan.clusters_found,
    progress: scan.progress,
    eta_seconds: scan.eta_seconds ?? 0,
  };
}

function adaptCluster(c: ClusterSummary): SimilarityCluster {
  return {
    id: c.cluster_id,
    images: (c.members ?? []).map(adaptMember),
    avg_similarity: c.avg_similarity,
    status: c.status,
  };
}

function adaptMember(m: ClusterMember): SimilarityImage {
  return {
    id: m.image.image_id,
    filename: m.image.original_filename || m.image.name,
    url: m.image.download_url ?? '',
    thumbnail_url: m.image.download_url ?? '',
    width: m.image.width,
    height: m.image.height,
    file_size: m.image.file_size,
    upload_date: m.image.created_at,
    similarity_score: m.similarity_score,
    is_representative: m.role === 'representative',
    datasets: [],
  };
}
```

**New hook flow:**
1. `startScan(config)` → calls `createScan(orgId, { scope, algorithm, similarity_threshold: config.threshold, project_id: config.dataset_id })` → stores `scanId` in `useState`
2. `useScan(scanId, { poll: isActive })` polls via React Query — no more `setInterval`
3. `useEffect` on `liveScan.status` → when `completed`, call `getScanResults(orgId, scanId, {}, {}, true)` then `store.setScanResults(clusters.map(adaptCluster), ...)`
4. Action functions (`archiveDuplicates`, `deleteDuplicates`, `markReviewed`, `setRepresentative`, `bulkAction`) call real `performClusterAction` / `performBulkClusterAction`, then update store optimistically (remove/mark clusters)
5. `cancelScan()` calls real `cancelScan(orgId, scanId)` API

**Imports from `useSimilarity.ts`:** `createScan`, `getScan`, `getScanResults`, `performClusterAction`, `performBulkClusterAction`, `cancelScan`

### File 6 (Delete): `src/hooks/useScanHistory.ts`
No longer referenced after changes to `ScanHistory.tsx` and `ScanRow.tsx`.

### File 7 (Delete): `src/lib/similarity.ts`
No longer referenced after rewrite of `useSimilarityScan.ts`.

---

## Critical Files Summary

| File | Change |
|------|--------|
| `perceptra_hub/api/routers/similarity/queries/similarity.py` | Add `download_url`, `include_members` support |
| `src/types/similarity.ts` | Add `download_url` to `ImageStub`, `members?` to `ClusterSummary` |
| `src/hooks/useSimilarity.ts` | Add `include_members` param to results functions |
| `src/pages/ScanHistory.tsx` | Swap `useScanList`/`useScanStats` → `useScans`/`useSimilarityStats` |
| `src/components/similarity/ScanRow.tsx` | Swap `useLiveScan`/`useCancelScan` → `useScan`/`useCancelScan` |
| `src/hooks/useSimilarityScan.ts` | Full rewrite (adapters + real API calls) |
| `src/hooks/useScanHistory.ts` | **Delete** |
| `src/lib/similarity.ts` | **Delete** |

No changes to `DataLake.tsx`, `ScanResultsPanel.tsx`, `SimilarityClusterCard.tsx`, `ScanProgressView.tsx`, `ImageLightbox.tsx` — maintained via adapters.

---

## Verification

1. Navigate to `/similarity/scans` — scan list and stats load from real API (no mock delay)
2. Start a scan from DataLake → see real progress bar updating every 2s
3. Watch scan complete → real clusters load with image thumbnails from `download_url`
4. Archive/delete/mark-reviewed on a cluster → optimistic UI update
5. Bulk action on selected clusters → all update
6. Cancel a running scan → status shows cancelled
7. `ScanRow.tsx` live polling: a running scan row auto-updates without page refresh
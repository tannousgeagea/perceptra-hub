"""
End-to-end check of ingestion by reference through the SDK.

Registers images that already live in an external (read-only) storage profile,
re-registers them (must be "unchanged"), checks that the platform serves each
image through a working URL, and optionally withdraws them again.

    export PERCEPTRA_API_KEY=ph_live_...          # scope ingest:write, write permission
    python ingest_by_reference.py \
        --base-url https://hub.example.com \
        --project 11111111-2222-3333-4444-555555555555 \
        --profile data-hub-dropzone \
        --key images/t1/2026/10/<sample_id>.jpg [--width 1920 --height 1080] \
        [--annotate] [--cleanup]

    Without --width/--height the hub reads the size from the image header.

    # or take key, size and detections from a data_hub sample document
    python ingest_by_reference.py ... --sample-doc r1.json

Exit code 0 = every step passed.
"""
import argparse
import json
import os
import sys
from typing import List

import httpx

import perceptra

OK, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"


def step(title: str, ok: bool, detail: str = "") -> bool:
    print(f"[{OK if ok else FAIL}] {title}{(' — ' + detail) if detail else ''}")
    return ok


def items_from_args(args) -> List[dict]:
    if args.sample_doc:
        with open(args.sample_doc) as fh:
            doc = json.load(fh)
        image = doc["image"]
        annotations = [
            {
                "external_id": d["detection_id"],
                "class_name": args.class_name,
                "bbox_xyxyn": d["bbox_xyxyn"],
                "polygon_xyn": d.get("polygon_xyn"),
                "source": "verified" if d.get("label_hint") == "verified_positive" else "prediction",
                "confidence": d.get("confidence") if isinstance(d.get("confidence"), (int, float)) else None,
                "metadata": {"label_hint": d.get("label_hint"), "kind": d.get("kind")},
            }
            for d in doc.get("detections") or []
            if d.get("label_hint") not in ("verified_negative", "auto_negative")
        ]
        return [{
            "external_id": doc["sample_id"],
            "storage_profile_name": args.profile,
            "storage_key": image["key"],
            "width": image["width"],
            "height": image["height"],
            "checksum": image.get("sha256"),
            "metadata": {"source": doc.get("source"), "revision": doc.get("revision")},
            "tags": ["sdk-e2e", f"tenant:{(doc.get('source') or {}).get('tenant', {}).get('id')}"],
            "annotations": annotations,
        }]

    items = []
    for key in args.key:
        item = {
            "external_id": f"sdk-e2e:{key}",
            "storage_profile_name": args.profile,
            "storage_key": key,
            "metadata": {"origin": "ingest_by_reference.py"},
            "tags": ["sdk-e2e"],
        }
        if args.width and args.height:  # otherwise the hub reads the size from the image header
            item.update(width=args.width, height=args.height)
        if args.annotate:
            item["annotations"] = [{
                "external_id": "sdk-e2e-box-1",
                "class_name": args.class_name,
                "bbox_xyxyn": [0.25, 0.25, 0.5, 0.5],
                "source": "prediction",
                "confidence": 0.5,
            }]
        items.append(item)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("PERCEPTRA_BASE_URL"), required=not os.getenv("PERCEPTRA_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("PERCEPTRA_API_KEY"))
    parser.add_argument("--project", required=True, help="Project UUID")
    parser.add_argument("--profile", required=True, help="External storage profile name")
    parser.add_argument("--key", action="append", default=[], help="Blob key in that storage (repeatable)")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--sample-doc", help="data_hub sample document (samples/<tenant>/<id>/rN.json)")
    parser.add_argument("--annotate", action="store_true", help="Attach a test prediction box to --key images")
    parser.add_argument("--class-name", default="impurity")
    parser.add_argument("--create-missing-classes", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Withdraw the images at the end (storage untouched)")
    args = parser.parse_args()

    if not args.sample_doc and not args.key:
        parser.error("give --sample-doc, or one or more --key (with --width/--height, or the hub probes the size)")

    client = perceptra.Perceptra(api_key=args.api_key, base_url=args.base_url)
    all_ok = True

    # 1. The storage profile exists and is external (read-only).
    profiles = client._request("GET", "/storage/profiles", query={"limit": 1000}, cast_to=dict).get("profiles", [])
    profile = next((p for p in profiles if p["name"] == args.profile), None)
    all_ok &= step("storage profile exists", profile is not None, args.profile)
    if profile is None:
        return 1
    all_ok &= step("profile is external read-only", profile.get("access_mode") == "external_readonly",
                   f"access_mode={profile.get('access_mode')}")

    # 2. Register by reference.
    items = items_from_args(args)
    first = client.ingest.register_images(items, project_id=args.project,
                                          create_missing_classes=args.create_missing_classes)
    print(json.dumps(first, indent=2))
    all_ok &= step("register", first["errors"] == 0,
                   f"created={first['created']} updated={first['updated']} unchanged={first['unchanged']}")
    for r in first["results"]:
        ann = r.get("annotations") or {}
        if ann.get("errors"):
            all_ok &= step(f"annotations of {r['external_id']}", False, json.dumps(ann["errors"]))

    # 3. Idempotency: the same payload again must change nothing.
    second = client.ingest.register_images(items, project_id=args.project,
                                           create_missing_classes=args.create_missing_classes)
    all_ok &= step("re-register is idempotent", second["unchanged"] == len(items),
                   f"unchanged={second['unchanged']}/{len(items)}")

    # 4. The platform serves each image through a working URL (presigned or signed proxy).
    wanted = {r.get("image_id") for r in first["results"] if r.get("image_id")}
    found, skip = {}, 0
    while wanted - set(found) and skip < 5000:
        page = client._request("GET", f"/projects/{args.project}/images",
                               query={"skip": skip, "limit": 100}, cast_to=dict)
        rows = page.get("images") or page.get("items") or page.get("data") or []
        if not rows:
            break
        for row in rows:
            image = row.get("image") or row
            if image.get("image_id") in wanted:
                found[image["image_id"]] = image.get("download_url") or row.get("download_url")
        skip += 100
    for image_id in wanted:
        url = found.get(image_id)
        if not url:
            all_ok &= step(f"image {image_id} listed in project", False)
            continue
        absolute = url if url.startswith("http") else args.base_url.rstrip("/") + url
        try:
            resp = httpx.get(absolute, timeout=30, follow_redirects=True)
            ok = resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/")
            detail = f"HTTP {resp.status_code}, {resp.headers.get('content-type')}, {len(resp.content)} bytes"
        except httpx.HTTPError as exc:
            ok, detail = False, str(exc)
        kind = "proxy" if "/api/v1/media/images/" in url else "presigned"
        all_ok &= step(f"image {image_id} loads via {kind} URL", ok, detail)

    # 5. Optional cleanup (project link + ingested annotations; the blob stays).
    if args.cleanup:
        removed = client.ingest.remove_images([i["external_id"] for i in items], project_id=args.project)
        all_ok &= step("remove", all(r["status"] == "removed" for r in removed["results"]), json.dumps(removed["results"]))

    print("\nALL PASSED" if all_ok else "\nSOME STEPS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

# api/routers/suggestions/inference_client.py

import base64
import io
import logging
from typing import List, Tuple

import httpx
import numpy as np
from fastapi import HTTPException
from PIL import Image as PILImage

from .segmentation_service import SegmentationOutput

logger = logging.getLogger(__name__)


def _encode_image(image: np.ndarray) -> str:
    """numpy RGB array → base64 PNG string."""
    buf = io.BytesIO()
    PILImage.fromarray(image.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _norm_to_pixel_box(box: Tuple, w: int, h: int) -> List[int]:
    """Normalized (x, y, bw, bh) → pixel [x1, y1, x2, y2]."""
    x, y, bw, bh = box
    return [int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)]


def _norm_to_pixel_points(points, w: int, h: int) -> List[dict]:
    """[(x_norm, y_norm, label), ...] → [{"x": px, "y": py, "label": l}, ...]."""
    return [{"x": int(px * w), "y": int(py * h), "label": lbl} for px, py, lbl in points]


def _pixel_bbox_to_norm(bbox, w: int, h: int) -> Tuple[float, float, float, float]:
    """Pixel [x1, y1, x2, y2] → normalized (x, y, bw, bh)."""
    x1, y1, x2, y2 = bbox
    return (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)


def _response_to_output(resp: dict, w: int, h: int) -> SegmentationOutput:
    bbox_norm = _pixel_bbox_to_norm(resp["bbox"], w, h) if resp.get("bbox") else (0.0, 0.0, 0.0, 0.0)
    return SegmentationOutput(
        bbox=bbox_norm,
        mask_rle=resp.get("rle"),
        polygons=resp.get("polygons"),
        confidence=resp.get("score", 0.0),
    )


def _handle_error(e: Exception) -> None:
    if isinstance(e, httpx.TimeoutException):
        raise HTTPException(status_code=504, detail="Inference server timed out")
    if isinstance(e, httpx.ConnectError):
        raise HTTPException(
            status_code=503,
            detail="Inference server unreachable — check SEG_INFERENCE_URL",
        )
    raise HTTPException(status_code=502, detail=f"Inference service error: {e}")


class SegInferenceClient:
    """
    Async HTTP client for the perceptra-seg inference service.

    Converts normalized platform coordinates ↔ pixel coordinates for the service API.
    Implements the same async interface as MockSegmentationService.
    """

    def __init__(self, base_url: str, timeout_s: float = 10.0, api_key: str = "") -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            headers=headers,
        )

    async def health_check(self) -> dict:
        r = await self._client.get("/v1/healthz")
        r.raise_for_status()
        return r.json()

    async def segment_from_points(
        self, image: np.ndarray, points, model: str, device: str, precision: str
    ) -> SegmentationOutput:
        h, w = image.shape[:2]
        try:
            r = await self._client.post(
                "/v1/segment/points",
                params={"model": model},
                json={
                    "image": _encode_image(image),
                    "points": _norm_to_pixel_points(points, w, h),
                    "output_formats": ["rle", "polygons"],
                },
            )
            r.raise_for_status()
        except Exception as e:
            _handle_error(e)
        return _response_to_output(r.json(), w, h)

    async def segment_from_box(
        self, image: np.ndarray, box, model: str, device: str, precision: str
    ) -> SegmentationOutput:
        h, w = image.shape[:2]
        try:
            r = await self._client.post(
                "/v1/segment/box",
                params={"model": model},
                json={
                    "image": _encode_image(image),
                    "box": _norm_to_pixel_box(box, w, h),
                    "output_formats": ["rle", "polygons"],
                },
            )
            r.raise_for_status()
        except Exception as e:
            _handle_error(e)
        return _response_to_output(r.json(), w, h)

    async def segment_from_text(
        self, image: np.ndarray, text: str, model: str, device: str, precision: str
    ) -> List[SegmentationOutput]:
        h, w = image.shape[:2]
        try:
            r = await self._client.post(
                "/v1/segment/text",
                params={"model": model},
                json={
                    "image": _encode_image(image),
                    "text": text,
                    "output_formats": ["rle", "polygons"],
                },
            )
            r.raise_for_status()
        except Exception as e:
            _handle_error(e)
        return [_response_to_output(item, w, h) for item in r.json()]

    async def segment_from_exemplar(
        self, image: np.ndarray, exemplar_box, model: str, device: str, precision: str
    ) -> List[SegmentationOutput]:
        h, w = image.shape[:2]
        try:
            r = await self._client.post(
                "/v1/segment/exemplar",
                params={"model": model},
                json={
                    "image": _encode_image(image),
                    "exemplar_box": _norm_to_pixel_box(exemplar_box, w, h),
                    "output_formats": ["rle", "polygons"],
                },
            )
            r.raise_for_status()
        except Exception as e:
            _handle_error(e)
        return [_response_to_output(item, w, h) for item in r.json()]

    async def close(self) -> None:
        await self._client.aclose()


class SegInferenceClientSync:
    """
    Synchronous variant of SegInferenceClient for use inside Celery tasks.
    """

    def __init__(self, base_url: str, timeout_s: float = 120.0, api_key: str = "") -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers=headers,
        )

    def segment_auto(
        self, image: np.ndarray, config: dict, model: str, device: str, precision: str
    ) -> List[SegmentationOutput]:
        h, w = image.shape[:2]
        r = self._client.post(
            "/v1/segment/auto",
            params={"model": model},
            json={
                "image": _encode_image(image),
                "points_per_side": config.get("points_per_side", 32),
                "pred_iou_thresh": config.get("pred_iou_thresh", 0.88),
                "stability_score_thresh": config.get("stability_score_thresh", 0.95),
                "output_formats": ["rle", "polygons"],
            },
        )
        r.raise_for_status()
        return [_response_to_output(item, w, h) for item in r.json()]

    def close(self) -> None:
        self._client.close()

"""
On-Premise Inference Job Executor
Runs ONNX inference locally on agent GPUs, posts predictions back to the platform.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger('agent.inference')

WORK_DIR = Path('/tmp/agent-work/inference')


class InferenceJobExecutor:
    """Executes inference jobs dispatched from the platform."""

    def __init__(self, client):
        self.client = client

    def execute(self, job: Dict[str, Any]) -> tuple[bool, dict, Optional[str]]:
        """
        Download ONNX, run inference on all images, post results back.
        Returns (success, result_summary, error_message).
        """
        job_id = job['job_id']
        image_ids: List[int] = job.get('image_ids', [])
        confidence_threshold: float = job.get('confidence_threshold', 0.25)
        onnx_url: str = job.get('onnx_presigned_url', '')

        logger.info("Starting inference job %s: %d images", job_id, len(image_ids))

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        onnx_path = WORK_DIR / f"{job_id}.onnx"

        try:
            self._download_onnx(onnx_url, onnx_path)
            session, input_name = self._load_session(onnx_path)

            all_predictions = []
            for image_id in image_ids:
                try:
                    image_bytes = self._fetch_image(image_id)
                    if image_bytes is None:
                        continue
                    preds = self._run_inference(session, input_name, image_bytes, confidence_threshold)
                    all_predictions.append({'image_id': image_id, 'predictions': preds})
                except Exception as e:
                    logger.warning("Inference failed for image %s: %s", image_id, e)

            self._post_results(job_id, all_predictions)

            total = sum(len(p['predictions']) for p in all_predictions)
            logger.info("Inference job %s done: %d predictions across %d images",
                        job_id, total, len(all_predictions))
            return True, {'predictions_count': total}, None

        except Exception as e:
            error_msg = f"Inference job failed: {e}"
            logger.error(error_msg, exc_info=True)
            self._post_error(job_id, error_msg)
            return False, {}, error_msg
        finally:
            onnx_path.unlink(missing_ok=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _download_onnx(self, url: str, dest: Path) -> None:
        logger.info("Downloading ONNX model…")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("ONNX downloaded: %d bytes", dest.stat().st_size)

    def _load_session(self, onnx_path: Path):
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(onnx_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        input_name = session.get_inputs()[0].name
        logger.info("ONNX session loaded, input: %s", input_name)
        return session, input_name

    def _fetch_image(self, image_id: int) -> Optional[bytes]:
        try:
            resp = requests.get(
                f"{self.client.api_url}/api/v1/images/{image_id}/download",
                headers=self.client.headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning("Failed to fetch image %s: %s", image_id, e)
            return None

    def _run_inference(self, session, input_name: str, image_bytes: bytes, conf: float) -> List[dict]:
        import numpy as np
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Letterbox resize to 640x640
        size = 640
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]

        outputs = session.run(None, {input_name: blob})
        raw = outputs[0]

        # Handle both YOLOv8 (1, 4+cls, N) and RT-DETR (1, N, 4+cls) layouts
        if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
            raw = raw[0].T
        else:
            raw = raw[0]

        predictions = []
        for det in raw:
            cx, cy, bw, bh = det[:4]
            scores = det[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf:
                continue
            x1 = max(0.0, ((cx - bw / 2) - pad_x) / nw)
            y1 = max(0.0, ((cy - bh / 2) - pad_y) / nh)
            x2 = min(1.0, ((cx + bw / 2) - pad_x) / nw)
            y2 = min(1.0, ((cy + bh / 2) - pad_y) / nh)
            predictions.append({
                'class_id': class_id,
                'class_label': str(class_id),
                'confidence': round(confidence, 4),
                'bbox': [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            })

        return predictions

    def _post_results(self, job_id: str, predictions: List[dict]) -> None:
        resp = requests.post(
            f"{self.client.api_url}/api/v1/agents/inference-jobs/{job_id}/results",
            headers=self.client.headers,
            json={'predictions': predictions},
            timeout=60,
        )
        resp.raise_for_status()

    def _post_error(self, job_id: str, error: str) -> None:
        try:
            requests.post(
                f"{self.client.api_url}/api/v1/agents/inference-jobs/{job_id}/results",
                headers=self.client.headers,
                json={'predictions': [], 'error': error},
                timeout=30,
            )
        except Exception:
            pass

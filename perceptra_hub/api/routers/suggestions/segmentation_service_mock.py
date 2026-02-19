# api/routers/suggestions/mock_segmentation_service.py

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import random
from typing import Dict


# ============================================================
# Output Schema (Same as real service)
# ============================================================

@dataclass
class SegmentationOutput:
    """Normalized output for API layer."""
    bbox: Tuple[float, float, float, float]  # x, y, w, h normalized
    mask_rle: Optional[dict] = None
    polygons: Optional[List] = None
    confidence: float = 0.0

    def to_dict(self):
        return {
            'bbox': {
                'x': self.bbox[0],
                'y': self.bbox[1],
                'width': self.bbox[2],
                'height': self.bbox[3],
            },
            'mask_rle': self.mask_rle,
            'polygons': self.polygons,
            'confidence': self.confidence
        }


# ============================================================
# Mock Service
# ============================================================

class MockSegmentationService:
    """
    Lightweight mock segmentation service.
    No GPU, no model loading.
    Returns deterministic synthetic data.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    # ============================================================
    # Utilities
    # ============================================================

    def _random_bbox(self) -> Tuple[float, float, float, float]:
        """Generate random normalized bbox."""
        x = random.uniform(0.1, 0.6)
        y = random.uniform(0.1, 0.6)
        w = random.uniform(0.2, 0.4)
        h = random.uniform(0.2, 0.4)
        return (x, y, w, h)

    def _mock_mask(self):
        return {
            "size": [512, 512],
            "counts": "mock_rle_encoded_string"
        }

    def _mock_polygon(self, bbox):
        x, y, w, h = bbox
        return [[
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]]

    def _create_output(self) -> SegmentationOutput:
        bbox = self._random_bbox()
        return SegmentationOutput(
            bbox=bbox,
            mask_rle=self._mock_mask(),
            polygons=self._mock_polygon(bbox),
            confidence=round(random.uniform(0.75, 0.99), 3)
        )

    # ============================================================
    # Same Public API as Real Service
    # ============================================================

    def segment_from_points(
        self,
        image: np.ndarray,
        points: List[Tuple[float, float, int]],
        **kwargs
    ) -> SegmentationOutput:
        return self._create_output()

    def segment_from_box(
        self,
        image: np.ndarray,
        box: Tuple[float, float, float, float],
        **kwargs
    ) -> SegmentationOutput:
        return self._create_output()

    def segment_batch_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        **kwargs
    ) -> List[SegmentationOutput]:
        return [self._create_output() for _ in boxes]

    def segment_from_text(
        self,
        image: np.ndarray,
        text: str,
        **kwargs
    ) -> List[SegmentationOutput]:
        return [self._create_output() for _ in range(2)]

    def segment_from_exemplar(
        self,
        image: np.ndarray,
        exemplar_box: Tuple[float, float, float, float],
        **kwargs
    ) -> List[SegmentationOutput]:
        return [self._create_output() for _ in range(3)]

    def segment_text_and_box(
        self,
        image: np.ndarray,
        text: str,
        box: Tuple[float, float, float, float],
        **kwargs
    ) -> List[SegmentationOutput]:
        return [self._create_output() for _ in range(2)]

    def close(self):
        pass

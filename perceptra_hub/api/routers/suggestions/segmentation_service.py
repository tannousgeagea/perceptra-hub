# api/routers/suggestions/segmentation_service.py

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict
from threading import Lock

from perceptra_seg import Segmentor, SegmentorConfig
from perceptra_seg.models import SegmentationResult


@dataclass
class SegmentationOutput:
    """Normalized output for API layer."""
    bbox: Tuple[float, float, float, float]  # x, y, w, h normalized
    mask_rle: Optional[dict] = None
    polygons: Optional[List] = None
    confidence: float = 0.0

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            'bbox': {'x': self.bbox[0], 'y': self.bbox[1], 
                     'width': self.bbox[2], 'height': self.bbox[3]},
            'mask_rle': self.mask_rle,
            'polygons': self.polygons,
            'confidence': self.confidence
        }

class SegmentationService:
    """
    Wraps perceptra-seg Segmentor for platform integration.
    Singleton per worker — expensive to initialize.
    """
    
    _instances: Dict[str, "Segmentor"] = {}
    _lock = Lock()
    
    # def __init__(
    #     self,
    #     model: str = "sam_v2",
    #     device: str = "cuda",
    #     precision: str = "fp16"
    # ):
    #     config = SegmentorConfig()
    #     self.segmentor = Segmentor(
    #         config=config,
    #         model=model,
    #         device=device,
    #         precision=precision
    #     )
    #     self.segmentor.warmup()
    
    @classmethod
    def get_segmentor(
        cls, 
        model: str = "sam_v2",
        device: str = "cuda",
        precision: str = "fp16"
    ) -> "Segmentor":
        """Get or create segmentor instance."""
        key = f"{model}_{device}_{precision}"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    config = SegmentorConfig()
                    cls._instances[key] = Segmentor(
                        config=config,
                        model=model,
                        device=device,
                        precision=precision
                    )
                    cls._instances[key].warmup()
        
        return cls._instances[key]
    
    
    @classmethod
    def get_instance(cls, **kwargs) -> "SegmentationService":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    def _normalize_bbox(
        self, 
        bbox: Tuple[int, int, int, int], 
        img_w: int, 
        img_h: int
    ) -> Tuple[float, float, float, float]:
        """Convert pixel bbox to normalized (x, y, w, h)."""
        x1, y1, x2, y2 = bbox
        return (x1/img_w, y1/img_h, (x2-x1)/img_w, (y2-y1)/img_h)
    
    def _to_output(
        self, 
        result: SegmentationResult, 
        img_w: int, 
        img_h: int
    ) -> SegmentationOutput:
        """Convert SegmentationResult to API output."""
        bbox_norm = self._normalize_bbox(result.bbox, img_w, img_h) if result.bbox else (0,0,0,0)
        return SegmentationOutput(
            bbox=bbox_norm,
            mask_rle=result.rle,
            polygons=result.polygons,
            confidence=result.score
        )
    
    # ============================================================
    # Core Methods
    # ============================================================
    
    def segment_from_points(
        self,
        image: np.ndarray,
        points: List[Tuple[float, float, int]],  # normalized coords + label
        model: str = "sam_v2",
        device: str = "cuda",
        precision: str = "fp16"
    ) -> SegmentationOutput:
        """Single point-prompt segmentation."""
        segmentor = self.get_segmentor(model, device, precision)
        h, w = image.shape[:2]
        
        # Convert normalized to pixel coords
        pixel_points = [(int(x*w), int(y*h), label) for x, y, label in points]
        
        result = segmentor.segment_from_points(
            image, 
            pixel_points,
            output_formats=["rle", "polygons"]
        )
        
        return self._to_output(result, w, h)
    
    def segment_from_box(
        self,
        image: np.ndarray,
        box: Tuple[float, float, float, float],  # normalized x, y, w, h
        model: str = "sam_v2",
        device: str = "cuda",
        precision: str = "fp16",
    ) -> SegmentationOutput:
        """Single box-prompt segmentation."""
        segmentor = self.get_segmentor(model, device, precision)
        h, w = image.shape[:2]
        
        # Convert to pixel coords (x1, y1, x2, y2)
        x, y, bw, bh = box
        pixel_box = (int(x*w), int(y*h), int((x+bw)*w), int((y+bh)*h))
        
        result = segmentor.segment_from_box(
            image,
            pixel_box,
            output_formats=["rle", "polygons"]
        )
        
        return self._to_output(result, w, h)
    
    def segment_batch_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        model: str = "sam_v2",
        device: str = "cuda",
        precision: str = "fp16"
    ) -> List[SegmentationOutput]:
        """Batch box segmentation — efficient for auto-segment."""
        segmentor = self.get_segmentor(model, device, precision)
        h, w = image.shape[:2]
        
        pixel_boxes = [
            (int(x*w), int(y*h), int((x+bw)*w), int((y+bh)*h))
            for x, y, bw, bh in boxes
        ]
        
        results = segmentor.segment_batch(
            image,
            boxes=pixel_boxes,
            output_formats=["rle", "polygons"]
        )
        
        return [self._to_output(r, w, h) for r in results]
    
    # ============================================================
    # SAM3 Semantic Methods
    # ============================================================
    
    def segment_from_text(
        self,
        image: np.ndarray,
        text: str,
        model: str = "sam_v3",
        device: str = "cuda",
        precision: str = "fp16"
    ) -> List[SegmentationOutput]:
        """Text prompt segmentation (SAM3 only)."""
        segmentor = self.get_segmentor(model, device, precision)
        h, w = image.shape[:2]
        
        results = segmentor.segment_from_text(
            image,
            text,
            output_formats=["rle", "polygons"]
        )
        
        return [self._to_output(r, w, h) for r in results]
    
    def segment_from_exemplar(
        self,
        image: np.ndarray,
        exemplar_box: Tuple[float, float, float, float],
        model: str = "sam_v3",
        device: str = "cuda",
        precision: str = "fp16"
    ) -> List[SegmentationOutput]:
        """Find similar objects to exemplar (SAM3 only)."""
        segmentor = self.get_segmentor(model, device, precision)
        h, w = image.shape[:2]
        
        x, y, bw, bh = exemplar_box
        pixel_box = (int(x*w), int(y*h), int((x+bw)*w), int((y+bh)*h))
        
        results = segmentor.segment_from_exemplar_box(
            image,
            pixel_box,
            output_formats=["rle", "polygons"]
        )
        
        return [self._to_output(r, w, h) for r in results]
    
    def segment_text_and_box(
        self,
        image: np.ndarray,
        text: str,
        box: Tuple[float, float, float, float],
        model: str = "sam_v2",
        device: str = "cuda",
        precision: str = "fp16"
    ) -> List[SegmentationOutput]:
        """Combined text + box prompt (SAM3 only)."""
        segmentor = self.get_segmentor(model, device, precision)
        h, w = image.shape[:2]
        
        x, y, bw, bh = box
        pixel_box = (int(x*w), int(y*h), int((x+bw)*w), int((y+bh)*h))
        
        results = segmentor.segment_from_text_and_box(
            image,
            text,
            pixel_box,
            output_formats=["rle", "polygons"]
        )
        
        return [self._to_output(r, w, h) for r in results]
    
    def close(self):
        self.segmentor.close()
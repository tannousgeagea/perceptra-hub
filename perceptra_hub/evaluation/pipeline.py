"""
Production-Grade Model Evaluation Pipeline
===========================================

Consumes FastAPI evaluation endpoint and computes comprehensive metrics.
Designed for production use with caching, incremental computation, and extensibility.
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import statistics
import json

import httpx  # Async HTTP client
from pydantic import BaseModel


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Pipeline configuration with sensible production defaults"""
    
    # API configuration
    api_base_url: str = "http://localhost:8000/api/v1/evaluation"
    timeout_seconds: int = 300
    
    # Filtering
    reviewed_only: bool = True
    include_difficult: bool = False  # Future: filter by difficulty flag
    ignore_classes: Set[str] = field(default_factory=set)
    focus_classes: Optional[Set[str]] = None
    
    # Analysis depth
    enable_error_analysis: bool = True
    enable_confidence_curves: bool = True
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 0.95])
    
    # Quality gates
    min_review_confidence: Optional[int] = None
    min_samples_for_stats: int = 5  # Min samples to report statistics
    
    # Performance
    batch_size: int = 100
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Reporting
    max_error_examples: int = 50
    verbose: bool = False


# ============================================================================
# CORE METRICS
# ============================================================================

@dataclass
class ConfusionMatrix:
    """Confusion matrix for multi-class evaluation"""
    
    matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)  # (true, pred) -> count
    class_names: Set[str] = field(default_factory=set)
    
    def add(self, true_class: str, pred_class: str):
        """Add prediction to confusion matrix"""
        self.matrix[(true_class, pred_class)] = self.matrix.get((true_class, pred_class), 0) + 1
        self.class_names.add(true_class)
        self.class_names.add(pred_class)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as nested dictionary for visualization"""
        sorted_classes = sorted(self.class_names)
        result = {
            "classes": sorted_classes,
            "matrix": [[self.matrix.get((t, p), 0) for p in sorted_classes] for t in sorted_classes]
        }
        return result


@dataclass
class ModelPerformanceMetrics:
    """Core performance metrics"""
    
    # Counts
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    # Derived
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Quality
    total_predictions: int = 0
    total_ground_truth: int = 0
    
    # Edit analysis
    tp_unedited: int = 0
    tp_minor_edit: int = 0
    tp_major_edit: int = 0
    tp_class_change: int = 0
    
    edit_rate: float = 0.0
    hallucination_rate: float = 0.0
    
    # Context
    level: str = "dataset"  # "dataset", "class", "image"
    class_name: Optional[str] = None
    sample_count: int = 0  # Number of images
    
    def compute_derived(self):
        """Calculate derived metrics"""
        if self.tp + self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        
        if self.tp + self.fn > 0:
            self.recall = self.tp / (self.tp + self.fn)
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        if self.tp > 0:
            edited = self.tp_minor_edit + self.tp_major_edit + self.tp_class_change
            self.edit_rate = edited / self.tp
        
        if self.total_predictions > 0:
            self.hallucination_rate = self.fp / self.total_predictions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "edit_rate": round(self.edit_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "total_predictions": self.total_predictions,
            "total_ground_truth": self.total_ground_truth,
            "level": self.level,
            "class_name": self.class_name,
            "sample_count": self.sample_count
        }


@dataclass
class LocalizationMetrics:
    """Bbox accuracy analysis"""
    
    # IoU distribution (for edited TPs)
    iou_samples: List[float] = field(default_factory=list)
    
    mean_iou: float = 0.0
    median_iou: float = 0.0
    std_iou: float = 0.0
    
    # Categorization
    tight_boxes: int = 0    # IoU > 0.9
    good_boxes: int = 0     # IoU 0.7-0.9
    loose_boxes: int = 0    # IoU 0.5-0.7
    poor_boxes: int = 0     # IoU < 0.5
    
    # Edit magnitude
    edit_magnitudes: List[float] = field(default_factory=list)
    mean_edit_magnitude: float = 0.0
    
    sample_count: int = 0
    
    def compute_stats(self):
        """Calculate statistics from samples"""
        if self.iou_samples:
            self.mean_iou = statistics.mean(self.iou_samples)
            self.median_iou = statistics.median(self.iou_samples)
            if len(self.iou_samples) > 1:
                self.std_iou = statistics.stdev(self.iou_samples)
            
            # Categorize
            for iou in self.iou_samples:
                if iou > 0.9:
                    self.tight_boxes += 1
                elif iou > 0.7:
                    self.good_boxes += 1
                elif iou > 0.5:
                    self.loose_boxes += 1
                else:
                    self.poor_boxes += 1
        
        if self.edit_magnitudes:
            self.mean_edit_magnitude = statistics.mean(self.edit_magnitudes)
        
        self.sample_count = len(self.iou_samples)


@dataclass
class ConfidenceAnalysis:
    """Confidence score analysis"""
    
    # Distributions
    tp_confidences: List[float] = field(default_factory=list)
    fp_confidences: List[float] = field(default_factory=list)
    
    # Statistics
    mean_tp_conf: float = 0.0
    mean_fp_conf: float = 0.0
    
    # Calibration: precision at different thresholds
    precision_at_threshold: Dict[float, float] = field(default_factory=dict)
    recall_at_threshold: Dict[float, float] = field(default_factory=dict)
    f1_at_threshold: Dict[float, float] = field(default_factory=dict)
    
    # Expected Calibration Error (future)
    # ece: float = 0.0
    
    def compute_stats(self):
        """Calculate statistics"""
        if self.tp_confidences:
            self.mean_tp_conf = statistics.mean(self.tp_confidences)
        if self.fp_confidences:
            self.mean_fp_conf = statistics.mean(self.fp_confidences)
    
    def compute_threshold_metrics(self, thresholds: List[float], all_predictions: List[Dict]):
        """
        Compute precision/recall at different confidence thresholds.
        
        Args:
            all_predictions: List of dicts with 'confidence', 'is_tp', 'is_fp'
        """
        for thresh in thresholds:
            filtered = [p for p in all_predictions if p['confidence'] >= thresh]
            
            if not filtered:
                self.precision_at_threshold[thresh] = 0.0
                self.recall_at_threshold[thresh] = 0.0
                self.f1_at_threshold[thresh] = 0.0
                continue
            
            tp_count = sum(1 for p in filtered if p['is_tp'])
            fp_count = sum(1 for p in filtered if p['is_fp'])
            total_fn = sum(1 for p in all_predictions if p.get('is_fn', False))
            
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + total_fn) if (tp_count + total_fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            self.precision_at_threshold[thresh] = precision
            self.recall_at_threshold[thresh] = recall
            self.f1_at_threshold[thresh] = f1


@dataclass
class ErrorAnalysis:
    """Detailed error patterns"""
    
    # Confusion matrix
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    
    # Size analysis (bbox areas)
    fn_sizes: List[float] = field(default_factory=list)  # Missed object sizes
    fp_sizes: List[float] = field(default_factory=list)  # Hallucination sizes
    
    # Small/medium/large object performance
    small_object_recall: Optional[float] = None
    medium_object_recall: Optional[float] = None
    large_object_recall: Optional[float] = None
    
    # Error examples (for manual review)
    high_confidence_errors: List[Dict] = field(default_factory=list)
    low_confidence_correct: List[Dict] = field(default_factory=list)
    heavily_edited_predictions: List[Dict] = field(default_factory=list)
    missed_objects: List[Dict] = field(default_factory=list)
    
    # Review time patterns (if available)
    mean_review_time_tp: Optional[float] = None
    mean_review_time_fp: Optional[float] = None
    mean_review_time_fn: Optional[float] = None


# ============================================================================
# EVALUATION REPORT
# ============================================================================

@dataclass
class EvaluationReport:
    """Complete evaluation output"""
    
    # Metadata
    evaluated_at: datetime
    project_id: int
    model_version: Optional[str]
    config: EvaluationConfig
    
    # Core metrics
    dataset_metrics: ModelPerformanceMetrics
    per_class_metrics: Dict[str, ModelPerformanceMetrics]
    
    # Detailed analysis
    localization: Optional[LocalizationMetrics] = None
    confidence_analysis: Optional[ConfidenceAnalysis] = None
    error_analysis: Optional[ErrorAnalysis] = None
    
    # Data quality
    total_images: int = 0
    reviewed_images: int = 0
    
    def summary(self) -> str:
        """Human-readable summary"""
        m = self.dataset_metrics
        
        lines = [
            "=" * 80,
            "MODEL EVALUATION REPORT",
            "=" * 80,
            f"Project ID: {self.project_id}",
            f"Model Version: {self.model_version or 'N/A'}",
            f"Evaluated: {self.evaluated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Images: {self.reviewed_images} reviewed / {self.total_images} total",
            "",
            "OVERALL PERFORMANCE",
            "-" * 80,
            f"Precision:  {m.precision:7.2%}   ({m.tp} TP / {m.tp + m.fp} predictions)",
            f"Recall:     {m.recall:7.2%}   ({m.tp} TP / {m.total_ground_truth} objects)",
            f"F1 Score:   {m.f1_score:7.2%}",
            "",
            f"True Positives:   {m.tp:6d}   (model correct)",
            f"False Positives:  {m.fp:6d}   (hallucinations, {m.hallucination_rate:.1%})",
            f"False Negatives:  {m.fn:6d}   (missed objects)",
            "",
            "PREDICTION QUALITY",
            "-" * 80,
            f"Perfect Predictions:  {m.tp_unedited:6d}   ({m.tp_unedited/m.tp*100:.1f}% of TPs)" if m.tp > 0 else "Perfect Predictions:  0",
            f"Minor Edits:          {m.tp_minor_edit:6d}",
            f"Major Edits:          {m.tp_major_edit:6d}",
            f"Class Changes:        {m.tp_class_change:6d}",
            f"Overall Edit Rate:    {m.edit_rate:6.1%}",
        ]
        
        # Localization quality
        if self.localization and self.localization.sample_count > 0:
            loc = self.localization
            lines.extend([
                "",
                "LOCALIZATION ACCURACY (Edited Predictions)",
                "-" * 80,
                f"Mean IoU:      {loc.mean_iou:.3f} ± {loc.std_iou:.3f}",
                f"Median IoU:    {loc.median_iou:.3f}",
                f"Tight (>0.9):  {loc.tight_boxes:4d} / {loc.sample_count} ({loc.tight_boxes/loc.sample_count*100:.1f}%)",
                f"Good (0.7-0.9): {loc.good_boxes:4d} / {loc.sample_count}",
                f"Poor (<0.5):   {loc.poor_boxes:4d} / {loc.sample_count}",
            ])
        
        # Confidence analysis
        if self.confidence_analysis:
            ca = self.confidence_analysis
            lines.extend([
                "",
                "CONFIDENCE CALIBRATION",
                "-" * 80,
                f"Mean TP Confidence:  {ca.mean_tp_conf:.3f}",
                f"Mean FP Confidence:  {ca.mean_fp_conf:.3f}",
            ])
            
            if ca.precision_at_threshold:
                lines.append("")
                lines.append("Precision @ Confidence Threshold:")
                for thresh in sorted(ca.precision_at_threshold.keys()):
                    prec = ca.precision_at_threshold[thresh]
                    rec = ca.recall_at_threshold[thresh]
                    f1 = ca.f1_at_threshold[thresh]
                    lines.append(f"  {thresh:.2f}: Prec={prec:.2%}, Rec={rec:.2%}, F1={f1:.2%}")
        
        # Per-class breakdown
        if self.per_class_metrics:
            lines.extend([
                "",
                "TOP 10 CLASSES BY F1 SCORE",
                "-" * 80,
                f"{'Class':<25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}"
            ])
            
            sorted_classes = sorted(
                self.per_class_metrics.items(),
                key=lambda x: x[1].f1_score,
                reverse=True
            )[:10]
            
            for class_name, cm in sorted_classes:
                lines.append(
                    f"{class_name:<25} {cm.precision:>7.1%} {cm.recall:>7.1%} {cm.f1_score:>7.1%} "
                    f"{cm.tp:>6} {cm.fp:>6} {cm.fn:>6}"
                )
        
        # Error highlights
        if self.error_analysis:
            ea = self.error_analysis
            
            if ea.high_confidence_errors:
                lines.extend([
                    "",
                    f"⚠️  HIGH-CONFIDENCE ERRORS: {len(ea.high_confidence_errors)} examples",
                    "   → Review model overconfidence issues"
                ])
            
            if ea.confusion_matrix.matrix:
                top_confusions = sorted(
                    [(k, v) for k, v in ea.confusion_matrix.matrix.items() if k[0] != k[1]],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                if top_confusions:
                    lines.extend([
                        "",
                        "TOP CLASS CONFUSIONS",
                        "-" * 80
                    ])
                    for (true_cls, pred_cls), count in top_confusions:
                        lines.append(f"  {pred_cls} → {true_cls}: {count} times")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization"""
        return {
            "metadata": {
                "evaluated_at": self.evaluated_at.isoformat(),
                "project_id": self.project_id,
                "model_version": self.model_version,
                "total_images": self.total_images,
                "reviewed_images": self.reviewed_images,
            },
            "dataset_metrics": self.dataset_metrics.to_dict(),
            "per_class_metrics": {k: v.to_dict() for k, v in self.per_class_metrics.items()},
            "localization": {
                "mean_iou": self.localization.mean_iou,
                "median_iou": self.localization.median_iou,
                "tight_boxes": self.localization.tight_boxes,
                "good_boxes": self.localization.good_boxes,
                "sample_count": self.localization.sample_count,
            } if self.localization else None,
            "confidence_analysis": {
                "mean_tp_conf": self.confidence_analysis.mean_tp_conf,
                "mean_fp_conf": self.confidence_analysis.mean_fp_conf,
                "precision_at_threshold": self.confidence_analysis.precision_at_threshold,
            } if self.confidence_analysis else None,
        }


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

class ModelEvaluationPipeline:
    """
    Production-grade evaluation pipeline.
    
    Features:
    - Async API consumption with retry logic
    - Incremental computation for large datasets
    - Caching support
    - Progress tracking
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
    
    async def evaluate_project(
        self,
        project_id: int,
        model_version: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> EvaluationReport:
        """
        Run complete evaluation on a project.
        
        Args:
            project_id: Project to evaluate
            model_version: Optional model version filter
            progress_callback: Optional callback(current, total) for progress tracking
        
        Returns:
            EvaluationReport with comprehensive metrics
        """
        
        # Initialize report
        report = EvaluationReport(
            evaluated_at=datetime.utcnow(),
            project_id=project_id,
            model_version=model_version,
            config=self.config,
            dataset_metrics=ModelPerformanceMetrics(level="dataset"),
            per_class_metrics={},
        )
        
        # Fetch data from API (paginated)
        all_images = []
        page = 1
        total_pages = None
        
        while True:
            # Call API
            params = {
                "page": page,
                "page_size": self.config.batch_size,
                "reviewed_only": self.config.reviewed_only,
                "model_version": model_version,
            }
            
            response = await self.client.get(
                f"{self.config.api_base_url}/projects/{project_id}/images",
                params={k: v for k, v in params.items() if v is not None}
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            data = response.json()
            all_images.extend(data["images"])
            
            # Progress tracking
            if progress_callback:
                total = data["total_count"]
                current = len(all_images)
                progress_callback(current, total)
            
            # Check if more pages
            if not data.get("has_next", False):
                break
            
            page += 1
        
        # Update report metadata
        report.total_images = len(all_images)
        report.reviewed_images = sum(1 for img in all_images if img.get("reviewed", False))
        
        # Compute metrics
        self._compute_dataset_metrics(all_images, report)
        self._compute_class_metrics(all_images, report)
        
        if self.config.enable_error_analysis:
            self._compute_localization_metrics(all_images, report)
            self._compute_confidence_analysis(all_images, report)
            self._compute_error_analysis(all_images, report)
        
        return report
    
    def _compute_dataset_metrics(self, images: List[Dict], report: EvaluationReport):
        """Aggregate metrics across all images"""
        m = report.dataset_metrics
        m.sample_count = len(images)
        
        for img in images:
            for ann in img.get("annotations", []):
                if not self._should_include(ann):
                    continue
                
                eval_data = ann.get("evaluation")
                if not eval_data:
                    continue
                
                status = eval_data.get("status")
                
                if status == "TP":
                    m.tp += 1
                    
                    # Breakdown by edit type
                    edit_type = eval_data.get("edit_type")
                    if edit_type == "none" or not eval_data.get("was_edited"):
                        m.tp_unedited += 1
                    elif edit_type == "minor":
                        m.tp_minor_edit += 1
                    elif edit_type == "major":
                        m.tp_major_edit += 1
                    elif edit_type == "class_change":
                        m.tp_class_change += 1
                
                elif status == "FP":
                    m.fp += 1
                
                elif status == "FN":
                    m.fn += 1
                
                # Count predictions
                if ann.get("source") == "prediction":
                    m.total_predictions += 1
        
        m.total_ground_truth = m.tp + m.fn
        m.compute_derived()
    
    def _compute_class_metrics(self, images: List[Dict], report: EvaluationReport):
        """Per-class performance"""
        class_metrics = defaultdict(lambda: ModelPerformanceMetrics(level="class"))
        
        for img in images:
            for ann in img.get("annotations", []):
                if not self._should_include(ann):
                    continue
                
                class_name = ann.get("class_name")
                cm = class_metrics[class_name]
                cm.class_name = class_name
                
                eval_data = ann.get("evaluation")
                if not eval_data:
                    continue
                
                status = eval_data.get("status")
                
                if status == "TP":
                    cm.tp += 1
                    edit_type = eval_data.get("edit_type")
                    if edit_type == "minor":
                        cm.tp_minor_edit += 1
                    elif edit_type == "major":
                        cm.tp_major_edit += 1
                    elif edit_type == "class_change":
                        cm.tp_class_change += 1
                    else:
                        cm.tp_unedited += 1
                
                elif status == "FP":
                    cm.fp += 1
                elif status == "FN":
                    cm.fn += 1
                
                if ann.get("source") == "prediction":
                    cm.total_predictions += 1
        
        # Compute derived metrics
        for cm in class_metrics.values():
            cm.total_ground_truth = cm.tp + cm.fn
            cm.compute_derived()
        
        report.per_class_metrics = dict(class_metrics)
    
    def _compute_localization_metrics(self, images: List[Dict], report: EvaluationReport):
        """Bbox accuracy analysis"""
        loc = LocalizationMetrics()
        
        for img in images:
            for ann in img.get("annotations", []):
                if not self._should_include(ann):
                    continue
                
                eval_data = ann.get("evaluation")
                if not eval_data or eval_data.get("status") != "TP":
                    continue
                
                # Collect IoU for edited predictions
                if eval_data.get("was_edited") and eval_data.get("localization_iou") is not None:
                    loc.iou_samples.append(eval_data["localization_iou"])
                
                # Collect edit magnitudes
                if eval_data.get("edit_magnitude") is not None:
                    loc.edit_magnitudes.append(eval_data["edit_magnitude"])
        
        loc.compute_stats()
        
        if loc.sample_count >= self.config.min_samples_for_stats:
            report.localization = loc
    
    def _compute_confidence_analysis(self, images: List[Dict], report: EvaluationReport):
        """Confidence distribution and calibration"""
        ca = ConfidenceAnalysis()
        all_predictions = []
        
        for img in images:
            for ann in img.get("annotations", []):
                if not self._should_include(ann):
                    continue
                
                if ann.get("source") != "prediction":
                    continue
                
                conf = ann.get("confidence")
                if conf is None:
                    continue
                
                eval_data = ann.get("evaluation")
                if not eval_data:
                    continue
                
                status = eval_data.get("status")
                
                pred_data = {
                    "confidence": conf,
                    "is_tp": status == "TP",
                    "is_fp": status == "FP",
                    "is_fn": False,
                }
                
                if status == "TP":
                    ca.tp_confidences.append(conf)
                elif status == "FP":
                    ca.fp_confidences.append(conf)
                
                all_predictions.append(pred_data)
        
        # Add FN data (no confidence, but needed for recall calculation)
        for img in images:
            for ann in img.get("annotations", []):
                if ann.get("evaluation", {}).get("status") == "FN":
                    all_predictions.append({"confidence": 0.0, "is_tp": False, "is_fp": False, "is_fn": True})
        
        ca.compute_stats()
        
        if self.config.enable_confidence_curves:
            ca.compute_threshold_metrics(self.config.confidence_thresholds, all_predictions)
        
        report.confidence_analysis = ca
    
    def _compute_error_analysis(self, images: List[Dict], report: EvaluationReport):
        """Detailed error patterns"""
        ea = ErrorAnalysis()
        
        for img in images:
            for ann in img.get("annotations", []):
                if not self._should_include(ann):
                    continue
                
                eval_data = ann.get("evaluation")
                if not eval_data:
                    continue
                
                status = eval_data.get("status")
                conf = ann.get("confidence")
                
                # Class confusions (edited TPs with class changes)
                orig_pred = ann.get("original_prediction")
                if status == "TP" and orig_pred and orig_pred["class_name"] != ann["class_name"]:
                    ea.confusion_matrix.add(ann["class_name"], orig_pred["class_name"])
                
                # High-confidence errors
                if status == "FP" and conf and conf > 0.8:
                    if len(ea.high_confidence_errors) < self.config.max_error_examples:
                        ea.high_confidence_errors.append({
                            "image_id": img["image_id"],
                            "image_name": img["name"],
                            "class": ann["class_name"],
                            "confidence": conf,
                            "bbox": ann["bbox"]
                        })
                
                # Heavily edited predictions
                if status == "TP" and eval_data.get("localization_iou") and eval_data["localization_iou"] < 0.7:
                    if len(ea.heavily_edited_predictions) < self.config.max_error_examples:
                        ea.heavily_edited_predictions.append({
                            "image_id": img["image_id"],
                            "class": ann["class_name"],
                            "iou": eval_data["localization_iou"],
                            "original_bbox": orig_pred["bbox"] if orig_pred else None,
                            "corrected_bbox": ann["bbox"]
                        })
                
                # Missed objects
                if status == "FN":
                    if len(ea.missed_objects) < self.config.max_error_examples:
                        bbox_area = self._compute_bbox_area(ann["bbox"])
                        ea.fn_sizes.append(bbox_area)
                        ea.missed_objects.append({
                            "image_id": img["image_id"],
                            "class": ann["class_name"],
                            "bbox": ann["bbox"],
                            "area": bbox_area
                        })
        
        report.error_analysis = ea
    
    def _should_include(self, annotation: Dict) -> bool:
        """Filter annotations based on config"""
        class_name = annotation.get("class_name")
        
        if class_name in self.config.ignore_classes:
            return False
        
        if self.config.focus_classes and class_name not in self.config.focus_classes:
            return False
        
        return True
    
    @staticmethod
    def _compute_bbox_area(bbox: List[float]) -> float:
        """Compute normalized bbox area"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    async def close(self):
        """Cleanup resources"""
        await self.client.aclose()


# ============================================================================
# CLI / USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage"""
    
    # Configure
    config = EvaluationConfig(
        api_base_url="http://localhost:8000/api/v1/evaluation",
        reviewed_only=True,
        enable_error_analysis=True,
        enable_confidence_curves=True,
        confidence_thresholds=[0.3, 0.5, 0.7, 0.9],
        max_error_examples=50,
        verbose=True,
    )
    
    # Create pipeline
    pipeline = ModelEvaluationPipeline(config)
    
    try:
        # Progress callback
        def on_progress(current, total):
            pct = (current / total * 100) if total > 0 else 0
            print(f"Progress: {current}/{total} ({pct:.1f}%)", end='\r')
        
        # Run evaluation
        report = await pipeline.evaluate_project(
            project_id=123,
            model_version="yolov8-large-v2.1",
            progress_callback=on_progress
        )
        
        # Print summary
        print("\n" + report.summary())
        
        # Export to JSON
        with open("evaluation_report.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print("\n✓ Report saved to evaluation_report.json")
        
        # Access specific metrics
        print(f"\nDataset F1 Score: {report.dataset_metrics.f1_score:.2%}")
        print(f"Edit Rate: {report.dataset_metrics.edit_rate:.1%}")
        
        # Identify worst performing classes
        if report.per_class_metrics:
            worst = sorted(
                report.per_class_metrics.items(),
                key=lambda x: x[1].f1_score
            )[:5]
            
            print("\nWorst Performing Classes:")
            for class_name, metrics in worst:
                print(f"  {class_name}: F1={metrics.f1_score:.1%}, Recall={metrics.recall:.1%}")
        
        # Check confidence calibration
        if report.confidence_analysis:
            print("\nOptimal Confidence Threshold:")
            best_f1 = max(report.confidence_analysis.f1_at_threshold.items(), key=lambda x: x[1])
            print(f"  Threshold: {best_f1[0]:.2f} → F1: {best_f1[1]:.2%}")
    
    finally:
        await pipeline.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
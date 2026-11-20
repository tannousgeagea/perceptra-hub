"""
Alert rules and threshold management
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AlertThreshold:
    """Defines when to trigger alerts"""
    
    metric_name: str
    critical_threshold: float
    warning_threshold: float
    higher_is_better: bool = True  # True for F1/precision, False for error rates
    
    def check(self, current_value: float, previous_value: Optional[float] = None) -> Optional[str]:
        """
        Check if alert should be triggered.
        Returns: 'critical', 'warning', or None
        """
        
        if self.higher_is_better:
            # Lower values = worse (F1, precision, recall)
            if current_value < self.critical_threshold:
                return 'critical'
            elif current_value < self.warning_threshold:
                return 'warning'
        else:
            # Higher values = worse (edit_rate, hallucination_rate)
            if current_value > self.critical_threshold:
                return 'critical'
            elif current_value > self.warning_threshold:
                return 'warning'
        
        # A worsening of more than 10% since the previous snapshot. Only the
        # bad direction counts: a metric that improves is not an alert.
        if previous_value:
            change_percent = (current_value - previous_value) / previous_value * 100
            worsened = -change_percent if self.higher_is_better else change_percent
            if worsened > 10:
                return 'warning'
        
        return None


class AlertRuleEngine:
    """Manages alert rules and evaluation"""
    
    DEFAULT_THRESHOLDS = [
        AlertThreshold('f1_score', critical_threshold=0.60, warning_threshold=0.70),
        AlertThreshold('precision', critical_threshold=0.65, warning_threshold=0.75),
        AlertThreshold('recall', critical_threshold=0.60, warning_threshold=0.70),
        AlertThreshold('edit_rate', critical_threshold=0.50, warning_threshold=0.30, higher_is_better=False),
        AlertThreshold('hallucination_rate', critical_threshold=0.25, warning_threshold=0.15, higher_is_better=False),
    ]
    
    def __init__(self, custom_thresholds: Optional[List[AlertThreshold]] = None):
        self.thresholds = {t.metric_name: t for t in (custom_thresholds or self.DEFAULT_THRESHOLDS)}
    
    def evaluate_snapshot(
        self,
        current_snapshot,
        previous_snapshot = None
    ) -> List[Dict]:
        """
        Evaluate a snapshot against thresholds.
        Returns list of alerts to create.
        """
        
        alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            current_value = getattr(current_snapshot, metric_name, None)
            if current_value is None:
                continue
            
            previous_value = getattr(previous_snapshot, metric_name, None) if previous_snapshot else None
            
            severity = threshold.check(current_value, previous_value)
            
            if severity:
                # Calculate change
                change_percent = None
                if previous_value and previous_value != 0:
                    change_percent = ((current_value - previous_value) / previous_value) * 100
                
                # Build message
                if change_percent:
                    message = (
                        f"{metric_name.upper()} {severity}: {current_value:.2%} "
                        f"({change_percent:+.1f}% from previous: {previous_value:.2%})"
                    )
                else:
                    message = f"{metric_name.upper()} {severity}: {current_value:.2%}"
                
                alerts.append({
                    'severity': severity,
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'previous_value': previous_value,
                    'threshold_value': threshold.critical_threshold if severity == 'critical' else threshold.warning_threshold,
                    'change_percent': change_percent,
                    'message': message,
                })
        
        return alerts


# ── Per-project thresholds ────────────────────────────────────────────────
# Stored in Project.settings["alert_thresholds"] as
# {metric_name: {"warning": float, "critical": float}}; metrics not listed
# use DEFAULT_THRESHOLDS. Direction (higher_is_better) is fixed per metric.

SETTINGS_KEY = "alert_thresholds"


def thresholds_for_project(project) -> List[AlertThreshold]:
    saved = ((project.settings or {}) if project is not None else {}).get(SETTINGS_KEY) or {}
    result = []
    for default in AlertRuleEngine.DEFAULT_THRESHOLDS:
        values = saved.get(default.metric_name) or {}
        result.append(AlertThreshold(
            metric_name=default.metric_name,
            warning_threshold=float(values.get("warning", default.warning_threshold)),
            critical_threshold=float(values.get("critical", default.critical_threshold)),
            higher_is_better=default.higher_is_better,
        ))
    return result


def validate_thresholds(values: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Checked values to store; raises ValueError with a readable message."""
    defaults = {t.metric_name: t for t in AlertRuleEngine.DEFAULT_THRESHOLDS}
    clean = {}
    for metric, pair in values.items():
        if metric not in defaults:
            raise ValueError(f"Unknown metric '{metric}'")
        try:
            warning, critical = float(pair["warning"]), float(pair["critical"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{metric}: warning and critical must both be numbers")
        if not (0 <= warning <= 1 and 0 <= critical <= 1):
            raise ValueError(f"{metric}: thresholds are rates between 0 and 1")
        if defaults[metric].higher_is_better and critical > warning:
            raise ValueError(f"{metric}: higher is better, so critical must be at or below warning")
        if not defaults[metric].higher_is_better and critical < warning:
            raise ValueError(f"{metric}: lower is better, so critical must be at or above warning")
        clean[metric] = {"warning": warning, "critical": critical}
    return clean


def save_project_thresholds(project_id, values: Optional[Dict[str, Dict[str, float]]]) -> None:
    """Replace the project's thresholds (None or {} = back to the defaults). Other settings stay."""
    from django.db import transaction
    from projects.models import Project

    with transaction.atomic():
        settings = (
            Project.objects.select_for_update().filter(pk=project_id)
            .values_list("settings", flat=True).first()
        )
        settings = dict(settings or {})
        if values:
            settings[SETTINGS_KEY] = values
        else:
            settings.pop(SETTINGS_KEY, None)
        Project.objects.filter(pk=project_id).update(settings=settings)
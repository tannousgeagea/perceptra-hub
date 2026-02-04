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
        
        # Check for drops (if previous value available)
        if previous_value:
            change_percent = abs((current_value - previous_value) / previous_value * 100)
            if change_percent > 10:  # 10% drop
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
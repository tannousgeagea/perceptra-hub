# apps/activity/monitoring.py
import logging
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)


class ActivityMonitor:
    """Monitor activity tracking system health."""
    
    @staticmethod
    def check_event_lag():
        """Check if event processing is lagging."""
        latest_event = ActivityEvent.objects.order_by('-timestamp').first()
        
        if latest_event:
            lag = (timezone.now() - latest_event.timestamp).total_seconds()
            
            if lag > 300:  # 5 minutes
                logger.warning(f"Activity event lag: {lag}s")
                return False
        
        return True
    
    @staticmethod
    def check_aggregation_freshness():
        """Check if aggregations are up to date."""
        latest_metric = UserActivityMetrics.objects.order_by('-updated_at').first()
        
        if latest_metric:
            staleness = (timezone.now() - latest_metric.updated_at).total_seconds()
            
            if staleness > 3600:  # 1 hour
                logger.warning(f"Metrics staleness: {staleness}s")
                return False
        
        return True
    
    @staticmethod
    def get_system_stats():
        """Get system-wide statistics."""
        return {
            'total_events_24h': ActivityEvent.objects.filter(
                timestamp__gte=timezone.now() - timezone.timedelta(hours=24)
            ).count(),
            'active_users_24h': ActivityEvent.objects.filter(
                timestamp__gte=timezone.now() - timezone.timedelta(hours=24)
            ).values('user_id').distinct().count(),
            'cache_hit_rate': cache.get('activity:hit_rate', 0),
        }
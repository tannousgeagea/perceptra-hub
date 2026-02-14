"""
Temporal Analysis API Endpoints
"""

from fastapi import APIRouter, Query, Path, HTTPException
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel

from temporal_analysis.models import MetricSnapshot, MetricAlert
from projects.models import Project
from api.routers.temporal.schemas import TemporalResponse, TrendPoint, TrendAnalysis, SnapshotResponse


router = APIRouter(prefix="/temporal",)

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/projects/{project_id}/trends",
    response_model=TemporalResponse,
    summary="Get performance trends over time"
)
async def get_project_trends(
    project_id: int = Path(...),
    days: int = Query(30, ge=1, le=365, description="Lookback period in days"),
    model_version_id: Optional[int] = Query(None, description="Filter by model version"),
):
    """
    Get time-series performance trends.
    Returns daily snapshots for the specified period.
    """
    
    # Verify project exists
    try:
        project = Project.objects.get(id=project_id, is_deleted=False)
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get snapshots
    start_date = datetime.utcnow() - timedelta(days=days)
    
    snapshots = MetricSnapshot.objects.filter(
        project_id=project_id,
        snapshot_date__gte=start_date
    ).order_by('snapshot_date')
    
    if model_version_id:
        snapshots = snapshots.filter(model_version_id=model_version_id)
    
    # Transform to trend points
    trends = [
        TrendPoint(
            date=s.snapshot_date,
            precision=s.precision,
            recall=s.recall,
            f1_score=s.f1_score,
            edit_rate=s.edit_rate,
            hallucination_rate=s.hallucination_rate,
            tp=s.tp,
            fp=s.fp,
            fn=s.fn
        )
        for s in snapshots
    ]
    
    # Compute analysis
    analysis = []
    if len(snapshots) >= 2:
        current = snapshots.last()
        previous = snapshots[len(snapshots) - 2]
        
        for metric_name in ['precision', 'recall', 'f1_score', 'edit_rate']:
            current_val = getattr(current, metric_name)
            previous_val = getattr(previous, metric_name)
            
            change_pct = None
            direction = 'stable'
            if previous_val and previous_val != 0:
                change_pct = ((current_val - previous_val) / previous_val) * 100
                
                if abs(change_pct) < 2:
                    direction = 'stable'
                elif change_pct > 0:
                    direction = 'improving' if metric_name != 'edit_rate' else 'degrading'
                else:
                    direction = 'degrading' if metric_name != 'edit_rate' else 'improving'
            
            # Compute rolling means
            recent_7d = snapshots[max(0, len(snapshots) - 7):]
            recent_30d = snapshots
            
            mean_7d = sum(getattr(s, metric_name) for s in recent_7d) / len(recent_7d) if recent_7d else None
            mean_30d = sum(getattr(s, metric_name) for s in recent_30d) / len(recent_30d) if recent_30d else None
            
            analysis.append(TrendAnalysis(
                metric_name=metric_name,
                current_value=current_val,
                previous_value=previous_val,
                change_percent=change_pct,
                direction=direction,
                mean_7d=mean_7d,
                mean_30d=mean_30d
            ))
    
    # Build response
    date_range = (snapshots.first().snapshot_date, snapshots.last().snapshot_date) if snapshots else (start_date, datetime.utcnow())
    
    return TemporalResponse(
        project_id=project_id,
        project_name=project.name,
        date_range=date_range,
        trends=trends,
        analysis=analysis,
        snapshot_count=len(snapshots)
    )


@router.get(
    "/projects/{project_id}/snapshots",
    response_model=List[SnapshotResponse],
    summary="Get all snapshots for a project"
)
async def get_project_snapshots(
    project_id: int = Path(...),
    limit: int = Query(100, ge=1, le=500),
):
    """Get list of all metric snapshots"""
    
    snapshots = MetricSnapshot.objects.filter(
        project_id=project_id
    ).select_related('model_version').order_by('-snapshot_date')[:limit]
    
    return [
        SnapshotResponse(
            id=s.id,
            snapshot_date=s.snapshot_date,
            model_version=s.model_version.display_name if s.model_version else None,
            precision=s.precision,
            recall=s.recall,
            f1_score=s.f1_score,
            edit_rate=s.edit_rate,
            total_images=s.total_images
        )
        for s in snapshots
    ]
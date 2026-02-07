
import asyncio
import time
from celery import shared_task
from datetime import datetime

@shared_task(name='evaluation.create_daily_snapshots')
def create_daily_snapshots():
    """Create daily metric snapshots for all active projects"""
    from projects.models import Project
    from temporal_analysis.models import MetricSnapshot
    from api.routers.evaluation.queries.evaluation import EvaluationQueryBuilder
    from .alert import check_metric_alerts
    
    active_projects = Project.objects.filter(is_active=True, is_deleted=False)
    
    results = {
        'total_projects': 0,
        'snapshots_created': 0,
        'errors': []
    }
    
    for project in active_projects:
        try:
            start_time = time.time()
            
            # Get current production model version (if exists)
            production_model = project.ml_models.filter(
                is_deleted=False
            ).first()
            
            model_version = None
            if production_model:
                model_version = production_model.get_production_version()
            
            # Get metrics
            query_builder = EvaluationQueryBuilder()
            
            async def get_metrics():
                return await query_builder.get_quick_summary(
                    project_id=project.id,
                    model_version=model_version.version_id if model_version else None
                )
            
            summary = asyncio.run(get_metrics())
            
            # Create snapshot
            snapshot = MetricSnapshot.objects.create(
                project=project,
                model_version=model_version,
                total_images=summary.total_images,
                reviewed_images=summary.reviewed_images,
                total_annotations=summary.total_annotations,
                precision=summary.precision,
                recall=summary.recall,
                f1_score=summary.f1_score,
                tp=summary.tp,
                fp=summary.fp,
                fn=summary.fn,
                edit_rate=summary.edit_rate,
                hallucination_rate=summary.hallucination_rate,
                computation_time_seconds=time.time() - start_time,
            )
            
            
            check_metric_alerts.delay(project.id, snapshot.id)

            results['snapshots_created'] += 1
            results['total_projects'] += 1
            
        except Exception as e:
            results['errors'].append({
                'project_id': project.id,
                'error': str(e)
            })
    
    return results
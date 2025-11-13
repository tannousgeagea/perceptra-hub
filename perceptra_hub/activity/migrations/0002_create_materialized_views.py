# apps/activity/migrations/0003_create_materialized_views.py
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('activity', '0001_initial'), 
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                CREATE MATERIALIZED VIEW org_activity_summary AS
                SELECT 
                    organization_id,
                    DATE_TRUNC('day', timestamp) as date,
                    COUNT(*) as total_events,
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) FILTER (WHERE event_type LIKE 'annotation%') as annotation_events,
                    COUNT(*) FILTER (WHERE event_type LIKE 'image%') as image_events,
                    COUNT(*) FILTER (WHERE event_type = 'annotation_create') as annotations_created,
                    COUNT(*) FILTER (WHERE event_type = 'image_review') as images_reviewed,
                    COUNT(*) FILTER (WHERE event_type = 'image_upload') as images_uploaded
                FROM activity_event
                WHERE timestamp > NOW() - INTERVAL '90 days'
                GROUP BY organization_id, DATE_TRUNC('day', timestamp);
                
                CREATE UNIQUE INDEX org_activity_summary_org_date_idx 
                ON org_activity_summary (organization_id, date DESC);
            """,
            reverse_sql="DROP MATERIALIZED VIEW IF EXISTS org_activity_summary CASCADE;"
        )
    ]
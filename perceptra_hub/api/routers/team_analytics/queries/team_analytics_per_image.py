
from enum import Enum
import os
import sys
import time
import django
django.setup()
from django.conf import settings
from django.db.models import Count, Q, Avg, Sum
from django.db.models.functions import TruncDate
from django.contrib.auth import get_user_model
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi import Request, Response
from fastapi.routing import APIRoute, APIRouter
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, date, timedelta
from asgiref.sync import sync_to_async
from django.utils import timezone
django.setup()

# Import Django models after setup
from jobs.models import Job, JobImage
from projects.models import ProjectImage
from fastapi import Depends
from api.dependencies import RequestContext, require_organization_admin
from api.routers.team_analytics.queries.team_analytics import org_jobs, org_roles
User = get_user_model()

# Pydantic models for request/response
class TimeFrame(str, Enum):
    day = "day"
    week = "week"
    month = "month"

class ImageAnalyticsFilters(BaseModel):
    timeFrame: TimeFrame
    role: Optional[str] = None
    userId: Optional[str] = None
    projectId: Optional[str] = None
    status: Optional[str] = None  # Filter by image status

class UserImageAnalytics(BaseModel):
    userId: str
    userName: str
    userRole: str
    date: str  # YYYY-MM-DD format
    annotatedImages: int
    reviewedImages: int
    finalizedImages: int
    unannotatedImages: int
    nullMarkedImages: int
    totalImagesWorked: int
    averageImagesPerJob: float

class ImageAnalyticsKPIs(BaseModel):
    totalAnnotatedThisWeek: int
    totalReviewedThisWeek: int
    totalFinalizedThisWeek: int
    totalImagesInProgress: int
    topPerformer: Dict[str, Any]
    averageImagesPerUser: float
    imageCompletionRate: float  # Percentage of finalized vs total

class ImageAnalyticsResponse(BaseModel):
    data: List[UserImageAnalytics]
    kpis: ImageAnalyticsKPIs

class ImageStatusBreakdown(BaseModel):
    unannotated: int
    annotated: int
    reviewed: int
    dataset: int
    total: int

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)
def get_date_range(timeframe: TimeFrame, base_date: datetime = None) -> tuple[datetime, datetime]:
    """Generate date range based on timeframe"""
    if base_date is None:
        base_date = timezone.now()
    
    if timeframe == TimeFrame.day:
        start_date = base_date - timedelta(days=6)
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), base_date
    elif timeframe == TimeFrame.week:
        start_date = base_date - timedelta(weeks=5)
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), base_date
    elif timeframe == TimeFrame.month:
        start_date = base_date - timedelta(days=150)
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), base_date

def get_user_display_name(user):
    """Get user display name"""
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.username

def parse_user_id(user_id: str) -> int:
    if not str(user_id).isdigit():
        raise HTTPException(status_code=400, detail="Invalid userId")
    return int(user_id)

@sync_to_async
def get_user_image_analytics_data_sync(organization, filters: ImageAnalyticsFilters) -> List[UserImageAnalytics]:
    """Fetch one organization's image-level analytics using Django ORM (sync version)"""
    start_date, end_date = get_date_range(filters.timeFrame)
    roles = org_roles(organization)

    # Base queryset: this organization's job images within the date range
    job_images_queryset = JobImage.objects.filter(
        job__in=org_jobs(organization),
        is_active=True,
        project_image__updated_at__gte=start_date,
        project_image__updated_at__lte=end_date,
    )

    # Apply filters
    if filters.role:
        job_images_queryset = job_images_queryset.filter(
            job__assignee_id__in=[uid for uid, name in roles.items() if name == filters.role]
        )

    if filters.userId:
        job_images_queryset = job_images_queryset.filter(job__assignee_id=parse_user_id(filters.userId))

    if filters.projectId:
        job_images_queryset = job_images_queryset.filter(job__project__project_id=filters.projectId)
    
    if filters.status:
        job_images_queryset = job_images_queryset.filter(project_image__status=filters.status)
    
    # Get daily image activity grouped by user and date
    daily_image_activity = list(
        job_images_queryset
        .annotate(activity_date=TruncDate('created_at'))
        .values(
            'job__assignee_id',
            'job__assignee__username',
            'job__assignee__first_name',
            'job__assignee__last_name',
            'activity_date'
        )
        .annotate(
            total_images=Count('id'),
            annotated_images=Count('id', filter=Q(project_image__annotated=True)),
            reviewed_images=Count('id', filter=Q(project_image__reviewed=True)),
            finalized_images=Count('id', filter=Q(project_image__finalized=True)),
            unannotated_images=Count('id', filter=Q(project_image__status='unannotated')),
            null_marked_images=Count('id', filter=Q(project_image__marked_as_null=True)),
        )
        .order_by('activity_date', 'job__assignee__username')
    )
    
    # Get average images per job for each user
    job_stats = {}
    if daily_image_activity:
        user_job_stats = list(
            org_jobs(organization).filter(
                created_at__gte=start_date,
                created_at__lte=end_date
            )
            .values('assignee_id')
            .annotate(
                avg_images=Avg('image_count'),
                job_count=Count('id')
            )
        )
        
        for stat in user_job_stats:
            job_stats[stat['assignee_id']] = {
                'avg_images': float(stat['avg_images'] or 0),
                'job_count': stat['job_count']
            }
    
    analytics_data = []
    for item in daily_image_activity:
        user_id = item['job__assignee_id']
        
        # Get user display name
        user_name = get_user_display_name(
            type('User', (), {
                'first_name': item['job__assignee__first_name'] or '',
                'last_name': item['job__assignee__last_name'] or '',
                'username': item['job__assignee__username']
            })()
        )
        
        # Get average images per job
        avg_images_per_job = job_stats.get(user_id, {}).get('avg_images', 0.0)
        
        analytics_data.append(UserImageAnalytics(
            userId=str(user_id),
            userName=user_name,
            userRole=roles.get(user_id, ""),
            date=item['activity_date'].strftime('%Y-%m-%d'),
            annotatedImages=item['annotated_images'],
            reviewedImages=item['reviewed_images'],
            finalizedImages=item['finalized_images'],
            unannotatedImages=item['unannotated_images'],
            nullMarkedImages=item['null_marked_images'],
            totalImagesWorked=item['total_images'],
            averageImagesPerJob=round(avg_images_per_job, 2)
        ))
    
    return analytics_data

async def get_user_image_analytics_data(organization, filters: ImageAnalyticsFilters) -> List[UserImageAnalytics]:
    """Fetch image-level analytics data (async wrapper)"""
    return await get_user_image_analytics_data_sync(organization, filters)

@sync_to_async
def calculate_image_kpis_sync(organization, analytics_data: List[UserImageAnalytics]) -> ImageAnalyticsKPIs:
    """Calculate image-level KPIs (sync version)"""
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    
    # Filter data for this week
    this_week_data = [
        entry for entry in analytics_data 
        if datetime.strptime(entry.date, '%Y-%m-%d').date() >= week_ago
    ]
    
    total_annotated_this_week = sum(entry.annotatedImages for entry in this_week_data)
    total_reviewed_this_week = sum(entry.reviewedImages for entry in this_week_data)
    total_finalized_this_week = sum(entry.finalizedImages for entry in this_week_data)
    
    # Get total images in progress (assigned but not finalized)
    total_images_in_progress = ProjectImage.objects.filter(
        project__organization=organization,
        project__is_deleted=False,
        job_assignments__isnull=False,
        finalized=False,
        is_active=True
    ).distinct().count()
    
    # Calculate top performer (by total images worked)
    user_totals: Dict[str, Dict[str, Any]] = {}
    for entry in analytics_data:
        if entry.userId not in user_totals:
            user_totals[entry.userId] = {
                'name': entry.userName,
                'total': 0,
                'finalized': 0
            }
        user_totals[entry.userId]['total'] += entry.totalImagesWorked
        user_totals[entry.userId]['finalized'] += entry.finalizedImages
    
    # Find top performer
    top_performer_data = {'userId': '', 'userName': '', 'score': 0, 'finalizedCount': 0}
    for user_id, data in user_totals.items():
        if data['total'] > top_performer_data['score']:
            top_performer_data = {
                'userId': user_id,
                'userName': data['name'],
                'score': data['total'],
                'finalizedCount': data['finalized']
            }
    
    # Calculate average images per user
    unique_users = len(user_totals)
    avg_images_per_user = (
        sum(data['total'] for data in user_totals.values()) / unique_users
        if unique_users > 0 else 0
    )
    
    # Calculate image completion rate
    total_images_all_time = sum(entry.totalImagesWorked for entry in analytics_data)
    total_finalized_all_time = sum(entry.finalizedImages for entry in analytics_data)
    completion_rate = (
        (total_finalized_all_time / total_images_all_time * 100)
        if total_images_all_time > 0 else 0
    )
    
    return ImageAnalyticsKPIs(
        totalAnnotatedThisWeek=total_annotated_this_week,
        totalReviewedThisWeek=total_reviewed_this_week,
        totalFinalizedThisWeek=total_finalized_this_week,
        totalImagesInProgress=total_images_in_progress,
        topPerformer=top_performer_data,
        averageImagesPerUser=round(avg_images_per_user, 2),
        imageCompletionRate=round(completion_rate, 2)
    )

async def calculate_image_kpis(organization, analytics_data: List[UserImageAnalytics]) -> ImageAnalyticsKPIs:
    """Calculate image-level KPIs (async wrapper)"""
    return await calculate_image_kpis_sync(organization, analytics_data)

@router.get("/analytics/images", response_model=ImageAnalyticsResponse)
async def get_image_analytics(
    timeFrame: TimeFrame = Query(..., description="Time frame for analytics"),
    role: Optional[str] = Query(None, description="Filter by user role"),
    userId: Optional[str] = Query(None, description="Filter by specific user ID"),
    projectId: Optional[str] = Query(None, description="Filter by project ID"),
    status: Optional[str] = Query(None, description="Filter by image status"),
    ctx: RequestContext = Depends(require_organization_admin()),
):
    """
    Get team analytics data at the image level with optional filters.
    
    - **timeFrame**: day (last 7 days), week (last 6 weeks), or month (last 6 months)
    - **role**: Filter by user role (optional)
    - **userId**: Filter by specific user ID (optional)
    - **projectId**: Filter by project ID (optional)
    - **status**: Filter by image status (unannotated, annotated, reviewed, dataset)
    """
    try:
        filters = ImageAnalyticsFilters(
            timeFrame=timeFrame,
            role=role,
            userId=userId,
            projectId=projectId,
            status=status
        )
        
        # Get analytics data
        analytics_data = await get_user_image_analytics_data(ctx.organization, filters)

        # Calculate KPIs
        kpis = await calculate_image_kpis(ctx.organization, analytics_data)

        return ImageAnalyticsResponse(
            data=analytics_data,
            kpis=kpis
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image analytics: {str(e)}")

@sync_to_async
def get_image_status_breakdown_sync(organization, projectId: Optional[str] = None) -> ImageStatusBreakdown:
    """Get breakdown of one organization's images by status (sync version)"""
    queryset = ProjectImage.objects.filter(
        project__organization=organization, project__is_deleted=False, is_active=True
    )

    if projectId:
        queryset = queryset.filter(project__project_id=projectId)
    
    status_counts = queryset.aggregate(
        unannotated=Count('id', filter=Q(status='unannotated')),
        annotated=Count('id', filter=Q(status='annotated')),
        reviewed=Count('id', filter=Q(status='reviewed')),
        dataset=Count('id', filter=Q(status='dataset')),
        total=Count('id')
    )
    
    return ImageStatusBreakdown(**status_counts)

@router.get("/analytics/images/status-breakdown", response_model=ImageStatusBreakdown)
async def get_image_status_breakdown(
    projectId: Optional[str] = Query(None, description="Filter by project ID (UUID)"),
    ctx: RequestContext = Depends(require_organization_admin()),
):
    """
    Get breakdown of images by status.
    
    - **projectId**: Filter by project ID (optional)
    """
    try:
        return await get_image_status_breakdown_sync(ctx.organization, projectId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching status breakdown: {str(e)}")

@sync_to_async
def get_user_image_performance_sync(organization, userId: str, timeFrame: TimeFrame) -> Dict[str, Any]:
    """Get detailed image performance for one member of the organization (sync version)"""
    start_date, end_date = get_date_range(timeFrame)
    roles = org_roles(organization)

    # Only members of this organization
    user_id = parse_user_id(userId)
    user = User.objects.filter(id=user_id).first() if user_id in roles else None
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Get image statistics (this organization's jobs only)
    job_images = JobImage.objects.filter(
        job__in=org_jobs(organization),
        is_active=True,
        job__assignee_id=user_id,
        project_image__updated_at__gte=start_date,  # Use updated_at
        project_image__updated_at__lte=end_date,
    )
    
    stats = job_images.aggregate(
        total_images=Count('id'),
        annotated=Count('id', filter=Q(project_image__annotated=True)),
        reviewed=Count('id', filter=Q(project_image__reviewed=True)),
        finalized=Count('id', filter=Q(project_image__finalized=True)),
        null_marked=Count('id', filter=Q(project_image__marked_as_null=True)),
    )
    
    # Get jobs statistics
    jobs_stats = org_jobs(organization).filter(
        assignee_id=user_id,
        created_at__gte=start_date,
        created_at__lte=end_date
    ).aggregate(
        total_jobs=Count('id'),
        completed_jobs=Count('id', filter=Q(status='completed')),
        in_review_jobs=Count('id', filter=Q(status='in_review')),
        avg_images_per_job=Avg('image_count')
    )
    
    return {
        'userId': str(user.id),
        'userName': get_user_display_name(user),
        'userRole': roles.get(user_id, ""),
        'imageStats': stats,
        'jobStats': jobs_stats,
        'timeFrame': timeFrame,
        'dateRange': {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
    }

@router.get("/analytics/images/user/{userId}")
async def get_user_image_performance(
    userId: str,
    timeFrame: TimeFrame = Query(TimeFrame.week, description="Time frame for analytics"),
    ctx: RequestContext = Depends(require_organization_admin()),
):
    """
    Get detailed image performance for a specific user.
    
    - **userId**: User ID
    - **timeFrame**: Time frame for analytics
    """
    try:
        return await get_user_image_performance_sync(ctx.organization, userId, timeFrame)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user performance: {str(e)}")

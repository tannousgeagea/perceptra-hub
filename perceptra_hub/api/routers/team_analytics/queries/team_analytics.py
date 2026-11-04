
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


from fastapi import Depends
from api.dependencies import RequestContext, require_organization_admin
from jobs.models import Job, JobImage
from memberships.models import OrganizationMembership
User = get_user_model()

# Pydantic models for request/response
class TimeFrame(str, Enum):
    day = "day"
    week = "week"
    month = "month"

class AnalyticsFilters(BaseModel):
    timeFrame: TimeFrame
    role: Optional[str] = None
    userId: Optional[str] = None
    projectId: Optional[str] = None

class UserAnalytics(BaseModel):
    userId: str# Update with your actual  name
    userName: str
    userRole: str
    date: str  # YYYY-MM-DD format
    annotatedCount: int
    reviewedCount: int
    completedCount: int
    totalTime: int  # in minutes

class TopPerformer(BaseModel):
    userId: str
    userName: str
    score: int

class AnalyticsKPIs(BaseModel):
    totalAnnotationsThisWeek: int
    totalReviewsThisWeek: int
    totalCompletionsThisWeek: int
    topPerformer: TopPerformer
    averageCompletionTimeMinutes: int

class AnalyticsResponse(BaseModel):
    data: List[UserAnalytics]
    kpis: AnalyticsKPIs

class UserInfo(BaseModel):
    id: str
    name: str
    role: str

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
        # Last 7 days
        start_date = base_date - timedelta(days=6)
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), base_date
    elif timeframe == TimeFrame.week:
        # Last 6 weeks
        start_date = base_date - timedelta(weeks=5)
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), base_date
    elif timeframe == TimeFrame.month:
        # Last 6 months
        start_date = base_date - timedelta(days=150)  # roximate 6 months
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), base_date

def get_user_display_name(user):
    """Get user display name"""
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.username

def org_roles(organization) -> Dict[int, str]:
    """user id -> role name, for the organization's active members."""
    return {
        m.user_id: m.role.name
        for m in OrganizationMembership.objects.filter(
            organization=organization, status="active"
        ).select_related("role")
    }


def org_jobs(organization):
    """Assigned, active jobs of the organization's live projects."""
    return Job.objects.filter(
        project__organization=organization,
        project__is_deleted=False,
        is_active=True,
        assignee__isnull=False,
    )


@sync_to_async
def get_user_analytics_data(organization, filters: AnalyticsFilters) -> List[UserAnalytics]:
    """Fetch analytics data for one organization using Django ORM"""
    start_date, end_date = get_date_range(filters.timeFrame)
    roles = org_roles(organization)

    # Base queryset: this organization's jobs within the date range
    jobs_queryset = org_jobs(organization).filter(
        updated_at__gte=start_date,
        updated_at__lte=end_date,
    )

    # Role filter: members holding that org role
    if filters.role:
        jobs_queryset = jobs_queryset.filter(
            assignee_id__in=[uid for uid, name in roles.items() if name == filters.role]
        )

    if filters.userId:
        if not filters.userId.isdigit():
            raise HTTPException(status_code=400, detail="Invalid userId")
        jobs_queryset = jobs_queryset.filter(assignee_id=int(filters.userId))

    # Project filter: the project's public UUID
    if filters.projectId:
        jobs_queryset = jobs_queryset.filter(project__project_id=filters.projectId)
    
    # Get daily activity data grouped by user and date
    daily_activity = (
        jobs_queryset
        .annotate(activity_date=TruncDate('updated_at'))
        .values('assignee_id', 'assignee__username', 'assignee__first_name', 
                'assignee__last_name', 'activity_date')
        .annotate(
            completed_count=Count('id', filter=Q(status='completed')),
            reviewed_count=Count('id', filter=Q(status='in_review')),
            total_jobs=Count('id'),
            avg_image_count=Avg('image_count')
        )
        .order_by('activity_date', 'assignee__username')
    )
    
    # Get annotation data (JobImage entries)
    annotation_data = {}
    if daily_activity:
        job_images = (
            JobImage.objects
            .filter(
                job__in=jobs_queryset,
                created_at__gte=start_date,
                created_at__lte=end_date,
            )
            .annotate(activity_date=TruncDate('created_at'))
            .values('job__assignee_id', 'activity_date')
            .annotate(annotated_count=Count('id'))
        )
        
        for item in job_images:
            key = (item['job__assignee_id'], item['activity_date'])
            annotation_data[key] = item['annotated_count']
    
    analytics_data = []
    for item in daily_activity:
        user_id = item['assignee_id']
        activity_date = item['activity_date']
        
        # Get user display name
        user_name = get_user_display_name(
            type('User', (), {
                'first_name': item['assignee__first_name'] or '',
                'last_name': item['assignee__last_name'] or '',
                'username': item['assignee__username']
            })()
        )
        
        # Get annotation count for this user/date
        annotated_count = annotation_data.get((user_id, activity_date), 0)
        
        # Estimate total time based on activity (rough estimation)
        # This is a placeholder - you may want to track actual time spent
        base_time = 120  # 2 hours base
        activity_multiplier = (
            item['completed_count'] * 30 + 
            item['reviewed_count'] * 20 + 
            annotated_count * 2
        )
        total_time = min(base_time + activity_multiplier, 480)  # Max 8 hours
        
        analytics_data.append(UserAnalytics(
            userId=str(user_id),
            userName=user_name,
            userRole=roles.get(user_id, ""),
            date=activity_date.strftime('%Y-%m-%d'),
            annotatedCount=annotated_count,
            reviewedCount=item['reviewed_count'],
            completedCount=item['completed_count'],
            totalTime=int(total_time)
        ))
    
    return analytics_data

async def calculate_kpis(analytics_data: List[UserAnalytics]) -> AnalyticsKPIs:
    """Calculate KPIs from analytics data"""
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    
    # Filter data for this week
    this_week_data = [
        entry for entry in analytics_data 
        if datetime.strptime(entry.date, '%Y-%m-%d').date() >= week_ago
    ]
    
    total_annotations_this_week = sum(entry.annotatedCount for entry in this_week_data)
    total_reviews_this_week = sum(entry.reviewedCount for entry in this_week_data)
    total_completions_this_week = sum(entry.completedCount for entry in this_week_data)
    
    # Calculate top performer (by total activity)
    user_totals: Dict[str, Dict[str, Any]] = {}
    for entry in analytics_data:
        if entry.userId not in user_totals:
            user_totals[entry.userId] = {
                'name': entry.userName,
                'total': 0
            }
        user_totals[entry.userId]['total'] += (
            entry.annotatedCount + entry.reviewedCount + entry.completedCount
        )
    
    # Find top performer
    top_performer_data = {'userId': '', 'userName': '', 'score': 0}
    for user_id, data in user_totals.items():
        if data['total'] > top_performer_data['score']:
            top_performer_data = {
                'userId': user_id,
                'userName': data['name'],
                'score': data['total']
            }
    
    # Calculate average completion time
    avg_completion_time = (
        sum(entry.totalTime for entry in this_week_data) // len(this_week_data)
        if this_week_data else 0
    )
    
    return AnalyticsKPIs(
        totalAnnotationsThisWeek=total_annotations_this_week,
        totalReviewsThisWeek=total_reviews_this_week,
        totalCompletionsThisWeek=total_completions_this_week,
        topPerformer=TopPerformer(
            userId=top_performer_data['userId'],
            userName=top_performer_data['userName'],
            score=top_performer_data['score']
        ),
        averageCompletionTimeMinutes=avg_completion_time
    )

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_team_analytics(
    timeFrame: TimeFrame = Query(..., description="Time frame for analytics"),
    role: Optional[str] = Query(None, description="Filter by user role"),
    userId: Optional[str] = Query(None, description="Filter by specific user ID"),
    projectId: Optional[str] = Query(None, description="Filter by project ID (UUID)"),
    ctx: RequestContext = Depends(require_organization_admin()),
):
    """
    Get team analytics data with optional filters.
    
    - **timeFrame**: day (last 7 days), week (last 6 weeks), or month (last 6 months)
    - **role**: Filter by user role (optional)
    - **userId**: Filter by specific user ID (optional)
    - **projectId**: Filter by project ID (optional)
    """
    try:
        filters = AnalyticsFilters(
            timeFrame=timeFrame,
            role=role,
            userId=userId,
            projectId=projectId
        )
        
        # Get analytics data
        analytics_data = await get_user_analytics_data(ctx.organization, filters)
        
        # Calculate KPIs
        kpis = await calculate_kpis(analytics_data)
        
        return AnalyticsResponse(
            data=analytics_data,
            kpis=kpis
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

@sync_to_async
def get_analytics_users_sync(organization):
    """Members of the organization who have been assigned jobs there"""
    roles = org_roles(organization)
    users = User.objects.filter(
        id__in=org_jobs(organization).filter(assignee_id__in=roles.keys()).values("assignee_id")
    ).order_by('first_name', 'last_name', 'username')
    return [
        UserInfo(id=str(user.id), name=get_user_display_name(user), role=roles.get(user.id, ""))
        for user in users
    ]

@router.get("/analytics/users", response_model=List[UserInfo])
async def get_analytics_users(ctx: RequestContext = Depends(require_organization_admin())):
    """Get list of users for filtering"""
    try:
        return await get_analytics_users_sync(ctx.organization)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")


@sync_to_async
def get_analytics_roles_sync(organization):
    """Org roles held by members who have been assigned jobs there"""
    roles = org_roles(organization)
    assignees = set(org_jobs(organization).values_list("assignee_id", flat=True).distinct())
    return sorted({name for uid, name in roles.items() if uid in assignees})

@router.get("/analytics/roles", response_model=List[str])
async def get_analytics_roles(ctx: RequestContext = Depends(require_organization_admin())):
    """Get list of available roles for filtering"""
    try:
        return await get_analytics_roles_sync(ctx.organization)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching roles: {str(e)}")

@sync_to_async
def get_analytics_projects_sync(organization):
    """The organization's live projects that have jobs"""
    from projects.models import Project

    projects = Project.objects.filter(
        organization=organization,
        is_deleted=False,
        jobs__isnull=False,
    ).distinct().order_by('name')
    return [{"id": str(project.project_id), "name": project.name} for project in projects]

@router.get("/analytics/projects", response_model=List[Dict[str, str]])
async def get_analytics_projects(ctx: RequestContext = Depends(require_organization_admin())):
    """Get list of projects for filtering"""
    try:
        return await get_analytics_projects_sync(ctx.organization)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


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


from jobs.models import Job, JobImage 
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
        base_date = datetime.now()
    
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

def get_user_role(user):
    """Get user role - assumes you have a role field or method on User model"""
    # Adjust this based on how roles are stored in your User model
    if hasattr(user, 'role') and user.role:
        return user.role
    elif hasattr(user, 'profile') and hasattr(user.profile, 'role'):
        return user.profile.role
    else:
        return "Annotator"  # Default role


@sync_to_async
def get_user_analytics_data(filters: AnalyticsFilters) -> List[UserAnalytics]:
    """Fetch analytics data using Django ORM"""
    start_date, end_date = get_date_range(filters.timeFrame)
    
    # Base queryset for jobs within date range
    jobs_queryset = Job.objects.filter(
        updated_at__gte=start_date,
        updated_at__lte=end_date,
        assignee__isnull=False  # Only jobs with assignees
    )
    
    # ly role filter if provided
    if filters.role:
        # Adjust this filter based on how roles are stored in your User model
        jobs_queryset = jobs_queryset.filter(assignee__role=filters.role)
    
    # ly user filter if provided
    if filters.userId:
        jobs_queryset = jobs_queryset.filter(assignee_id=filters.userId)
    
    # Apply project filter if provided
    if filters.projectId:
        jobs_queryset = jobs_queryset.filter(project_id=filters.projectId)
    
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
                created_at__gte=start_date,
                created_at__lte=end_date,
                job__assignee__isnull=False
            )
            .annotate(activity_date=TruncDate('created_at'))
            .values('job__assignee_id', 'activity_date')
            .annotate(annotated_count=Count('id'))
        )
        
        for item in job_images:
            key = (item['job__assignee_id'], item['activity_date'])
            annotation_data[key] = item['annotated_count']
    
    # Get user role information
    user_roles = {}
    if daily_activity:
        users = User.objects.filter(
            id__in=[item['assignee_id'] for item in daily_activity]
        )
        for user in users:
            user_roles[user.id] = get_user_role(user)
    
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
            userRole=user_roles.get(user_id, "Annotator"),
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
    projectId: Optional[str] = Query(None, description="Filter by project ID")
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
        analytics_data = await get_user_analytics_data(filters)
        
        # Calculate KPIs
        kpis = await calculate_kpis(analytics_data)
        
        return AnalyticsResponse(
            data=analytics_data,
            kpis=kpis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

@sync_to_async
def get_analytics_users_sync():
    """Get list of users for filtering (sync version)"""
    # Get users who have been assigned jobs
    users = list(User.objects.filter(
        assigned_jobs__isnull=False
    ).distinct().order_by('first_name', 'last_name', 'username'))
    
    user_list = []
    for user in users:
        user_list.append(UserInfo(
            id=str(user.id),
            name=get_user_display_name(user),
            role=get_user_role(user)
        ))
    
    return user_list

@router.get("/analytics/users", response_model=List[UserInfo])
async def get_analytics_users():
    """Get list of users for filtering"""
    try:
        return await get_analytics_users_sync()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")


@sync_to_async
def get_analytics_roles_sync():
    """Get list of available roles for filtering (sync version)"""
    # Get distinct roles from users who have been assigned jobs
    users = list(User.objects.filter(
        assigned_jobs__isnull=False
    ).distinct())
    
    roles = set()
    for user in users:
        role = get_user_role(user)
        if role:
            roles.add(role)
    
    return sorted(list(roles))

@router.get("/analytics/roles", response_model=List[str])
async def get_analytics_roles():
    """Get list of available roles for filtering"""
    try:
        # Get distinct roles from users who have been assigned jobs
        return await get_analytics_roles_sync()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching roles: {str(e)}")

@sync_to_async
def get_analytics_projects_sync():
    """Get list of projects for filtering (sync version)"""
    from projects.models import Project  # Update with your actual app name
    
    # Get projects that have jobs
    projects = list(Project.objects.filter(
        jobs__isnull=False
    ).distinct().order_by('name'))
    
    return [
        {
            "id": str(project.id),
            "name": project.name
        }
        for project in projects
    ]

@router.get("/analytics/projects", response_model=List[Dict[str, str]])
async def get_analytics_projects():
    """Get list of projects for filtering"""
    try:
        return await get_analytics_projects_sync()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

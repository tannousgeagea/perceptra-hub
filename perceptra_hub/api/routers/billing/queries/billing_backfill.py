from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal
from uuid import UUID
from fastapi import Query
from datetime import date, timedelta
from asgiref.sync import sync_to_async
from django.utils import timezone
from django.db.models import Q
from api.dependencies import RequestContext, get_request_context, require_permission
from api.routers.billing.schemas import BackfillTaskResponse

router = APIRouter(prefix="/billing")


# ============================================================================
# BACKFILL ENDPOINTS
# ============================================================================

@router.post(
    "/organizations/{org_id}/members/{user_id}/backfill",
    response_model=BackfillTaskResponse,
    summary="Backfill User Billing in Organization"
)
async def backfill_user_in_org(
    org_id: UUID,
    user_id: int,
    start_date: Optional[datetime] = Body(None),
    end_date: Optional[datetime] = Body(None),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    Backfill all unbilled activities for a user in an organization.
    
    **Use Case**: User just marked as external contractor, calculate past work.
    """
    from api.tasks.billing.billable_action import backfill_billing_for_user_in_org
    
    task = backfill_billing_for_user_in_org.delay(
        str(user_id),
        str(ctx.organization.pk),
        start_date=start_date,
        end_date=end_date
    )
    
    return BackfillTaskResponse(
        message='Backfill task queued for user in organization',
        task_id=task.id,
        scope='organization',
        user_id=user_id,
        organization_id=org_id
    )


@router.post(
    "/projects/{project_id}/members/{user_id}/backfill",
    response_model=BackfillTaskResponse,
    summary="Backfill User Billing in Project"
)
async def backfill_user_in_project(
    project_id: UUID,
    user_id: int,
    start_date: Optional[datetime] = Body(None),
    end_date: Optional[datetime] = Body(None),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    Backfill all unbilled activities for a user in a specific project.
    
    **Use Case**: User marked as external for this project only.
    """
    from api.tasks.billing.billable_action import backfill_billing_for_user_in_project
    
    @sync_to_async
    def get_project_pk(project_id):
        from projects.models import Project
        try:
            return Project.objects.get(project_id=project_id).id
        except Project.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
    
    project_pk = await get_project_pk(project_id)
    
    task = backfill_billing_for_user_in_project.delay(
        str(user_id),
        project_pk,
        start_date=start_date,
        end_date=end_date
    )
    
    return BackfillTaskResponse(
        message='Backfill task queued for user in project',
        task_id=task.id,
        scope='project',
        user_id=user_id,
        project_id=project_id
    )

@router.post(
    "/organizations/{org_id}/backfill",
    response_model=BackfillTaskResponse,
    summary="Backfill All Organization Billing"
)
async def backfill_entire_organization(
    org_id: UUID,
    start_date: Optional[datetime] = Body(None),
    end_date: Optional[datetime] = Body(None),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Backfill all unbilled activities for entire organization.
    
    **Use Case**: Organization just marked as vendor, calculate all historical work.
    **Warning**: Can be slow for large organizations.
    """
    from api.tasks.billing.billable_action import backfill_billing_for_organization
    
    task = backfill_billing_for_organization.delay(
        str(ctx.organization.pk),
        start_date=start_date,
        end_date=end_date
    )
    
    return BackfillTaskResponse(
        message='Backfill task queued for entire organization',
        task_id=task.id,
        scope='organization',
        organization_id=org_id
    )


@router.get(
    "/backfill-tasks/{task_id}/status",
    summary="Get Backfill Task Status"
)
async def get_backfill_task_status(
    task_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Check the status of a backfill task.
    
    **States**: PENDING, STARTED, SUCCESS, FAILURE, RETRY
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id)
    
    response = {
        'task_id': task_id,
        'state': task.state,
        'ready': task.ready(),
    }
    
    if task.ready():
        if task.successful():
            response['result'] = task.result
        elif task.failed():
            response['error'] = str(task.info)
    else:
        response['status'] = 'Processing...'
    
    return response


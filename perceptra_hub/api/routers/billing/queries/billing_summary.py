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
from api.routers.billing.schemas import UserBillingSummary, BillableActionDetail

router = APIRouter(prefix="/billing")

# ============================================================================
# SUMMARY & REPORTING ENDPOINTS
# ============================================================================

@router.get(
    "/organizations/{org_id}/members/{user_id}/summary",
    response_model=UserBillingSummary,
    summary="Get User Billing Summary in Organization"
)
async def get_user_org_billing_summary(
    org_id: UUID,
    user_id: UUID,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    Get comprehensive billing summary for user in organization.
    
    **Includes**: Total costs, action breakdown, billing status.
    """
    @sync_to_async
    def fetch_summary(org_id, user_id, start_date, end_date):
        from memberships.models import OrganizationMembership
        from billing.models import BillableAction
        from django.contrib.auth import get_user_model
        from django.db.models import Sum, Count, Avg
        from organizations.models import Organization
        
        User = get_user_model()
        
        try:
            user = User.objects.get(id=user_id)
            org = ctx.organization
            membership = OrganizationMembership.objects.select_related(
                'billing_rate_card'
            ).get(user=user, organization=org)
        except (User.DoesNotExist, Organization.DoesNotExist, OrganizationMembership.DoesNotExist):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User or organization membership not found"
            )
        
        # Build query
        actions = BillableAction.objects.filter(
            user=user,
            organization=org
        )
        
        if start_date:
            actions = actions.filter(created_at__date__gte=start_date)
        if end_date:
            actions = actions.filter(created_at__date__lte=end_date)
        
        # Aggregate
        summary = actions.aggregate(
            total_actions=Count('action_id'),
            total_amount=Sum('total_amount'),
            total_billed=Sum('total_amount', filter=Q(billed_at__isnull=False)),
            total_unbilled=Sum('total_amount', filter=Q(billed_at__isnull=True)),
            avg_rate=Avg('unit_rate')
        )
        
        # Breakdown by action type
        breakdown = list(actions.values('action_type').annotate(
            count=Count('action_id'),
            total=Sum('total_amount'),
            avg_rate=Avg('unit_rate')
        ).order_by('-total'))
        
        return {
            'user': user,
            'org': org,
            'membership': membership,
            'summary': summary,
            'breakdown': breakdown,
            'period_start': start_date,
            'period_end': end_date
        }
    
    data = await fetch_summary(org_id, user_id, start_date, end_date)
    
    return UserBillingSummary(
        user_id=data['user'].id,
        username=data['user'].username,
        full_name=data['user'].get_full_name() or data['user'].username,
        scope='organization',
        scope_id=data['org'].id,
        scope_name=data['org'].name,
        is_external_annotator=data['membership'].is_external_annotator,
        billing_enabled=data['membership'].billing_enabled,
        hourly_rate=data['membership'].hourly_rate,
        rate_card_id=data['membership'].billing_rate_card.rate_card_id if data['membership'].billing_rate_card else None,
        rate_card_name=data['membership'].billing_rate_card.name if data['membership'].billing_rate_card else None,
        total_actions=data['summary']['total_actions'] or 0,
        total_amount=data['summary']['total_amount'] or Decimal('0.00'),
        total_billed=data['summary']['total_billed'] or Decimal('0.00'),
        total_unbilled=data['summary']['total_unbilled'] or Decimal('0.00'),
        avg_rate=data['summary']['avg_rate'],
        action_breakdown=[
            {
                'action_type': item['action_type'],
                'count': item['count'],
                'total_amount': float(item['total']),
                'avg_rate': float(item['avg_rate'])
            }
            for item in data['breakdown']
        ],
        period_start=data['period_start'],
        period_end=data['period_end']
    )


@router.get(
    "/projects/{project_id}/members/{user_id}/summary",
    response_model=UserBillingSummary,
    summary="Get User Billing Summary in Project"
)
async def get_user_project_billing_summary(
    project_id: UUID,
    user_id: UUID,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """Get billing summary for user in specific project."""
    @sync_to_async
    def fetch_summary(project_id, user_id, start_date, end_date):
        from memberships.models import ProjectMembership
        from billing.models import BillableAction
        from django.contrib.auth import get_user_model
        from django.db.models import Sum, Count, Avg
        from projects.models import Project
        
        User = get_user_model()
        
        try:
            user = User.objects.get(id=user_id)
            project = Project.objects.get(project_id=project_id)
            membership = ProjectMembership.objects.select_related(
                'billing_rate_card'
            ).get(user=user, project=project)
        except (User.DoesNotExist, Project.DoesNotExist, ProjectMembership.DoesNotExist):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User or project membership not found"
            )
        
        # Build query
        actions = BillableAction.objects.filter(
            user=user,
            project=project
        )
        
        if start_date:
            actions = actions.filter(created_at__date__gte=start_date)
        if end_date:
            actions = actions.filter(created_at__date__lte=end_date)
        
        # Aggregate
        summary = actions.aggregate(
            total_actions=Count('action_id'),
            total_amount=Sum('total_amount'),
            total_billed=Sum('total_amount', filter=Q(billed_at__isnull=False)),
            total_unbilled=Sum('total_amount', filter=Q(billed_at__isnull=True)),
            avg_rate=Avg('unit_rate')
        )
        
        # Breakdown
        breakdown = list(actions.values('action_type').annotate(
            count=Count('action_id'),
            total=Sum('total_amount'),
            avg_rate=Avg('unit_rate')
        ).order_by('-total'))
        
        return {
            'user': user,
            'project': project,
            'membership': membership,
            'summary': summary,
            'breakdown': breakdown,
            'period_start': start_date,
            'period_end': end_date
        }
    
    data = await fetch_summary(project_id, user_id, start_date, end_date)
    
    return UserBillingSummary(
        user_id=data['user'].id,
        username=data['user'].username,
        full_name=data['user'].get_full_name() or data['user'].username,
        scope='project',
        scope_id=data['project'].project_id,
        scope_name=data['project'].name,
        is_external_annotator=data['membership'].is_external_annotator,
        billing_enabled=data['membership'].billing_enabled,
        hourly_rate=data['membership'].hourly_rate,
        rate_card_id=data['membership'].billing_rate_card.rate_card_id if data['membership'].billing_rate_card else None,
        rate_card_name=data['membership'].billing_rate_card.name if data['membership'].billing_rate_card else None,
        total_actions=data['summary']['total_actions'] or 0,
        total_amount=data['summary']['total_amount'] or Decimal('0.00'),
        total_billed=data['summary']['total_billed'] or Decimal('0.00'),
        total_unbilled=data['summary']['total_unbilled'] or Decimal('0.00'),
        avg_rate=data['summary']['avg_rate'],
        action_breakdown=[
            {
                'action_type': item['action_type'],
                'count': item['count'],
                'total_amount': float(item['total']),
                'avg_rate': float(item['avg_rate'])
            }
            for item in data['breakdown']
        ],
        period_start=data['period_start'],
        period_end=data['period_end']
    )


@router.get(
    "/organizations/{org_id}/members/{user_id}/actions",
    response_model=List[BillableActionDetail],
    summary="Get User's Billable Actions in Organization"
)
async def get_user_billable_actions_in_org(
    org_id: UUID,
    user_id: UUID,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    action_type: Optional[str] = Query(None),
    is_billed: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    Get detailed list of billable actions for user in organization.
    
    **Filters**: date range, action type, billed status
    **Use Case**: Detailed audit, invoice verification
    """
    @sync_to_async
    def fetch_actions(org_id, user_id, filters, limit, offset):
        from billing.models import BillableAction
        
        # Build query
        actions = BillableAction.objects.filter(
            organization_id=org_id,
            user_id=user_id
        ).select_related('project', 'invoice')
        
        if filters['start_date']:
            actions = actions.filter(created_at__date__gte=filters['start_date'])
        if filters['end_date']:
            actions = actions.filter(created_at__date__lte=filters['end_date'])
        if filters['action_type']:
            actions = actions.filter(action_type=filters['action_type'])
        if filters['is_billed'] is not None:
            if filters['is_billed']:
                actions = actions.filter(billed_at__isnull=False)
            else:
                actions = actions.filter(billed_at__isnull=True)
        
        total = actions.count()
        actions = actions.order_by('-created_at')[offset:offset+limit]
        
        return list(actions), total
    
    actions, total = await fetch_actions(
        org_id,
        user_id,
        {
            'start_date': start_date,
            'end_date': end_date,
            'action_type': action_type,
            'is_billed': is_billed
        },
        limit,
        offset
    )
    
    return [
        BillableActionDetail(
            action_id=action.action_id,
            action_type=action.action_type,
            quantity=action.quantity,
            unit_rate=action.unit_rate,
            total_amount=action.total_amount,
            is_billable=action.is_billable,
            billed_at=action.billed_at,
            invoice_number=action.invoice.invoice_number if action.invoice else None,
            created_at=action.created_at,
            project_id=action.project.project_id if action.project else None,
            project_name=action.project.name if action.project else None,
            metadata=action.metadata
        )
        for action in actions
    ]
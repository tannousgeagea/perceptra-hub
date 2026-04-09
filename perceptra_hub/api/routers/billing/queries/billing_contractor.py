

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
from api.routers.billing.schemas import ContractorListItem

router = APIRouter(prefix="/billing")


# ============================================================================
# CONTRACTOR MANAGEMENT ENDPOINTS
# ============================================================================

@router.get(
    "/organizations/{org_id}/contractors",
    response_model=List[ContractorListItem],
    summary="List All Contractors in Organization"
)
async def list_organization_contractors(
    org_id: UUID,
    billing_enabled: Optional[bool] = Query(None),
    has_unbilled_actions: Optional[bool] = Query(None),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    List all external contractors in organization with billing summary.
    
    **Use Case**: Contractor management dashboard, pending payments overview
    """
    @sync_to_async
    def fetch_contractors(org_id, filters):
        from memberships.models import OrganizationMembership
        from billing.models import BillableAction
        from django.db.models import Sum, Count
        from django.utils import timezone
        
        # Build query
        memberships = OrganizationMembership.objects.filter(
            organization_id=ctx.organization.pk,
            is_external_annotator=True,
            status='active'
        ).select_related('user')
        
        if filters['billing_enabled'] is not None:
            memberships = memberships.filter(billing_enabled=filters['billing_enabled'])
        
        contractors = []
        month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        for membership in memberships:
            # Get unbilled amount
            unbilled = BillableAction.objects.filter(
                user=membership.user,
                organization_id=org_id,
                billed_at__isnull=True
            ).aggregate(total=Sum('total_amount'))
            
            unbilled_amount = unbilled['total'] or Decimal('0.00')
            
            # Filter by has_unbilled_actions
            if filters['has_unbilled_actions'] is not None:
                if filters['has_unbilled_actions'] and unbilled_amount == 0:
                    continue
                if not filters['has_unbilled_actions'] and unbilled_amount > 0:
                    continue
            
            # Get this month's actions
            monthly_actions = BillableAction.objects.filter(
                user=membership.user,
                organization_id=org_id,
                created_at__gte=month_start
            ).count()
            
            contractors.append({
                'membership': membership,
                'unbilled_amount': unbilled_amount,
                'monthly_actions': monthly_actions
            })
        
        return contractors
    
    contractors = await fetch_contractors(
        org_id,
        {
            'billing_enabled': billing_enabled,
            'has_unbilled_actions': has_unbilled_actions
        }
    )
    
    return [
        ContractorListItem(
            user_id=c['membership'].user.id,
            username=c['membership'].user.username,
            full_name=c['membership'].user.get_full_name() or c['membership'].user.username,
            email=c['membership'].user.email,
            is_external_annotator=c['membership'].is_external_annotator,
            billing_enabled=c['membership'].billing_enabled,
            contractor_company=c['membership'].contractor_company,
            contract_start_date=c['membership'].contract_start_date,
            contract_end_date=c['membership'].contract_end_date,
            total_unbilled_amount=c['unbilled_amount'],
            total_actions_this_month=c['monthly_actions']
        )
        for c in contractors
    ]


@router.get(
    "/projects/{project_id}/contractors",
    response_model=List[ContractorListItem],
    summary="List All Contractors in Project"
)
async def list_project_contractors(
    project_id: UUID,
    billing_enabled: Optional[bool] = Query(None),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """List all external contractors working on specific project."""
    @sync_to_async
    def fetch_contractors(project_id, billing_enabled):
        from memberships.models import ProjectMembership
        from billing.models import BillableAction
        from projects.models import Project
        from django.db.models import Sum, Count
        from django.utils import timezone
        
        try:
            project = Project.objects.get(project_id=project_id)
        except Project.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Build query
        memberships = ProjectMembership.objects.filter(
            project=project,
            is_external_annotator=True
        ).select_related('user')
        
        if billing_enabled is not None:
            memberships = memberships.filter(billing_enabled=billing_enabled)
        
        contractors = []
        month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        for membership in memberships:
            # Get unbilled amount for this project
            unbilled = BillableAction.objects.filter(
                user=membership.user,
                project=project,
                billed_at__isnull=True
            ).aggregate(total=Sum('total_amount'))
            
            # Get this month's actions in this project
            monthly_actions = BillableAction.objects.filter(
                user=membership.user,
                project=project,
                created_at__gte=month_start
            ).count()
            
            contractors.append({
                'membership': membership,
                'unbilled_amount': unbilled['total'] or Decimal('0.00'),
                'monthly_actions': monthly_actions
            })
        
        return contractors
    
    contractors = await fetch_contractors(project_id, billing_enabled)
    
    return [
        ContractorListItem(
            user_id=c['membership'].user.id,
            username=c['membership'].user.username,
            full_name=c['membership'].user.get_full_name() or c['membership'].user.username,
            email=c['membership'].user.email,
            is_external_annotator=c['membership'].is_external_annotator,
            billing_enabled=c['membership'].billing_enabled,
            contractor_company=None,  # Project membership doesn't have this field
            contract_start_date=None,
            contract_end_date=None,
            total_unbilled_amount=c['unbilled_amount'],
            total_actions_this_month=c['monthly_actions']
        )
        for c in contractors
    ]


@router.get(
    "/organizations/{org_id}/billing-report",
    summary="Get Organization Billing Report"
)
async def get_organization_billing_report(
    org_id: UUID,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    group_by: str = Query('user', pattern='^(user|project|action_type|date)$'),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    Get comprehensive billing report for organization.
    
    **Group By**: user, project, action_type, date
    **Use Case**: Financial reporting, budget analysis
    """
    @sync_to_async
    def fetch_report(org_id, start_date, end_date, group_by):
        from billing.models import BillableAction
        from django.db.models import Sum, Count, Avg
        from django.db.models.functions import TruncDate
        
        # Build query
        actions = BillableAction.objects.filter(organization_id=org_id)
        
        if start_date:
            actions = actions.filter(created_at__date__gte=start_date)
        if end_date:
            actions = actions.filter(created_at__date__lte=end_date)
        
        # Group by requested dimension
        if group_by == 'user':
            report = actions.values(
                'user__id', 'user__username', 'user__first_name', 'user__last_name'
            ).annotate(
                total_actions=Count('action_id'),
                total_amount=Sum('total_amount'),
                billed_amount=Sum('total_amount', filter=Q(billed_at__isnull=False)),
                unbilled_amount=Sum('total_amount', filter=Q(billed_at__isnull=True)),
                avg_rate=Avg('unit_rate')
            ).order_by('-total_amount')
            
        elif group_by == 'project':
            report = actions.values(
                'project__id', 'project__project_id', 'project__name'
            ).annotate(
                total_actions=Count('action_id'),
                total_amount=Sum('total_amount'),
                billed_amount=Sum('total_amount', filter=Q(billed_at__isnull=False)),
                unbilled_amount=Sum('total_amount', filter=Q(billed_at__isnull=True))
            ).order_by('-total_amount')
            
        elif group_by == 'action_type':
            report = actions.values('action_type').annotate(
                total_actions=Count('action_id'),
                total_amount=Sum('total_amount'),
                avg_rate=Avg('unit_rate')
            ).order_by('-total_amount')
            
        elif group_by == 'date':
            report = actions.annotate(
                date=TruncDate('created_at')
            ).values('date').annotate(
                total_actions=Count('action_id'),
                total_amount=Sum('total_amount'),
                unique_users=Count('user_id', distinct=True)
            ).order_by('-date')
        
        # Overall summary
        summary = actions.aggregate(
            total_actions=Count('action_id'),
            total_amount=Sum('total_amount'),
            billed_amount=Sum('total_amount', filter=Q(billed_at__isnull=False)),
            unbilled_amount=Sum('total_amount', filter=Q(billed_at__isnull=True)),
            unique_users=Count('user_id', distinct=True),
            unique_projects=Count('project_id', distinct=True)
        )
        
        return {
            'summary': summary,
            'breakdown': list(report)
        }
    
    return await fetch_report(org_id, start_date, end_date, group_by)
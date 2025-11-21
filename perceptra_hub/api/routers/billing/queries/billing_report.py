# apps/billing/api/endpoints.py
from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import date, datetime
from pydantic import BaseModel
from decimal import Decimal
from asgiref.sync import sync_to_async

from api.dependencies import RequestContext, get_request_context

router = APIRouter(prefix="/billing")


class BillableActionSummary(BaseModel):
    action_type: str
    quantity: int
    unit_rate: float
    total_amount: Decimal


class BillingReport(BaseModel):
    period_start: date
    period_end: date
    total_actions: int
    total_amount: float
    currency: str
    breakdown: List[BillableActionSummary]


@router.get("/organizations/billing-report", response_model=BillingReport)
async def get_billing_report(
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ctx:RequestContext = Depends(get_request_context)
):
    """
    Get billing report for vendor organization.
    
    **Use Case**: Invoice generation, cost tracking
    """
    @sync_to_async
    def fetch_billing_report(org, project_id, user_id, start_date, end_date):
        from billing.models import BillableAction
        from django.db.models import Sum, Count, Avg
        
        # Build query
        actions = BillableAction.objects.filter(
            organization_id=org.id,
            is_billable=True,
            billed_at__isnull=True  # Unbilled actions
        )
        
        if project_id:
            actions = actions.filter(project__project_id=project_id)
        
        if user_id:
            actions = actions.filter(user_id=user_id)
        
        if start_date:
            actions = actions.filter(created_at__date__gte=start_date)
        
        if end_date:
            actions = actions.filter(created_at__date__lte=end_date)
        
        # Aggregate
        summary = actions.aggregate(
            total_actions=Count('action_id'),
            total_amount=Sum('total_amount')
        )
        
        # Breakdown by action type
        breakdown_data = actions.values('action_type').annotate(
            quantity=Count('action_id'),
            unit_rate=Avg('unit_rate'),
            total_amount=Sum('total_amount')
        )
        
        breakdown = [
            BillableActionSummary(
                action_type=item['action_type'],
                quantity=item['quantity'],
                unit_rate=item['unit_rate'],
                total_amount=float(item['total_amount'])
            )
            for item in breakdown_data
        ]
        
        return BillingReport(
            period_start=start_date or actions.earliest('created_at').created_at.date(),
            period_end=end_date or actions.latest('created_at').created_at.date(),
            total_actions=summary['total_actions'] or 0,
            total_amount=summary['total_amount'] or float('0.00'),
            currency='USD',
            breakdown=breakdown
        )
    
    return await fetch_billing_report(ctx.organization, project_id, user_id, start_date, end_date)
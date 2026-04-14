

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

router = APIRouter(prefix="/billing")

@router.post(
    "/organizations/{org_id}/members/{user_id}/billing/enable",
    summary="Enable Billing for User in Organization"
)
async def enable_org_member_billing(
    org_id: UUID,
    user_id: int,
    is_external: bool = Body(...),
    billing_enabled: bool = Body(...),
    rate_card_id: Optional[UUID] = Body(None),
    hourly_rate: Optional[Decimal] = Body(None),
    contractor_company: Optional[str] = Body(None),
    backfill: bool = Body(False),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """
    Enable billing for a user in a specific organization.
    
    **Use Case**: Mark student as external contractor in this org only.
    """
    @sync_to_async
    def enable(org_id, user_id, data):
        from memberships.models import OrganizationMembership
        from billing.models import BillingRateCard
        
        try:
            membership = OrganizationMembership.objects.get(
                user_id=user_id,
                organization_id=ctx.organization.pk,
            )
        except OrganizationMembership.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User is not a member of this organization"
            )
        
        membership.is_external_annotator = data['is_external']
        membership.billing_enabled = data['billing_enabled']
        membership.hourly_rate = data['hourly_rate']
        membership.contractor_company = data['contractor_company']
        
        if data['rate_card_id']:
            try:
                rate_card = BillingRateCard.objects.get(rate_card_id=data['rate_card_id'])
                membership.billing_rate_card = rate_card
            except BillingRateCard.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Rate card not found"
                )
        
        membership.save()
        
        task_id = None
        if data['backfill'] and data['billing_enabled']:
            from api.tasks.billing.billable_action import backfill_billing_for_user_in_org
            task = backfill_billing_for_user_in_org.delay(
                str(user_id),
                str(org_id)
            )
            task_id = task.id
        
        return membership, task_id
    
    membership, task_id = await enable(
        org_id,
        user_id,
        {
            'is_external': is_external,
            'billing_enabled': billing_enabled,
            'rate_card_id': rate_card_id,
            'hourly_rate': hourly_rate,
            'contractor_company': contractor_company,
            'backfill': backfill
        }
    )
    
    return {
        'message': 'User billing configuration updated for organization',
        'user_id': str(membership.user_id),
        'organization_id': str(membership.organization_id),
        'is_external_annotator': membership.is_external_annotator,
        'billing_enabled': membership.billing_enabled,
        'backfill_task_id': task_id
    }


@router.post(
    "/projects/{project_id}/members/{user_id}/billing/enable",
    summary="Enable Billing for User in Project"
)
async def enable_project_member_billing(
    project_id: UUID,
    user_id: int,
    is_external: bool = Body(...),
    billing_enabled: bool = Body(...),
    rate_card_id: Optional[UUID] = Body(None),
    hourly_rate: Optional[Decimal] = Body(None),
    backfill: bool = Body(False),
    ctx: RequestContext = Depends(require_permission('admin'))
):
    """Enable billing for user in specific project only."""
    @sync_to_async
    def enable(project_id, user_id, data):
        from memberships.models import ProjectMembership
        from billing.models import BillingRateCard
        
        try:
            membership = ProjectMembership.objects.get(
                user_id=user_id,
                project__project_id=project_id
            )
        except ProjectMembership.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User is not a member of this project"
            )
        
        membership.is_external_annotator = data['is_external']
        membership.billing_enabled = data['billing_enabled']
        membership.hourly_rate = data['hourly_rate']
        
        if data['rate_card_id']:
            try:
                rate_card = BillingRateCard.objects.get(rate_card_id=data['rate_card_id'])
                membership.billing_rate_card = rate_card
            except BillingRateCard.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Rate card not found"
                )
        
        membership.save()
        
        task_id = None
        if data['backfill'] and data['billing_enabled']:
            from api.tasks.billing.billable_action import backfill_billing_for_user_in_project
            task = backfill_billing_for_user_in_project.delay(
                str(user_id),
                membership.project.id
            )
            task_id = task.id
        
        return membership, task_id
    
    membership, task_id = await enable(
        project_id,
        user_id,
        {
            'is_external': is_external,
            'billing_enabled': billing_enabled,
            'rate_card_id': rate_card_id,
            'hourly_rate': hourly_rate,
            'backfill': backfill
        }
    )
    
    return {
        'message': 'User billing configuration updated for project',
        'user_id': str(membership.user_id),
        'project_id': str(membership.project.project_id),
        'is_external_annotator': membership.is_external_annotator,
        'billing_enabled': membership.billing_enabled,
        'backfill_task_id': task_id
    }
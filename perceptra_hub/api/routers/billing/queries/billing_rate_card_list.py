

# apps/billing/api/endpoints.py
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal
from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import RequestContext, get_request_context
from api.routers.billing.schemas import BillingRateCardOut

router = APIRouter(prefix="/billing")


@router.get(
    "/organizations/rate-cards",
    response_model=List[BillingRateCardOut],
    summary="List Rate Cards"
)
async def list_rate_cards(
    project_id: Optional[UUID] = None,
    is_active: Optional[bool] = None,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    List all billing rate cards for the organization.
    
    **Filters**:
    - project_id: Filter by project (null returns org-wide rate cards)
    - is_active: Filter by active status
    """
    @sync_to_async
    def fetch_rate_cards(org_id, project_id, is_active):
        from billing.models import BillingRateCard
        from organizations.models import Organization
        
        try:
            org = Organization.objects.get(id=org_id)
        except Organization.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Build query
        rate_cards = BillingRateCard.objects.filter(organization=org).select_related(
            'organization', 'project', 'created_by'
        )
        
        if project_id is not None:
            rate_cards = rate_cards.filter(project__project_id=project_id)
        
        if is_active is not None:
            rate_cards = rate_cards.filter(is_active=is_active)
        
        return list(rate_cards.order_by('-created_at'))
    
    rate_cards = await fetch_rate_cards(ctx.organization.id, project_id, is_active)
    
    return [
        BillingRateCardOut(
            rate_card_id=rc.rate_card_id,
            organization_id=rc.organization.org_id,
            organization_name=rc.organization.name,
            project_id=rc.project.project_id if rc.project else None,
            project_name=rc.project.name if rc.project else None,
            name=rc.name,
            currency=rc.currency,
            is_active=rc.is_active,
            rate_new_annotation=rc.rate_new_annotation,
            rate_untouched_prediction=rc.rate_untouched_prediction,
            rate_minor_edit=rc.rate_minor_edit,
            rate_major_edit=rc.rate_major_edit,
            rate_class_change=rc.rate_class_change,
            rate_deletion=rc.rate_deletion,
            rate_missed_object=rc.rate_missed_object,
            rate_image_review=rc.rate_image_review,
            rate_annotation_review=rc.rate_annotation_review,
            quality_bonus_threshold=rc.quality_bonus_threshold,
            quality_bonus_multiplier=rc.quality_bonus_multiplier,
            created_at=rc.created_at,
            updated_at=rc.updated_at,
            created_by_username=rc.created_by.username if rc.created_by else None
        )
        for rc in rate_cards
    ]

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
    "/rate-cards/{rate_card_id}",
    response_model=BillingRateCardOut,
    summary="Get Rate Card Details"
)
async def get_rate_card(
    rate_card_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get detailed information about a specific rate card.
    """
    @sync_to_async
    def fetch_rate_card(rate_card_id):
        from billing.models import BillingRateCard
        
        try:
            return BillingRateCard.objects.select_related(
                'organization', 'project', 'created_by'
            ).get(rate_card_id=rate_card_id)
        except BillingRateCard.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rate card not found"
            )
    
    rc = await fetch_rate_card(rate_card_id)
    
    return BillingRateCardOut(
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

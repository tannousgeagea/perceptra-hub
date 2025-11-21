
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal
from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import RequestContext, get_request_context
from api.routers.billing.schemas import BillingRateCardOut, BillingRateCardUpdate

router = APIRouter(prefix="/billing")


@router.patch(
    "/rate-cards/{rate_card_id}",
    response_model=BillingRateCardOut,
    summary="Update Rate Card"
)
async def update_rate_card(
    rate_card_id: UUID,
    rate_card_data: BillingRateCardUpdate,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Update an existing billing rate card (partial update).
    
    **Note**: Only provided fields will be updated.
    **Warning**: Changing rates affects future billing only, not existing billable actions.
    """
    @sync_to_async
    def update(rate_card_id, data):
        from billing.models import BillingRateCard
        
        try:
            rate_card = BillingRateCard.objects.select_related(
                'organization', 'project', 'created_by'
            ).get(rate_card_id=rate_card_id)
        except BillingRateCard.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rate card not found"
            )
        
        # Update fields
        update_fields = ['updated_at']
        
        if data.name is not None:
            # Check for duplicate name
            if BillingRateCard.objects.filter(
                organization=rate_card.organization,
                project=rate_card.project,
                name=data.name
            ).exclude(rate_card_id=rate_card_id).exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Rate card with this name already exists"
                )
            rate_card.name = data.name
            update_fields.append('name')
        
        if data.is_active is not None:
            rate_card.is_active = data.is_active
            update_fields.append('is_active')
        
        if data.currency is not None:
            rate_card.currency = data.currency
            update_fields.append('currency')
        
        # Update rates
        rate_fields = [
            'rate_new_annotation', 'rate_untouched_prediction', 'rate_minor_edit',
            'rate_major_edit', 'rate_class_change', 'rate_deletion', 'rate_missed_object',
            'rate_image_review', 'rate_annotation_review',
            'quality_bonus_threshold', 'quality_bonus_multiplier'
        ]
        
        for field in rate_fields:
            value = getattr(data, field)
            if value is not None:
                setattr(rate_card, field, value)
                update_fields.append(field)
        
        rate_card.save(update_fields=update_fields)
        return rate_card
    
    rc = await update(rate_card_id, rate_card_data)
    
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

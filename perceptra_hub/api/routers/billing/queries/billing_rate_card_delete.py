

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


@router.delete(
    "/rate-cards/{rate_card_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Rate Card"
)
async def delete_rate_card(
    rate_card_id: UUID,
    hard_delete: bool = False,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Delete a billing rate card.
    
    **Soft Delete (default)**: Sets is_active=False, preserves data
    **Hard Delete**: Permanently removes (fails if linked to billable actions)
    
    **Warning**: Cannot hard delete if billable actions reference this rate card.
    """
    @sync_to_async
    def delete(rate_card_id, hard_delete):
        from billing.models import BillingRateCard, BillableAction
        
        try:
            rate_card = BillingRateCard.objects.get(rate_card_id=rate_card_id)
        except BillingRateCard.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rate card not found"
            )
        
        if hard_delete:
            # Check if any billable actions reference this rate card
            if BillableAction.objects.filter(rate_card=rate_card).exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete rate card with associated billable actions. Use soft delete instead."
                )
            rate_card.delete()
        else:
            # Soft delete
            rate_card.is_active = False
            rate_card.save(update_fields=['is_active', 'updated_at'])
    
    await delete(rate_card_id, hard_delete)
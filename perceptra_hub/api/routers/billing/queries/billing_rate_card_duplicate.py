

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


@router.post(
    "/rate-cards/{rate_card_id}/duplicate",
    response_model=BillingRateCardOut,
    status_code=status.HTTP_201_CREATED,
    summary="Duplicate Rate Card"
)
async def duplicate_rate_card(
    rate_card_id: UUID,
    new_name: str = Body(..., embed=True),
    new_project_id: Optional[UUID] = Body(None, embed=True),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Duplicate an existing rate card with a new name and optionally different project scope.
    
    **Use Case**: Creating project-specific rates based on org defaults
    """
    @sync_to_async
    def duplicate(rate_card_id, new_name, new_project_id, user):
        from billing.models import BillingRateCard
        from projects.models import Project
        
        try:
            original = BillingRateCard.objects.get(rate_card_id=rate_card_id)
        except BillingRateCard.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rate card not found"
            )
        
        # Verify new project if specified
        new_project = None
        if new_project_id:
            try:
                new_project = Project.objects.get(
                    project_id=new_project_id,
                    organization=original.organization
                )
            except Project.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
        
        # Check for duplicate name
        if BillingRateCard.objects.filter(
            organization=original.organization,
            project=new_project,
            name=new_name
        ).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rate card with this name already exists"
            )
        
        # Create duplicate
        duplicate = BillingRateCard.objects.create(
            organization=original.organization,
            project=new_project,
            name=new_name,
            currency=original.currency,
            rate_new_annotation=original.rate_new_annotation,
            rate_untouched_prediction=original.rate_untouched_prediction,
            rate_minor_edit=original.rate_minor_edit,
            rate_major_edit=original.rate_major_edit,
            rate_class_change=original.rate_class_change,
            rate_deletion=original.rate_deletion,
            rate_missed_object=original.rate_missed_object,
            rate_image_review=original.rate_image_review,
            rate_annotation_review=original.rate_annotation_review,
            quality_bonus_threshold=original.quality_bonus_threshold,
            quality_bonus_multiplier=original.quality_bonus_multiplier,
            created_by=user
        )
        
        return duplicate
    
    rc = await duplicate(rate_card_id, new_name, new_project_id, ctx.user)
    
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
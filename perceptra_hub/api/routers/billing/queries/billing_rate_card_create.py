

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal
from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import RequestContext, get_request_context
from api.routers.billing.schemas import BillingRateCardOut, BillingRateCardCreate

router = APIRouter(prefix="/billing")



@router.post(
    "/organizations/rate-cards",
    response_model=BillingRateCardOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create Billing Rate Card"
)
async def create_rate_card(
    rate_card_data: BillingRateCardCreate,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Create a new billing rate card for the organization.
    
    **Permissions**: Organization admin or billing manager
    **Scope**: Can be org-wide (project_id=null) or project-specific
    """
    @sync_to_async
    def create(org, data, user):
        from billing.models import BillingRateCard
        from organizations.models import Organization
        from projects.models import Project
        
        
        # Verify project if specified
        project = None
        if data.project_id:
            try:
                project = Project.objects.get(
                    project_id=data.project_id,
                    organization=org
                )
            except Project.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found in this organization"
                )
        
        # Check for duplicate name
        if BillingRateCard.objects.filter(
            organization=org,
            project=project,
            name=data.name
        ).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rate card with this name already exists for this scope"
            )
        
        # Create rate card
        rate_card = BillingRateCard.objects.create(
            organization=org,
            project=project,
            name=data.name,
            currency=data.currency,
            rate_new_annotation=data.rate_new_annotation,
            rate_untouched_prediction=data.rate_untouched_prediction,
            rate_minor_edit=data.rate_minor_edit,
            rate_major_edit=data.rate_major_edit,
            rate_class_change=data.rate_class_change,
            rate_deletion=data.rate_deletion,
            rate_missed_object=data.rate_missed_object,
            rate_image_review=data.rate_image_review,
            rate_annotation_review=data.rate_annotation_review,
            quality_bonus_threshold=data.quality_bonus_threshold,
            quality_bonus_multiplier=data.quality_bonus_multiplier,
            created_by=user
        )
        
        return rate_card
    
    rate_card = await create(ctx.organization, rate_card_data, ctx.user)
    
    return BillingRateCardOut(
        rate_card_id=rate_card.rate_card_id,
        organization_id=rate_card.organization.org_id,
        organization_name=rate_card.organization.name,
        project_id=rate_card.project.project_id if rate_card.project else None,
        project_name=rate_card.project.name if rate_card.project else None,
        name=rate_card.name,
        currency=rate_card.currency,
        is_active=rate_card.is_active,
        rate_new_annotation=rate_card.rate_new_annotation,
        rate_untouched_prediction=rate_card.rate_untouched_prediction,
        rate_minor_edit=rate_card.rate_minor_edit,
        rate_major_edit=rate_card.rate_major_edit,
        rate_class_change=rate_card.rate_class_change,
        rate_deletion=rate_card.rate_deletion,
        rate_missed_object=rate_card.rate_missed_object,
        rate_image_review=rate_card.rate_image_review,
        rate_annotation_review=rate_card.rate_annotation_review,
        quality_bonus_threshold=rate_card.quality_bonus_threshold,
        quality_bonus_multiplier=rate_card.quality_bonus_multiplier,
        created_at=rate_card.created_at,
        updated_at=rate_card.updated_at,
        created_by_username=rate_card.created_by.username if rate_card.created_by else None
    )

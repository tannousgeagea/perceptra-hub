from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta, date
from pydantic import BaseModel, Field
from uuid import UUID

class BillableActionSummary(BaseModel):
    action_type: str
    quantity: int
    unit_rate: float
    total_amount: float


class BillingReport(BaseModel):
    period_start: date
    period_end: date
    total_actions: int
    total_amount: float
    currency: str
    breakdown: List[BillableActionSummary]
    
class BillingRateCardCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: Optional[UUID] = Field(None, description="Project-specific rates (null for org default)")
    currency: str = Field("USD", min_length=3, max_length=3)
    
    # Annotation rates
    rate_new_annotation: Decimal = Field(0.05, ge=0, description="Cost per new annotation")
    rate_untouched_prediction: Decimal = Field(0.02, ge=0, description="Cost per untouched AI prediction")
    rate_minor_edit: Decimal = Field(0.03, ge=0, description="Cost per minor edit")
    rate_major_edit: Decimal = Field(0.04, ge=0, description="Cost per major edit")
    rate_class_change: Decimal = Field(0.04, ge=0, description="Cost per class change")
    rate_deletion: Decimal = Field(0.02, ge=0, description="Cost per deletion")
    rate_missed_object: Decimal = Field(0.05, ge=0, description="Cost per missed object found")
    
    # Review ratesitem['total_amount']
    rate_image_review: Decimal = Field(0.10, ge=0, description="Cost per image reviewed")
    rate_annotation_review: Decimal = Field(0.01, ge=0, description="Cost per annotation reviewed")
    
    # Quality modifiers
    quality_bonus_threshold: Decimal = Field(95.0, ge=0, le=100, description="Quality % for bonus")
    quality_bonus_multiplier: Decimal = Field(1.1, ge=1.0, le=2.0, description="Bonus multiplier")


class BillingRateCardUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    is_active: Optional[bool] = None
    currency: Optional[str] = Field(None, min_length=3, max_length=3)
    
    # Annotation rates (all optional for partial updates)
    rate_new_annotation: Optional[Decimal] = Field(None, ge=0)
    rate_untouched_prediction: Optional[Decimal] = Field(None, ge=0)
    rate_minor_edit: Optional[Decimal] = Field(None, ge=0)
    rate_major_edit: Optional[Decimal] = Field(None, ge=0)
    rate_class_change: Optional[Decimal] = Field(None, ge=0)
    rate_deletion: Optional[Decimal] = Field(None, ge=0)
    rate_missed_object: Optional[Decimal] = Field(None, ge=0)
    
    # Review rates
    rate_image_review: Optional[Decimal] = Field(None, ge=0)
    rate_annotation_review: Optional[Decimal] = Field(None, ge=0)
    
    # Quality modifiers
    quality_bonus_threshold: Optional[Decimal] = Field(None, ge=0, le=100)
    quality_bonus_multiplier: Optional[Decimal] = Field(None, ge=1.0, le=2.0)


class BillingRateCardOut(BaseModel):
    rate_card_id: UUID
    organization_id: UUID
    organization_name: str
    project_id: Optional[UUID]
    project_name: Optional[str]
    
    name: str
    currency: str
    is_active: bool
    
    # Annotation rates
    rate_new_annotation: Decimal
    rate_untouched_prediction: Decimal
    rate_minor_edit: Decimal
    rate_major_edit: Decimal
    rate_class_change: Decimal
    rate_deletion: Decimal
    rate_missed_object: Decimal
    
    # Review rates
    rate_image_review: Decimal
    rate_annotation_review: Decimal
    
    # Quality modifiers
    quality_bonus_threshold: Decimal
    quality_bonus_multiplier: Decimal
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    created_by_username: Optional[str]
    
class InvoiceGenerateRequest(BaseModel):
    period_start: date = Field(..., description="Invoice period start date")
    period_end: date = Field(..., description="Invoice period end date")
    project_id: Optional[UUID] = Field(None, description="Generate for specific project only")
    client_organization_id: UUID = Field(..., description="Client organization being billed")
    tax_rate: Decimal = Field(0, ge=0, le=100, description="Tax rate percentage")
    notes: Optional[str] = Field(None, max_length=5000)
    due_days: int = Field(30, ge=1, le=365, description="Days until payment due")
    auto_issue: bool = Field(False, description="Automatically issue invoice after generation")


class InvoiceOut(BaseModel):
    invoice_id: UUID
    invoice_number: str
    vendor_organization_id: UUID
    vendor_organization_name: str
    client_organization_id: UUID
    client_organization_name: str
    project_id: Optional[UUID]
    project_name: Optional[str]
    
    # Period
    period_start: date
    period_end: date
    
    # Amounts
    subtotal: Decimal
    tax_rate: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    currency: str
    
    # Breakdown
    total_annotations: int
    total_reviews: int
    total_actions: int
    action_breakdown: Dict[str, Any]
    
    # Status
    status: str
    issued_at: Optional[datetime]
    due_date: Optional[date]
    paid_at: Optional[datetime]
    
    # Metadata
    notes: str
    created_at: datetime
    updated_at: datetime
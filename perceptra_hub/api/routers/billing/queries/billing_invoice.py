


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
from api.dependencies import RequestContext, get_request_context
from api.routers.billing.schemas import InvoiceOut, InvoiceGenerateRequest

router = APIRouter(prefix="/billing")


@router.post(
    "/organizations/invoices/generate",
    response_model=InvoiceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Invoice"
)
async def generate_invoice(
    org_id: UUID,
    invoice_data: InvoiceGenerateRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Generate an invoice from unbilled actions for a specific period.
    
    **Process**:
    1. Finds all unbilled `BillableAction` records in date range
    2. Applies quality multipliers if applicable
    3. Aggregates into line items by action type
    4. Creates `Invoice` record
    5. Marks actions as billed
    6. Optionally issues invoice immediately
    
    **Use Case**: Monthly/weekly billing cycles for annotation services
    """
    @sync_to_async
    def generate(vendor_org, data, user):
        from billing.models import Invoice, BillableAction, BillingRateCard
        from organizations.models import Organization
        from projects.models import Project
        from django.db import transaction
        from django.utils import timezone
        from decimal import Decimal
        
        # Verify client organization
        try:
            client_org = Organization.objects.get(id=data.client_organization_id)
        except Organization.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client organization not found"
            )
        
        # Verify project if specified
        project = None
        if data.project_id:
            try:
                project = Project.objects.get(
                    project_id=data.project_id,
                    organization=client_org
                )
            except Project.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found in client organization"
                )
        
        with transaction.atomic():
            # Find unbilled actions
            billable_actions = BillableAction.objects.select_for_update().filter(
                organization=vendor_org,
                is_billable=True,
                billed_at__isnull=True,
                created_at__date__gte=data.period_start,
                created_at__date__lte=data.period_end
            )
            
            if data.project_id:
                billable_actions = billable_actions.filter(project=project)
            
            if not billable_actions.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No unbilled actions found for the specified period"
                )
            
            # Calculate totals and breakdown
            from django.db.models import Sum, Count, Avg
            
            aggregates = billable_actions.aggregate(
                subtotal=Sum('total_amount'),
                total_actions=Count('action_id'),
                total_annotations=Count('action_id', filter=Q(
                    action_type__in=[
                        BillableAction.ActionType.NEW_ANNOTATION,
                        BillableAction.ActionType.UNTOUCHED_PREDICTION,
                        BillableAction.ActionType.MINOR_EDIT,
                        BillableAction.ActionType.MAJOR_EDIT,
                        BillableAction.ActionType.CLASS_CHANGE,
                        BillableAction.ActionType.DELETION,
                        BillableAction.ActionType.MISSED_OBJECT,
                    ]
                )),
                total_reviews=Count('action_id', filter=Q(
                    action_type__in=[
                        BillableAction.ActionType.IMAGE_REVIEW,
                        BillableAction.ActionType.ANNOTATION_REVIEW,
                    ]
                ))
            )
            
            subtotal = aggregates['subtotal'] or Decimal('0.00')
            
            # Calculate tax
            tax_amount = (subtotal * data.tax_rate / 100).quantize(Decimal('0.01'))
            total_amount = subtotal + tax_amount
            
            # Get breakdown by action type
            breakdown = billable_actions.values('action_type').annotate(
                count=Count('action_id'),
                total=Sum('total_amount'),
                avg_rate=Avg('unit_rate')
            ).order_by('-total')
            
            action_breakdown = {
                item['action_type']: {
                    'count': item['count'],
                    'total_amount': float(item['total']),
                    'avg_rate': float(item['avg_rate'])
                }
                for item in breakdown
            }
            
            # Generate invoice number
            invoice_count = Invoice.objects.filter(
                vendor_organization=vendor_org
            ).count()
            invoice_number = f"INV-{vendor_org.name[:3].upper()}-{invoice_count + 1:06d}"
            
            # Calculate due date
            due_date = timezone.now().date() + timedelta(days=data.due_days)
            
            # Create invoice
            invoice = Invoice.objects.create(
                invoice_number=invoice_number,
                vendor_organization=vendor_org,
                client_organization=client_org,
                project=project,
                period_start=data.period_start,
                period_end=data.period_end,
                subtotal=subtotal,
                tax_rate=data.tax_rate,
                tax_amount=tax_amount,
                total_amount=total_amount,
                currency=billable_actions.first().currency,
                total_annotations=aggregates['total_annotations'],
                total_reviews=aggregates['total_reviews'],
                total_actions=aggregates['total_actions'],
                status=Invoice.InvoiceStatus.DRAFT,
                due_date=due_date,
                notes=data.notes or '',
                metadata={
                    'action_breakdown': action_breakdown,
                    'generated_by': user.username,
                    'generated_at': timezone.now().isoformat()
                }
            )
            
            # Mark actions as billed
            now = timezone.now()
            billable_actions.update(
                billed_at=now,
                invoice=invoice
            )
            
            # Auto-issue if requested
            if data.auto_issue:
                invoice.status = Invoice.InvoiceStatus.PENDING
                invoice.issued_at = now
                invoice.save(update_fields=['status', 'issued_at', 'updated_at'])
            
            return invoice, action_breakdown
    
    invoice, action_breakdown = await generate(ctx.organization, invoice_data, ctx.user)
    
    return InvoiceOut(
        invoice_id=invoice.invoice_id,
        invoice_number=invoice.invoice_number,
        vendor_organization_id=invoice.vendor_organization.id,
        vendor_organization_name=invoice.vendor_organization.name,
        client_organization_id=invoice.client_organization.id,
        client_organization_name=invoice.client_organization.name,
        project_id=invoice.project.project_id if invoice.project else None,
        project_name=invoice.project.name if invoice.project else None,
        period_start=invoice.period_start,
        period_end=invoice.period_end,
        subtotal=invoice.subtotal,
        tax_rate=invoice.tax_rate,
        tax_amount=invoice.tax_amount,
        total_amount=invoice.total_amount,
        currency=invoice.currency,
        total_annotations=invoice.total_annotations,
        total_reviews=invoice.total_reviews,
        total_actions=invoice.total_actions,
        action_breakdown=action_breakdown,
        status=invoice.status,
        issued_at=invoice.issued_at,
        due_date=invoice.due_date,
        paid_at=invoice.paid_at,
        notes=invoice.notes,
        created_at=invoice.created_at,
        updated_at=invoice.updated_at
    )


@router.get(
    "/organizations/invoices",
    response_model=List[InvoiceOut],
    summary="List Invoices"
)
async def list_invoices(
    org_id: UUID,
    client_org_id: Optional[UUID] = None,
    project_id: Optional[UUID] = None,
    status: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    List invoices for vendor organization with filters.
    
    **Filters**:
    - client_org_id: Filter by client
    - project_id: Filter by project
    - status: Filter by invoice status
    - start_date/end_date: Filter by invoice period
    """
    @sync_to_async
    def fetch_invoices(org, filters, limit, offset):
        from billing.models import Invoice
        from organizations.models import Organization
        
        try:
            vendor_org = Organization.objects.get(id=org_id)
        except Organization.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Build query
        invoices = Invoice.objects.filter(
            vendor_organization=vendor_org
        ).select_related('vendor_organization', 'client_organization', 'project')
        
        if filters['client_org_id']:
            invoices = invoices.filter(client_organization_id=filters['client_org_id'])
        
        if filters['project_id']:
            invoices = invoices.filter(project__project_id=filters['project_id'])
        
        if filters['status']:
            invoices = invoices.filter(status=filters['status'])
        
        if filters['start_date']:
            invoices = invoices.filter(period_start__gte=filters['start_date'])
        
        if filters['end_date']:
            invoices = invoices.filter(period_end__lte=filters['end_date'])
        
        total_count = invoices.count()
        invoices = invoices.order_by('-created_at')[offset:offset+limit]
        
        return list(invoices), total_count
    
    invoices, total = await fetch_invoices(
        ctx.organization,
        {
            'client_org_id': client_org_id,
            'project_id': project_id,
            'status': status,
            'start_date': start_date,
            'end_date': end_date
        },
        limit,
        offset
    )
    
    return [
        InvoiceOut(
            invoice_id=inv.invoice_id,
            invoice_number=inv.invoice_number,
            vendor_organization_id=inv.vendor_organization.id,
            vendor_organization_name=inv.vendor_organization.name,
            client_organization_id=inv.client_organization.id,
            client_organization_name=inv.client_organization.name,
            project_id=inv.project.project_id if inv.project else None,
            project_name=inv.project.name if inv.project else None,
            period_start=inv.period_start,
            period_end=inv.period_end,
            subtotal=inv.subtotal,
            tax_rate=inv.tax_rate,
            tax_amount=inv.tax_amount,
            total_amount=inv.total_amount,
            currency=inv.currency,
            total_annotations=inv.total_annotations,
            total_reviews=inv.total_reviews,
            total_actions=inv.total_actions,
            action_breakdown=inv.metadata.get('action_breakdown', {}),
            status=inv.status,
            issued_at=inv.issued_at,
            due_date=inv.due_date,
            paid_at=inv.paid_at,
            notes=inv.notes,
            created_at=inv.created_at,
            updated_at=inv.updated_at
        )
        for inv in invoices
    ]


@router.get(
    "/invoices/{invoice_id}",
    response_model=InvoiceOut,
    summary="Get Invoice Details"
)
async def get_invoice(
    invoice_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """Get detailed information about a specific invoice."""
    @sync_to_async
    def fetch_invoice(invoice_id):
        from billing.models import Invoice
        
        try:
            return Invoice.objects.select_related(
                'vendor_organization', 'client_organization', 'project'
            ).get(invoice_id=invoice_id)
        except Invoice.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
    
    inv = await fetch_invoice(invoice_id)
    
    return InvoiceOut(
        invoice_id=inv.invoice_id,
        invoice_number=inv.invoice_number,
        vendor_organization_id=inv.vendor_organization.id,
        vendor_organization_name=inv.vendor_organization.name,
        client_organization_id=inv.client_organization.id,
        client_organization_name=inv.client_organization.name,
        project_id=inv.project.project_id if inv.project else None,
        project_name=inv.project.name if inv.project else None,
        period_start=inv.period_start,
        period_end=inv.period_end,
        subtotal=inv.subtotal,
        tax_rate=inv.tax_rate,
        tax_amount=inv.tax_amount,
        total_amount=inv.total_amount,
        currency=inv.currency,
        total_annotations=inv.total_annotations,
        total_reviews=inv.total_reviews,
        total_actions=inv.total_actions,
        action_breakdown=inv.metadata.get('action_breakdown', {}),
        status=inv.status,
        issued_at=inv.issued_at,
        due_date=inv.due_date,
        paid_at=inv.paid_at,
        notes=inv.notes,
        created_at=inv.created_at,
        updated_at=inv.updated_at
    )


@router.post(
    "/invoices/{invoice_id}/issue",
    response_model=InvoiceOut,
    summary="Issue Invoice"
)
async def issue_invoice(
    invoice_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Issue a draft invoice (changes status to PENDING).
    
    **Action**: Marks invoice as officially issued and ready for payment.
    **Side Effect**: Can trigger email notification to client.
    """
    @sync_to_async
    def issue(invoice_id):
        from billing.models import Invoice
        from django.utils import timezone
        
        try:
            invoice = Invoice.objects.select_related(
                'vendor_organization', 'client_organization', 'project'
            ).get(invoice_id=invoice_id)
        except Invoice.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        if invoice.status != Invoice.InvoiceStatus.DRAFT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot issue invoice with status: {invoice.status}"
            )
        
        invoice.status = Invoice.InvoiceStatus.PENDING
        invoice.issued_at = timezone.now()
        invoice.save(update_fields=['status', 'issued_at', 'updated_at'])
        
        # TODO: Send email notification to client
        
        return invoice
    
    inv = await issue(invoice_id)
    
    return InvoiceOut(
        invoice_id=inv.invoice_id,
        invoice_number=inv.invoice_number,
        vendor_organization_id=inv.vendor_organization.id,
        vendor_organization_name=inv.vendor_organization.name,
        client_organization_id=inv.client_organization.id,
        client_organization_name=inv.client_organization.name,
        project_id=inv.project.project_id if inv.project else None,
        project_name=inv.project.name if inv.project else None,
        period_start=inv.period_start,
        period_end=inv.period_end,
        subtotal=inv.subtotal,
        tax_rate=inv.tax_rate,
        tax_amount=inv.tax_amount,
        total_amount=inv.total_amount,
        currency=inv.currency,
        total_annotations=inv.total_annotations,
        total_reviews=inv.total_reviews,
        total_actions=inv.total_actions,
        action_breakdown=inv.metadata.get('action_breakdown', {}),
        status=inv.status,
        issued_at=inv.issued_at,
        due_date=inv.due_date,
        paid_at=inv.paid_at,
        notes=inv.notes,
        created_at=inv.created_at,
        updated_at=inv.updated_at
    )


@router.post(
    "/invoices/{invoice_id}/mark-paid",
    response_model=InvoiceOut,
    summary="Mark Invoice as Paid"
)
async def mark_invoice_paid(
    invoice_id: UUID,
    payment_date: Optional[datetime] = Body(None, embed=True),
    payment_notes: Optional[str] = Body(None, embed=True),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Mark an invoice as paid.
    
    **Action**: Updates status to PAID with payment timestamp.
    **Use Case**: Manual payment confirmation, bank transfer received, etc.
    """
    @sync_to_async
    def mark_paid(invoice_id, payment_date, payment_notes):
        from billing.models import Invoice
        from django.utils import timezone
        
        try:
            invoice = Invoice.objects.select_related(
                'vendor_organization', 'client_organization', 'project'
            ).get(invoice_id=invoice_id)
        except Invoice.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        if invoice.status not in [Invoice.InvoiceStatus.PENDING, Invoice.InvoiceStatus.DRAFT]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot mark invoice as paid with status: {invoice.status}"
            )
        
        invoice.status = Invoice.InvoiceStatus.PAID
        invoice.paid_at = payment_date or timezone.now()
        
        if payment_notes:
            invoice.metadata['payment_notes'] = payment_notes
        
        invoice.save(update_fields=['status', 'paid_at', 'metadata', 'updated_at'])
        
        return invoice
    
    inv = await mark_paid(invoice_id, payment_date, payment_notes)
    
    return InvoiceOut(
        invoice_id=inv.invoice_id,
        invoice_number=inv.invoice_number,
        vendor_organization_id=inv.vendor_organization.id,
        vendor_organization_name=inv.vendor_organization.name,
        client_organization_id=inv.client_organization.id,
        client_organization_name=inv.client_organization.name,
        project_id=inv.project.project_id if inv.project else None,
        project_name=inv.project.name if inv.project else None,
        period_start=inv.period_start,
        period_end=inv.period_end,
        subtotal=inv.subtotal,
        tax_rate=inv.tax_rate,
        tax_amount=inv.tax_amount,
        total_amount=inv.total_amount,
        currency=inv.currency,
        total_annotations=inv.total_annotations,
        total_reviews=inv.total_reviews,
        total_actions=inv.total_actions,
        action_breakdown=inv.metadata.get('action_breakdown', {}),
        status=inv.status,
        issued_at=inv.issued_at,
        due_date=inv.due_date,
        paid_at=inv.paid_at,
        notes=inv.notes,
        created_at=inv.created_at,
        updated_at=inv.updated_at
    )


@router.delete(
    "/invoices/{invoice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel Invoice"
)
async def cancel_invoice(
    invoice_id: UUID,
    reason: Optional[str] = Body(None, embed=True),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Cancel an invoice (only DRAFT or PENDING).
    
    **Action**: Sets status to CANCELLED, unbills associated actions.
    **Use Case**: Invoice error, client dispute, billing correction.
    """
    @sync_to_async
    def cancel(invoice_id, reason):
        from billing.models import Invoice, BillableAction
        from django.db import transaction
        
        try:
            invoice = Invoice.objects.get(invoice_id=invoice_id)
        except Invoice.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        if invoice.status == Invoice.InvoiceStatus.PAID:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel a paid invoice"
            )
        
        with transaction.atomic():
            # Unbill actions
            BillableAction.objects.filter(invoice=invoice).update(
                billed_at=None,
                invoice=None
            )
            
            # Cancel invoice
            invoice.status = Invoice.InvoiceStatus.CANCELLED
            if reason:
                invoice.metadata['cancellation_reason'] = reason
                invoice.metadata['cancelled_at'] = timezone.now().isoformat()
            invoice.save(update_fields=['status', 'metadata', 'updated_at'])
    
    await cancel(invoice_id, reason)
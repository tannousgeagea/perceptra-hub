# apps/billing/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _
from organizations.models import Organization
from projects.models import Project
from django.contrib.auth import get_user_model
import uuid

User = get_user_model()


class BillingRateCard(models.Model):
    """
    Pricing structure for annotation services.
    Can be org-level (vendor's standard rates) or project-specific.
    """
    rate_card_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    
    # Scope
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='billing_rate_cards',
        help_text=_('Vendor organization providing services')
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='billing_rate_cards',
        help_text=_('Project-specific rates (null = default rates)')
    )
    
    name = models.CharField(max_length=255, help_text=_('Rate card name'))
    currency = models.CharField(max_length=3, default='USD')
    is_active = models.BooleanField(default=True, db_index=True)
    
    # Annotation Rates (per action)
    rate_new_annotation = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.05,
        help_text=_('Cost per new annotation created from scratch')
    )
    rate_untouched_prediction = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.02,
        help_text=_('Cost per AI prediction accepted without changes')
    )
    rate_minor_edit = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.03,
        help_text=_('Cost per minor edit to prediction (resize, small adjustment)')
    )
    rate_major_edit = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.04,
        help_text=_('Cost per major edit to prediction (significant change)')
    )
    rate_class_change = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.04,
        help_text=_('Cost per class label change')
    )
    rate_deletion = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.02,
        help_text=_('Cost per annotation deleted (removing false positive)')
    )
    rate_missed_object = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.05,
        help_text=_('Cost per missed object added (finding false negative)')
    )
    
    # Review Rates
    rate_image_review = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.10,
        help_text=_('Cost per image reviewed')
    )
    rate_annotation_review = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.01,
        help_text=_('Cost per annotation reviewed')
    )
    
    # Bonus/Penalty Modifiers
    quality_bonus_threshold = models.DecimalField(
        max_digits=5, decimal_places=2, default=95.0,
        help_text=_('Quality score % for bonus eligibility')
    )
    quality_bonus_multiplier = models.DecimalField(
        max_digits=5, decimal_places=2, default=1.1,
        help_text=_('Multiplier when quality threshold met (1.1 = 10% bonus)')
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        db_table = 'billing_rate_card'
        unique_together = [('organization', 'project', 'name')]
        indexes = [
            models.Index(fields=['organization', 'is_active']),
            models.Index(fields=['project', 'is_active']),
        ]

    def __str__(self):
        scope = f"Project: {self.project.name}" if self.project else "Default"
        return f"{self.name} - {scope}"


class BillableAction(models.Model):
    """
    Individual billable action derived from ActivityEvent.
    One ActivityEvent may generate multiple BillableActions.
    """
    
    class ActionType(models.TextChoices):
        NEW_ANNOTATION = 'new_annotation', _('New Annotation')
        UNTOUCHED_PREDICTION = 'untouched_prediction', _('Untouched Prediction')
        MINOR_EDIT = 'minor_edit', _('Minor Edit')
        MAJOR_EDIT = 'major_edit', _('Major Edit')
        CLASS_CHANGE = 'class_change', _('Class Change')
        DELETION = 'deletion', _('Deletion')
        MISSED_OBJECT = 'missed_object', _('Missed Object')
        IMAGE_REVIEW = 'image_review', _('Image Review')
        ANNOTATION_REVIEW = 'annotation_review', _('Annotation Review')
    
    action_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False, db_index=True)
    
    # References
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    activity_event = models.ForeignKey(
        'activity.ActivityEvent',
        on_delete=models.SET_NULL,
        null=True,
        related_name='billable_actions'
    )
    rate_card = models.ForeignKey(
        BillingRateCard,
        on_delete=models.PROTECT,
        related_name='billable_actions'
    )
    
    # Action Details
    action_type = models.CharField(max_length=30, choices=ActionType.choices, db_index=True)
    quantity = models.PositiveIntegerField(default=1, help_text=_('Number of units (annotations, reviews, etc)'))
    
    # Pricing
    unit_rate = models.DecimalField(max_digits=10, decimal_places=4)
    subtotal = models.DecimalField(max_digits=12, decimal_places=4)
    quality_multiplier = models.DecimalField(max_digits=5, decimal_places=2, default=1.0)
    total_amount = models.DecimalField(max_digits=12, decimal_places=4)
    currency = models.CharField(max_length=3, default='USD')
    
    # Metadata
    metadata = models.JSONField(
        default=dict,
        help_text=_('Additional context (annotation_id, image_id, edit_magnitude, etc)')
    )
    
    # Billing Status
    is_billable = models.BooleanField(
        default=True,
        db_index=True,
        help_text=_('Can be excluded if action is free/internal')
    )
    billed_at = models.DateTimeField(null=True, blank=True, db_index=True)
    invoice = models.ForeignKey(
        'Invoice',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='billable_actions'
    )
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'billable_action'
        indexes = [
            models.Index(fields=['organization', 'created_at']),
            models.Index(fields=['project', 'created_at']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['is_billable', 'billed_at']),
            models.Index(fields=['action_type', 'created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.action_type} - {self.user.username if self.user else 'Unknown'} - ${self.total_amount}"
    
    def save(self, *args, **kwargs):
        from decimal import Decimal
    
     # Ensure quality_multiplier is Decimal
        if isinstance(self.quality_multiplier, float):
            self.quality_multiplier = Decimal(str(self.quality_multiplier))
        
        self.subtotal = self.unit_rate * self.quantity
        self.total_amount = self.subtotal * self.quality_multiplier
        super().save(*args, **kwargs)


class Invoice(models.Model):
    """
    Aggregated invoice for billing period.
    """
    
    class InvoiceStatus(models.TextChoices):
        DRAFT = 'draft', _('Draft')
        PENDING = 'pending', _('Pending')
        PAID = 'paid', _('Paid')
        CANCELLED = 'cancelled', _('Cancelled')
    
    invoice_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    invoice_number = models.CharField(max_length=50, unique=True)
    
    # Parties
    vendor_organization = models.ForeignKey(
        Organization,
        on_delete=models.PROTECT,
        related_name='invoices_issued',
        help_text=_('Vendor issuing invoice')
    )
    client_organization = models.ForeignKey(
        Organization,
        on_delete=models.PROTECT,
        related_name='invoices_received',
        help_text=_('Client receiving invoice')
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='invoices'
    )
    
    # Period
    period_start = models.DateField(db_index=True)
    period_end = models.DateField(db_index=True)
    
    # Amounts
    subtotal = models.DecimalField(max_digits=12, decimal_places=2)
    tax_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    tax_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_amount = models.DecimalField(max_digits=12, decimal_places=2)
    currency = models.CharField(max_length=3, default='USD')
    
    # Breakdown
    total_annotations = models.PositiveIntegerField(default=0)
    total_reviews = models.PositiveIntegerField(default=0)
    total_actions = models.PositiveIntegerField(default=0)
    
    # Status
    status = models.CharField(max_length=20, choices=InvoiceStatus.choices, default=InvoiceStatus.DRAFT)
    issued_at = models.DateTimeField(null=True, blank=True)
    due_date = models.DateField(null=True, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    notes = models.TextField(blank=True)
    metadata = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'invoice'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['vendor_organization', 'period_start']),
            models.Index(fields=['client_organization', 'status']),
            models.Index(fields=['status', 'due_date']),
        ]

    def __str__(self):
        return f"{self.invoice_number} - {self.vendor_organization.name}"
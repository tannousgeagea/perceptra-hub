from django.contrib import admin
from unfold.admin import ModelAdmin
from django.db.models import Sum, Count
from django.utils.html import format_html
from django.utils.timezone import now

from .models import BillingRateCard, BillableAction, Invoice


# =========================
# Billing Rate Card Admin
# =========================
@admin.register(BillingRateCard)
class BillingRateCardAdmin(ModelAdmin):
    list_display = (
        'name',
        'organization',
        'project',
        'currency',
        'is_active',
        'annotation_rates_summary',
        'review_rates_summary',
        'updated_at'
    )

    list_filter = (
        'organization',
        'project',
        'currency',
        'is_active',
    )

    search_fields = ('name', 'organization__name', 'project__name')

    readonly_fields = ('created_at', 'updated_at')

    fieldsets = (
        ("Scope", {
            'fields': ('organization', 'project', 'name', 'currency', 'is_active')
        }),

        ("Annotation Rates", {
            'fields': (
                'rate_new_annotation',
                'rate_untouched_prediction',
                'rate_minor_edit',
                'rate_major_edit',
                'rate_class_change',
                'rate_deletion',
                'rate_missed_object',
            )
        }),

        ("Review Rates", {
            'fields': (
                'rate_image_review',
                'rate_annotation_review',
            )
        }),

        ("Quality Bonus", {
            'fields': (
                'quality_bonus_threshold',
                'quality_bonus_multiplier',
            )
        }),

        ("Metadata", {
            'fields': ('created_at', 'updated_at', 'created_by'),
        }),
    )

    def annotation_rates_summary(self, obj):
        return f"${obj.rate_new_annotation} / ${obj.rate_minor_edit} / ${obj.rate_major_edit}"
    annotation_rates_summary.short_description = "Annotation Rates"

    def review_rates_summary(self, obj):
        return f"Img: ${obj.rate_image_review} | Ann: ${obj.rate_annotation_review}"
    review_rates_summary.short_description = "Review Rates"


# =========================
# Billable Action Admin
# =========================
@admin.register(BillableAction)
class BillableActionAdmin(ModelAdmin):
    list_display = (
        'action_id_short',
        'action_type',
        'organization',
        'project',
        'user',
        'quantity',
        'unit_rate',
        'total_amount_colored',
        'quality_multiplier',
        'is_billable',
        'billed_status',
        'created_at'
    )

    list_filter = (
        'action_type',
        'organization',
        'project',
        'is_billable',
        'billed_at',
        'created_at'
    )

    search_fields = (
        'action_id',
        'user__username',
        'project__name',
    )

    readonly_fields = (
        'subtotal',
        'total_amount',
        'created_at'
    )

    date_hierarchy = 'created_at'

    def action_id_short(self, obj):
        return str(obj.action_id)[:8]
    action_id_short.short_description = "ID"

    def total_amount_colored(self, obj):
        color = "green" if obj.total_amount > 0 else "red"
        return format_html(
            '<span style="color:{}; font-weight:bold;">${}</span>',
            color,
            obj.total_amount
        )
    total_amount_colored.short_description = "Total"

    def billed_status(self, obj):
        if obj.billed_at:
            return format_html('<span style="color:green;">✔ Billed</span>')
        return format_html('<span style="color:orange;">Pending</span>')
    billed_status.short_description = "Billing"

    actions = ['mark_as_billed']

    def mark_as_billed(self, request, queryset):
        updated = queryset.update(billed_at=now())
        self.message_user(request, f"{updated} actions marked as billed.")
    mark_as_billed.short_description = "Mark selected as billed"


# =========================
# Invoice Admin
# =========================
@admin.register(Invoice)
class InvoiceAdmin(ModelAdmin):
    list_display = (
        'invoice_number',
        'vendor_organization',
        'client_organization',
        'project',
        'period_start',
        'period_end',
        'status_badge',
        'total_amount_colored',
        'due_date',
        'created_at'
    )

    list_filter = (
        'status',
        'vendor_organization',
        'client_organization',
        'project',
        'period_start',
    )

    search_fields = (
        'invoice_number',
        'vendor_organization__name',
        'client_organization__name',
    )

    readonly_fields = (
        'subtotal',
        'tax_amount',
        'total_amount',
        'created_at',
        'updated_at'
    )

    date_hierarchy = 'period_start'

    fieldsets = (
        ("Invoice Info", {
            'fields': ('invoice_number', 'status')
        }),

        ("Parties", {
            'fields': ('vendor_organization', 'client_organization', 'project')
        }),

        ("Period", {
            'fields': ('period_start', 'period_end')
        }),

        ("Amounts", {
            'fields': ('subtotal', 'tax_rate', 'tax_amount', 'total_amount', 'currency')
        }),

        ("Breakdown", {
            'fields': ('total_annotations', 'total_reviews', 'total_actions')
        }),

        ("Lifecycle", {
            'fields': ('issued_at', 'due_date', 'paid_at')
        }),

        ("Notes", {
            'fields': ('notes', 'metadata')
        }),
    )

    def total_amount_colored(self, obj):
        return format_html(
            '<strong style="color:#2E86C1;">${}</strong>',
            obj.total_amount
        )
    total_amount_colored.short_description = "Total"

    def status_badge(self, obj):
        colors = {
            'draft': '#7f8c8d',
            'pending': '#f39c12',
            'paid': '#27ae60',
            'cancelled': '#c0392b'
        }
        color = colors.get(obj.status, 'black')
        return format_html(
            '<span style="color:white; background:{}; padding:4px 8px; border-radius:6px;">{}</span>',
            color,
            obj.status.upper()
        )
    status_badge.short_description = "Status"
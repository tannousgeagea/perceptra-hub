from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import BillingRateCard, BillableAction, Invoice
# Register your models here.

@admin.register(BillableAction)
class BillableActionAdmin(ModelAdmin):
    list_display = ['id', 'organization', 'project', "activity_event"]
from django.db import models

# Create your models here.
from django.db import models
from metadata.models import (
    Language
)

# Create your models here.
class Tenant(models.Model):
    tenant_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    default_language = models.ForeignKey(Language, models.RESTRICT)
    is_active = models.BooleanField(default=True, help_text="Indicates if the filter is currently active.")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'tenant'
        verbose_name = "Tenant"
        verbose_name_plural = "Tenants"
        ordering = ['name']
        constraints = [
            models.UniqueConstraint(fields=['name'], name='unique_tenant_name_constraint')
        ]
        
    def __str__(self):
        return f"{self.name}"
    
class Plant(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='plants')
    plant_name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    language = models.ForeignKey(Language, models.RESTRICT)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "plant"
        verbose_name = "Plant"
        verbose_name_plural = "Plants"
        ordering = ['plant_name']
        unique_together = ('tenant', 'plant_name')
        
    def __str__(self):
        return f'{self.plant_name} - {self.location}'
    
class Domain(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='domains')
    domain_name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'domain'
        verbose_name = "Domain"
        verbose_name_plural = "Domains"
        ordering = ['domain_name']
        constraints = [
            models.UniqueConstraint(fields=['domain_name'], name='unique_domain_constraint')
        ]

    def __str__(self):
        return self.domain_name
    
class EdgeBox(models.Model):
    edge_box_id = models.CharField(max_length=255, unique=True)
    edge_box_location = models.CharField(max_length=255)
    plant = models.ForeignKey(Plant, on_delete=models.RESTRICT)  # Assuming you have a Tenant model for multi-tenancy
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'edge_box'
        verbose_name_plural = 'Edge Boxes'

    def __str__(self):
        return f"{self.edge_box_id} - {self.edge_box_location}"

class SensorBox(models.Model):
    edge_box = models.ForeignKey(EdgeBox, on_delete=models.RESTRICT)
    sensor_box_name = models.CharField(max_length=255)
    sensor_box_location = models.CharField(max_length=255)  # E.g., 'front', 'top', 'bunker'
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    meta_info = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = 'sensor_box'
        unique_together = ('edge_box', 'sensor_box_name')
        verbose_name_plural = 'Sensor Boxes'

    def __str__(self):
        return f"{self.sensor_box_name} at {self.sensor_box_location} ({self.edge_box})"
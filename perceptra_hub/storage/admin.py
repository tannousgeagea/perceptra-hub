"""
Django admin configuration for storage management.
"""
from django.contrib import admin
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from django.urls import reverse
from django.utils.safestring import mark_safe
from django import forms
from unfold.admin import ModelAdmin
from .models import StorageProfile, SecretRef, EncryptedSecret

from .services import (
    test_storage_profile_connection,
    create_encrypted_secret,
    update_encrypted_secret
)

@admin.register(SecretRef)
class SecretRefAdmin(ModelAdmin):
    """Admin interface for SecretRef model."""
    
    list_display = [
        'id',
        'organization',
        'provider',
        'path',
        'key',
        'created_at'
    ]
    list_filter = ['provider', 'created_at']
    search_fields = ['organization__name', 'path', 'key']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('id', 'organization', 'provider')
        }),
        (_('Secret Location'), {
            'fields': ('path', 'key', 'metadata')
        }),
        (_('Timestamps'), {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def has_delete_permission(self, request, obj=None):
        """Prevent deletion if referenced by storage profiles."""
        if obj and obj.storage_profiles.exists():
            return False
        return super().has_delete_permission(request, obj)


@admin.register(StorageProfile)
class StorageProfileAdmin(ModelAdmin):
    """Admin interface for StorageProfile model."""
    
    list_display = [
        'name',
        'organization',
        'backend',
        'region',
        'default_badge',
        'active_badge',
        'created_at'
    ]
    list_filter = [
        'backend',
        'is_default',
        'is_active',
        'created_at'
    ]
    search_fields = [
        'name',
        'organization__name',
        'config'
    ]
    readonly_fields = [
        'id',
        'created_at',
        'updated_at',
        'config_display'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('id', 'organization', 'name', 'backend', 'region')
        }),
        (_('Configuration'), {
            'fields': ('config', 'config_display', 'credential_ref')
        }),
        (_('Status'), {
            'fields': ('is_default', 'is_active')
        }),
        (_('Timestamps'), {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def default_badge(self, obj):
        """Display default status as badge."""
        if obj.is_default:
            return format_html(
                '<span style="background-color: #28a745; color: white; '
                'padding: 3px 10px; border-radius: 3px;">DEFAULT</span>'
            )
        return format_html(
            '<span style="color: #6c757d;">—</span>'
        )
    default_badge.short_description = _('Default')

    def active_badge(self, obj):
        """Display active status as badge."""
        if obj.is_active:
            return format_html(
                '<span style="background-color: #17a2b8; color: white; '
                'padding: 3px 10px; border-radius: 3px;">ACTIVE</span>'
            )
        return format_html(
            '<span style="background-color: #dc3545; color: white; '
            'padding: 3px 10px; border-radius: 3px;">INACTIVE</span>'
        )
    active_badge.short_description = _('Status')

    def config_display(self, obj):
        """Display sanitized configuration."""
        import json
        safe_config = obj.get_config_display()
        return format_html(
            '<pre style="background: #f5f5f5; padding: 10px; '
            'border-radius: 4px;">{}</pre>',
            json.dumps(safe_config, indent=2)
        )
    config_display.short_description = _('Configuration (Safe)')

    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related('organization', 'credential_ref')

    def save_model(self, request, obj, form, change):
        """Override save to handle validation errors gracefully."""
        try:
            super().save_model(request, obj, form, change)
        except Exception as e:
            self.message_user(
                request,
                f"Error saving storage profile: {str(e)}",
                level='ERROR'
            )
            raise
        
class EncryptedSecretAdminForm(forms.ModelForm):
    """
    Custom form for EncryptedSecret admin.
    Provides a secure way to input secret data without displaying encrypted values.
    """
    secret_data = forms.JSONField(
        required=False,
        widget=forms.Textarea(attrs={
            'rows': 10,
            'cols': 80,
            'placeholder': '{\n  "access_key": "your-key",\n  "secret_key": "your-secret"\n}'
        }),
        help_text=_(
            'Enter the secret data as JSON. This will be encrypted before saving. '
            'Leave empty when editing to keep existing encrypted value.'
        )
    )
    
    class Meta:
        model = EncryptedSecret
        fields = ['organization', 'identifier', 'description', 'secret_data']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't show the encrypted_value field in the form
        if 'encrypted_value' in self.fields:
            del self.fields['encrypted_value']
        
        # If editing existing secret, make secret_data optional
        if self.instance.pk:
            self.fields['secret_data'].required = False
            self.fields['secret_data'].help_text = _(
                'Enter new secret data as JSON to update the secret. '
                'Leave empty to keep the existing encrypted value.'
            )
    
    def clean_secret_data(self):
        """Validate that secret_data is valid JSON."""
        secret_data = self.cleaned_data.get('secret_data')
        
        # If creating new secret, secret_data is required
        if not self.instance.pk and not secret_data:
            raise forms.ValidationError(_('Secret data is required when creating a new secret.'))
        
        return secret_data
    
    def save(self, commit=True):
        """Override save to encrypt the secret data."""
        instance = super().save(commit=False)
        secret_data = self.cleaned_data.get('secret_data')
        
        # Only update if new secret data is provided
        if secret_data:
            if self.instance.pk:
                # Update existing secret
                instance = update_encrypted_secret(
                    organization=instance.organization,
                    identifier=instance.identifier,
                    secret_data=secret_data
                )
            else:
                # Create new secret
                if commit:
                    instance = create_encrypted_secret(
                        organization=instance.organization,
                        identifier=instance.identifier,
                        secret_data=secret_data,
                        description=instance.description
                    )
                else:
                    # Just set the fields, encryption will happen on save
                    from .services import settings
                    from cryptography.fernet import Fernet
                    import json
                    
                    encryption_key = getattr(settings, 'SECRET_ENCRYPTION_KEY', None)
                    if not encryption_key:
                        raise ValueError("SECRET_ENCRYPTION_KEY not configured")
                    
                    fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                    secret_string = json.dumps(secret_data)
                    encrypted_bytes = fernet.encrypt(secret_string.encode())
                    instance.encrypted_value = encrypted_bytes.decode('utf-8')
        elif commit:
            instance.save()
        
        return instance
        

@admin.register(EncryptedSecret)
class EncryptedSecretAdmin(ModelAdmin):
    """Admin interface for managing encrypted secrets."""
    
    form = EncryptedSecretAdminForm
    
    list_display = [
        'identifier',
        'organization',
        'description_preview',
        'has_value',
        'reference_count',
        'created_at',
        'updated_at',
        'actions_column'
    ]
    
    list_filter = [
        'organization',
        'created_at',
        'updated_at'
    ]
    
    search_fields = [
        'identifier',
        'description',
        'organization__name'
    ]
    
    readonly_fields = [
        'id',
        'created_at',
        'updated_at',
        'created_by',
        'updated_by',
        'encrypted_value_preview',
        'reference_count_detail'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('organization', 'identifier', 'description')
        }),
        (_('Secret Data'), {
            'fields': ('secret_data',),
            'description': _(
                '<strong>Security Note:</strong> Secret data is encrypted before storage. '
                'The encryption key is stored in environment variables, never in the database.'
            )
        }),
        (_('Metadata'), {
            'fields': ('id', 'encrypted_value_preview', 'reference_count_detail', 
                      'created_at', 'updated_at', 'created_by', 'updated_by'),
            'classes': ('collapse',)
        }),
    )
    
    def description_preview(self, obj):
        """Show truncated description."""
        if obj.description:
            return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
        return '-'
    description_preview.short_description = _('Description')
    
    def has_value(self, obj):
        """Indicate whether the secret has an encrypted value."""
        if obj.encrypted_value:
            return format_html(
                '<span style="color: green;">✓ Encrypted</span>'
            )
        return format_html(
            '<span style="color: red;">✗ No Value</span>'
        )
    has_value.short_description = _('Status')
    
    def reference_count(self, obj):
        """Show count of SecretRefs pointing to this secret."""
        count = SecretRef.objects.filter(
            organization=obj.organization,
            provider='local_enc',
            path=obj.identifier
        ).count()
        
        if count > 0:
            return format_html(
                '<span style="color: blue;">{} reference(s)</span>',
                count
            )
        return format_html(
            '<span style="color: gray;">0</span>'
        )
    reference_count.short_description = _('References')
    
    def reference_count_detail(self, obj):
        """Show detailed list of SecretRefs."""
        if not obj.pk:
            return '-'
        
        refs = SecretRef.objects.filter(
            organization=obj.organization,
            provider='local_enc',
            path=obj.identifier
        )
        
        if not refs.exists():
            return _('No SecretRefs pointing to this secret')
        
        html = '<ul>'
        for ref in refs:
            admin_url = reverse('admin:storage_secretref_change', args=[ref.id])
            html += f'<li><a href="{admin_url}">{ref.provider}:{ref.path}/{ref.key}</a></li>'
        html += '</ul>'
        
        return mark_safe(html)
    reference_count_detail.short_description = _('Secret References')
    
    def encrypted_value_preview(self, obj):
        """Show preview of encrypted value (truncated for security)."""
        if not obj.encrypted_value:
            return '-'
        
        # Show only first 32 characters
        preview = obj.encrypted_value[:32] + '...'
        return format_html(
            '<code style="background: #f5f5f5; padding: 5px; font-size: 11px;">{}</code><br>'
            '<small style="color: #666;">Length: {} bytes</small>',
            preview,
            len(obj.encrypted_value)
        )
    encrypted_value_preview.short_description = _('Encrypted Value (Preview)')
    
    def actions_column(self, obj):
        """Custom actions column."""
        if not obj.pk:
            return '-'
        
        return format_html(
            '<a class="button" href="{}" style="padding: 5px 10px; '
            'background: #417690; color: white; text-decoration: none; '
            'border-radius: 3px;">View References</a>',
            reverse('admin:storage_secretref_changelist') + 
            f'?provider=local_enc&path={obj.identifier}'
        )
    actions_column.short_description = _('Actions')
    
    def get_readonly_fields(self, request, obj=None):
        """Make identifier readonly after creation to prevent breaking references."""
        readonly = list(super().get_readonly_fields(request, obj))
        
        if obj and obj.pk:
            # Make identifier readonly for existing objects
            readonly.append('identifier')
        
        return readonly
    
    def save_model(self, request, obj, form, change):
        """Override to set created_by and updated_by."""
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
    
    def delete_model(self, request, obj):
        """Override delete to warn about references."""
        # Delete associated SecretRefs
        refs = SecretRef.objects.filter(
            organization=obj.organization,
            provider='local_enc',
            path=obj.identifier
        )
        refs_count = refs.count()
        
        if refs_count > 0:
            from django.contrib import messages
            messages.warning(
                request,
                _(f'Deleted {refs_count} SecretRef(s) that were pointing to this secret.')
            )
            refs.delete()
        
        super().delete_model(request, obj)
    
    def delete_queryset(self, request, queryset):
        """Override bulk delete to handle references."""
        total_refs = 0
        for obj in queryset:
            refs = SecretRef.objects.filter(
                organization=obj.organization,
                provider='local_enc',
                path=obj.identifier
            )
            total_refs += refs.count()
            refs.delete()
        
        if total_refs > 0:
            from django.contrib import messages
            messages.warning(
                request,
                _(f'Deleted {total_refs} SecretRef(s) in total.')
            )
        
        super().delete_queryset(request, queryset)
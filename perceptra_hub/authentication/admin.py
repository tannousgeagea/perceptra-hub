from django.contrib import admin
from unfold.admin import ModelAdmin
from django.utils.html import format_html
from django.utils import timezone
from .models import SocialAccount, OAuthProvider


@admin.register(SocialAccount)
class SocialAccountAdmin(ModelAdmin):
    """Admin interface for SocialAccount model."""
    
    list_display = [
        'user_link',
        'provider_badge',
        'email',
        'email_verified_icon',
        'is_primary',
        'is_active_status',
        'last_login',
        'created_at',
    ]
    
    list_filter = [
        'provider',
        'email_verified',
        'is_primary',
        'is_revoked',
        'created_at',
        'last_login',
    ]
    
    search_fields = [
        'user__username',
        'user__email',
        'email',
        'provider_user_id',
        'subject',
        'display_name',
        'given_name',
        'family_name',
    ]
    
    readonly_fields = [
        'uuid',
        'created_at',
        'updated_at',
        'token_status',
        'account_age',
    ]
    
    fieldsets = (
        ('User Information', {
            'fields': (
                'uuid',
                'user',
                'is_primary',
                'is_revoked',
            )
        }),
        ('Provider Details', {
            'fields': (
                'provider',
                'provider_user_id',
                'subject',
                'issuer',
                'tenant_id',
            )
        }),
        ('Profile Information', {
            'fields': (
                'email',
                'email_verified',
                'display_name',
                'given_name',
                'family_name',
                'avatar_url',
            )
        }),
        ('OAuth Tokens', {
            'classes': ('collapse',),
            'fields': (
                'access_token',
                'refresh_token',
                'token_expires_at',
                'token_status',
            )
        }),
        ('Metadata', {
            'classes': ('collapse',),
            'fields': (
                'extra_data',
            )
        }),
        ('Timestamps', {
            'fields': (
                'created_at',
                'updated_at',
                'last_login',
                'account_age',
            )
        }),
    )
    
    list_per_page = 50
    date_hierarchy = 'created_at'
    
    actions = [
        'revoke_accounts',
        'unrevoke_accounts',
        'mark_as_primary',
        'verify_emails',
    ]
    
    def user_link(self, obj):
        """Display clickable link to user."""
        from django.urls import reverse
        from django.contrib.contenttypes.models import ContentType
        
        try:
            # Get the content type for the User model
            content_type = ContentType.objects.get_for_model(obj.user.__class__)
            url = reverse(
                f'admin:{content_type.app_label}_{content_type.model}_change',
                args=[obj.user.id]
            )
            return format_html('<a href="{}">{}</a>', url, obj.user.username)
        except Exception:
            # Fallback to just showing the username if URL reverse fails
            return obj.user.username
    user_link.short_description = 'User'
    user_link.admin_order_field = 'user__username'
    
    def provider_badge(self, obj):
        """Display provider with colored badge."""
        colors = {
            OAuthProvider.MICROSOFT: '#00A4EF',
            OAuthProvider.GOOGLE: '#4285F4',
        }
        color = colors.get(obj.provider, '#6c757d')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            obj.get_provider_display()
        )
    provider_badge.short_description = 'Provider'
    provider_badge.admin_order_field = 'provider'
    
    def email_verified_icon(self, obj):
        """Display email verification status with icon."""
        if obj.email_verified:
            return format_html(
                '<span style="color: green; font-size: 16px;" title="Verified">✓</span>'
            )
        return format_html(
            '<span style="color: red; font-size: 16px;" title="Not Verified">✗</span>'
        )
    email_verified_icon.short_description = 'Email Verified'
    email_verified_icon.admin_order_field = 'email_verified'
    
    def is_active_status(self, obj):
        """Display active/revoked status."""
        if obj.is_active:
            return format_html(
                '<span style="color: green;">●</span> Active'
            )
        return format_html(
            '<span style="color: red;">●</span> Revoked'
        )
    is_active_status.short_description = 'Status'
    is_active_status.admin_order_field = 'is_revoked'
    
    def token_status(self, obj):
        """Display token expiration status."""
        if not obj.token_expires_at:
            return format_html('<span style="color: gray;">No token</span>')
        
        now = timezone.now()
        if obj.token_expires_at > now:
            time_left = obj.token_expires_at - now
            days = time_left.days
            hours = time_left.seconds // 3600
            return format_html(
                '<span style="color: green;">Valid ({} days, {} hours left)</span>',
                days, hours
            )
        return format_html('<span style="color: red;">Expired</span>')
    token_status.short_description = 'Token Status'
    
    def account_age(self, obj):
        """Display account age."""
        age = timezone.now() - obj.created_at
        days = age.days
        if days < 1:
            hours = age.seconds // 3600
            return f"{hours} hours"
        elif days < 30:
            return f"{days} days"
        elif days < 365:
            months = days // 30
            return f"{months} months"
        else:
            years = days // 365
            return f"{years} years"
    account_age.short_description = 'Account Age'
    
    # Admin Actions
    
    @admin.action(description='Revoke selected accounts')
    def revoke_accounts(self, request, queryset):
        """Revoke selected social accounts."""
        count = 0
        for account in queryset:
            if not account.is_revoked:
                account.revoke()
                count += 1
        self.message_user(request, f'{count} account(s) revoked successfully.')
    
    @admin.action(description='Unrevoke selected accounts')
    def unrevoke_accounts(self, request, queryset):
        """Unrevoke selected social accounts."""
        count = queryset.filter(is_revoked=True).update(is_revoked=False)
        self.message_user(request, f'{count} account(s) unrevoked successfully.')
    
    @admin.action(description='Mark as primary account')
    def mark_as_primary(self, request, queryset):
        """Mark selected accounts as primary for their provider."""
        count = 0
        for account in queryset:
            # Unmark other accounts for same user+provider
            SocialAccount.objects.filter(
                user=account.user,
                provider=account.provider
            ).exclude(id=account.id).update(is_primary=False)
            
            # Mark this one as primary
            account.is_primary = True
            account.save(update_fields=['is_primary'])
            count += 1
        
        self.message_user(request, f'{count} account(s) marked as primary.')
    
    @admin.action(description='Mark emails as verified')
    def verify_emails(self, request, queryset):
        """Mark selected account emails as verified."""
        count = queryset.update(email_verified=True)
        self.message_user(request, f'{count} email(s) marked as verified.')
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related('user')
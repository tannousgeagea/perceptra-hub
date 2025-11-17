"""
FastAPI routes for OAuth authentication (Microsoft & Google).
"""
from fastapi import APIRouter, HTTPException, status, Request
from asgiref.sync import sync_to_async
import logging
from typing import Optional

from api.routers.auth.schemas import (
    OAuthInitiateRequest,
    OAuthInitiateResponse,
    OAuthCallbackRequest,
    TokenResponse,
    MessageResponse,
)
from common_utils.auth.oauth import (
    MicrosoftOAuth,
    GoogleOAuth,
    OAuthStateManager,
    create_or_update_oauth_user,
)
from common_utils.auth.utils import create_tokens_for_user
from organizations.models import Organization
from authentication.models import SocialAccount
from memberships.models import OrganizationMembership

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/oauth", tags=["OAuth Authentication"])


# ============= OAuth Flow: Step 1 - Initiate =============

@router.post(
    "/initiate",
    response_model=OAuthInitiateResponse,
    summary="Initiate OAuth Flow",
    description="Get OAuth authorization URL to redirect user to"
)
async def initiate_oauth(request: OAuthInitiateRequest):
    """
    Step 1: Initiate OAuth flow.
    
    Returns authorization URL for the user to visit.
    Frontend should:
    1. Store the 'state' value
    2. Redirect user to 'authorization_url'
    3. User logs in with Microsoft/Google
    4. Provider redirects back to your callback URL with code and state
    """
    provider = request.provider.lower()
    
    if provider not in ['microsoft', 'google']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider: {provider}. Supported: microsoft, google"
        )
    
    # Generate CSRF state token
    state = OAuthStateManager.generate_state({"provider": provider})
    
    # Get authorization URL
    if provider == 'microsoft':
        auth_url = MicrosoftOAuth.get_authorization_url(state)
    elif provider == 'google':
        auth_url = GoogleOAuth.get_authorization_url(state)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider: {provider}"
        )
    
    logger.info(f"OAuth initiated for provider: {provider}")
    
    return OAuthInitiateResponse(
        authorization_url=auth_url,
        state=state
    )


# ============= OAuth Flow: Step 2 - Callback =============

@sync_to_async
def get_user_organizations_data(user):
    """Get user's organizations (synchronous)."""
    memberships = OrganizationMembership.objects.filter(
        user=user
    ).select_related('organization', 'role')
    
    return [
        {
            "id": str(m.organization.org_id),
            "name": m.organization.name,
            "slug": m.organization.slug,
            "role": m.role.name
        }
        for m in memberships
    ]

@router.api_route(
    "/callback",
    methods=["GET", "POST"],
    response_model=TokenResponse,
    summary="OAuth Callback Handler",
    description="Handle OAuth callback with authorization code"
)
async def oauth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    provider: Optional[str] = None,
    callback_data: Optional[OAuthCallbackRequest] = None
):
    """
    Step 2: Handle OAuth callback.
    
    This endpoint receives the authorization code from the OAuth provider
    and completes the authentication flow.
    
    Process:
    1. Verify state (CSRF protection)
    2. Exchange code for access token
    3. Fetch user info from provider
    4. Create or update user in database
    5. Return JWT tokens for your platform
    """
    
    
    # Handle GET request (redirect from OAuth provider)
    if request.method == "GET":
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        provider = request.query_params.get("provider")
        
        # Provider might be in path or need to be inferred
        if not provider:
            # Try to get provider from stored state data
            if state and OAuthStateManager.verify_state(state):
                state_data = OAuthStateManager._states.get(state, {})
                provider = state_data.get("data", {}).get("provider")
            
            # Check session_state parameter (Microsoft includes this)
            if not provider and request.query_params.get("session_state"):
                provider = "microsoft"
            
            # Fallback to microsoft as default
            if not provider:
                logger.warning("Could not infer provider, defaulting to microsoft")
                provider = "microsoft"
        
        if not code or not state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing code or state parameter"
            )
    # Handle POST request (manual testing or API call)
    else:
        if not callback_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body required for POST requests"
            )
        code = callback_data.code
        state = callback_data.state
        provider = callback_data.provider
    
    # Verify state to prevent CSRF
    logger.warning(f"State: {state}")
    if not OAuthStateManager.verify_state(state):
        logger.warning(f"Invalid OAuth state received: {state}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state token. Please try again."
        )
    
    provider = provider.lower()
    
    # Remove state after verification
    OAuthStateManager.remove_state(state)
    
    try:
        if provider == 'microsoft':
            # Exchange code for token
            token_response = await MicrosoftOAuth.exchange_code_for_token(code)
            if not token_response:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange authorization code for token"
                )
            
            access_token = token_response.get('access_token')
            refresh_token = token_response.get('refresh_token')
            expires_in = token_response.get('expires_in')
            
            # Get user info from Microsoft Graph
            user_info = await MicrosoftOAuth.get_user_info(access_token)
            
            if not user_info:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to fetch user information from Microsoft"
                )
            
            # Extract and normalize user data
            user_data_dict = MicrosoftOAuth.extract_user_data(token_response, user_info)
            
            # Log what we received
            logger.info(f"Microsoft user info: {user_data_dict['display_name']} ({user_data_dict['email']})")
                        
        elif provider == 'google':
            # Exchange code for token
            token_response = await GoogleOAuth.exchange_code_for_token(code)
            
            if not token_response:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange authorization code for token"
                )
            
            access_token = token_response.get('access_token')
            refresh_token = token_response.get('refresh_token')
            expires_in = token_response.get('expires_in')
            
            # Get user info from Google UserInfo API
            user_info = await GoogleOAuth.get_user_info(access_token)
            
            if not user_info:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to fetch user information from Google"
                )
            
            # Extract and normalize user data
            user_data_dict = GoogleOAuth.extract_user_data(token_response, user_info)
            
            # Log what we received
            logger.info(f"Google user info: {user_data_dict['display_name']} ({user_data_dict['email']})")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider: {provider}"
            )
        
        # Create or update user with all OIDC fields
        user, social_account, created = await sync_to_async(create_or_update_oauth_user)(
            provider=provider,
            provider_user_id=user_data_dict['provider_user_id'],
            subject=user_data_dict['subject'],
            email=user_data_dict['email'],
            email_verified=user_data_dict['email_verified'],
            given_name=user_data_dict['given_name'],
            family_name=user_data_dict['family_name'],
            display_name=user_data_dict['display_name'],
            avatar_url=user_data_dict['avatar_url'],
            issuer=user_data_dict['issuer'],
            tenant_id=user_data_dict['tenant_id'],
            extra_data=user_data_dict['extra_data'],
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
        )
        
        # Create JWT tokens for your platform
        tokens = create_tokens_for_user(user)
        
        # Get user's organizations
        organizations = await get_user_organizations_data(user)
        
        # Prepare response
        user_data = {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "organizations": organizations
        }
        
        action = "registered" if created else "logged in"
        logger.info(f"User {action} via {provider}: {user.username}")
        
        return TokenResponse(**tokens, user=user_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback error ({provider}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


# ============= OAuth Account Management =============

@sync_to_async
def get_user_social_accounts(user):
    """Get all active social accounts for user."""
    accounts = SocialAccount.objects.filter(
        user=user,
        is_revoked=False
    ).order_by('-is_primary', '-last_login')
    
    return [
        {
            "provider": account.provider,
            "email": account.email,
            "email_verified": account.email_verified,
            "display_name": account.display_name,
            "is_primary": account.is_primary,
            "linked_at": account.created_at.isoformat(),
            "last_used": account.last_login.isoformat(),
        }
        for account in accounts
    ]

@router.get(
    "/linked-accounts",
    summary="Get Linked OAuth Accounts",
    description="Get all OAuth accounts linked to current user"
)
async def get_linked_accounts(user = None):  # TODO: Add Depends(get_current_user) when ready
    """
    Get user's linked OAuth accounts.
    
    TODO: Add authentication dependency
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication required - will be implemented after basic OAuth flow is working"
        )
    
    accounts = await get_user_social_accounts(user)
    
    return {
        "accounts": accounts,
        "count": len(accounts)
    }


@sync_to_async
def revoke_social_account(user, provider: str):
    """Revoke/unlink a social account."""
    try:
        account = SocialAccount.objects.get(
            user=user,
            provider=provider,
            is_revoked=False
        )
        
        # Don't allow unlinking if it's the only way to sign in
        has_password = user.has_usable_password()
        other_accounts = SocialAccount.objects.filter(
            user=user,
            is_revoked=False
        ).exclude(id=account.uuid).exists()
        
        if not has_password and not other_accounts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot unlink the only authentication method. Please set a password first."
            )
        
        account.revoke(save=True)
        logger.info(f"User {user.username} unlinked {provider} account")
        
        return True
        
    except SocialAccount.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active {provider} account found"
        )


@router.delete(
    "/unlink/{provider}",
    response_model=MessageResponse,
    summary="Unlink OAuth Account",
    description="Remove OAuth provider link from user account"
)
async def unlink_oauth_account(provider: str, user = None):  # TODO: Add Depends(get_current_user)
    """
    Unlink OAuth account from user.
    
    TODO: Add authentication dependency
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication required - will be implemented after basic OAuth flow is working"
        )
    
    await revoke_social_account(user, provider.lower())
    
    return MessageResponse(
        message=f"{provider.title()} account unlinked successfully"
    )


# ============= Development/Testing Endpoints =============

@router.get(
    "/debug/callback",
    summary="Debug OAuth Callback (Dev Only)",
    description="View all parameters received in callback"
)
async def debug_callback(request: Request):
    """Debug endpoint to see what parameters Microsoft is sending."""
    return {
        "method": request.method,
        "url": str(request.url),
        "query_params": dict(request.query_params),
        "headers": {
            "referer": request.headers.get("referer"),
            "user-agent": request.headers.get("user-agent"),
        },
        "message": "This is what the callback received"
    }


@router.get(
    "/test-state",
    summary="Test State Generation (Dev Only)",
    description="Generate and verify state token for testing"
)
async def test_state_generation():
    """Test state token generation and verification."""
    state = OAuthStateManager.generate_state({"test": "data"})
    is_valid = OAuthStateManager.verify_state(state)
    
    # Generate a test authorization URL
    test_url = MicrosoftOAuth.get_authorization_url(state)
    
    return {
        "state": state,
        "is_valid": is_valid,
        "test_authorization_url": test_url,
        "message": "State generated successfully. Check if 'state' appears in the test URL."
    }
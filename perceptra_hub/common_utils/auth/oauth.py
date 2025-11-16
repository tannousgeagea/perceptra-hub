"""
OAuth utility functions for Microsoft and Google authentication.
"""
import httpx
import secrets
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from django.conf import settings
from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)


# ============= Microsoft OAuth =============

class MicrosoftOAuth:
    """Microsoft OAuth 2.0 handler."""
    
    SCOPES = ["openid", "profile", "email", "User.Read"]
    
    @staticmethod
    def extract_user_data(token_response: Dict[str, Any], user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize user data from Microsoft OAuth response.
        
        Args:
            token_response: Token response from Microsoft
            user_info: User info from Microsoft Graph API
            
        Returns:
            Normalized user data dict
        """
        # Decode ID token to get OIDC claims (if available)
        id_token = token_response.get('id_token')
        id_token_claims = {}
        
        if id_token:
            try:
                import jwt
                # Decode without verification (we already trust the token from Microsoft)
                id_token_claims = jwt.decode(id_token, options={"verify_signature": False})
            except Exception as e:
                logger.warning(f"Could not decode ID token: {e}")
        
        # Extract data with fallbacks
        return {
            "provider_user_id": user_info.get('id', ''),
            "subject": id_token_claims.get('sub') or id_token_claims.get('oid') or user_info.get('id', ''),
            "email": user_info.get('mail') or user_info.get('userPrincipalName') or id_token_claims.get('email', ''),
            "email_verified": id_token_claims.get('email_verified', True),  # Microsoft emails are generally verified
            "given_name": user_info.get('givenName', ''),
            "family_name": user_info.get('surname', ''),
            "display_name": user_info.get('displayName', ''),
            "avatar_url": "",  # Microsoft Graph doesn't return photo URL directly
            "issuer": id_token_claims.get('iss', f"https://login.microsoftonline.com/{id_token_claims.get('tid', 'common')}/v2.0"),
            "tenant_id": id_token_claims.get('tid', ''),
            "extra_data": {
                "job_title": user_info.get('jobTitle', ''),
                "office_location": user_info.get('officeLocation', ''),
                "preferred_language": user_info.get('preferredLanguage', ''),
                "id_token_claims": id_token_claims,
                "raw_user_info": user_info,
            }
        }
    
    @staticmethod
    def get_authorization_url(state: str) -> str:
        """
        Generate Microsoft OAuth authorization URL.
        
        Args:
            state: Random state string for CSRF protection
            
        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": settings.MICROSOFT_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": settings.MICROSOFT_REDIRECT_URI,
            "scope": " ".join(MicrosoftOAuth.SCOPES),
            "state": state,
            "response_mode": "query",
            "prompt": "select_account",
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{settings.MICROSOFT_AUTHORIZATION_URL}?{query_string}"
    
    @staticmethod
    async def exchange_code_for_token(code: str) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code from Microsoft
            
        Returns:
            Token response dict or None if failed
        """
        token_data = {
            "client_id": settings.MICROSOFT_CLIENT_ID,
            "client_secret": settings.MICROSOFT_CLIENT_SECRET,
            "code": code,
            "redirect_uri": settings.MICROSOFT_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    settings.MICROSOFT_TOKEN_URL,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None
    
    @staticmethod
    async def get_user_info(access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from Microsoft Graph API.
        
        Args:
            access_token: Microsoft access token
            
        Returns:
            User info dict or None if failed
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    settings.MICROSOFT_GRAPH_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get user info: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None


# ============= State Management for CSRF Protection =============

class OAuthStateManager:
    """
    Manages OAuth state tokens for CSRF protection.
    In production, use Redis or database. For now, in-memory store.
    """
    
    # In-memory store (replace with Redis in production)
    _states = {}
    
    @classmethod
    def generate_state(cls, user_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate random state token.
        
        Args:
            user_data: Optional data to associate with state
            
        Returns:
            State token
        """
        state = secrets.token_urlsafe(32)
        cls._states[state] = {
            "created_at": datetime.now(),
            "data": user_data or {}
        }
        return state
    
    @classmethod
    def store_code_state_mapping(cls, code_prefix: str, state: str):
        """
        Store mapping between authorization code prefix and state.
        Useful when state is lost during redirect.
        
        Args:
            code_prefix: First 20 chars of authorization code
            state: State token to associate
        """
        cls._code_to_state[code_prefix] = state
        logger.debug(f"Stored code->state mapping: {code_prefix[:10]}... -> {state[:10]}...")
    
    @classmethod
    def get_state_by_code(cls, code: str) -> Optional[str]:
        """
        Retrieve state by authorization code prefix.
        
        Args:
            code: Authorization code
            
        Returns:
            State token if found, None otherwise
        """
        code_prefix = code[:20] if len(code) > 20 else code
        return cls._code_to_state.get(code_prefix)
    
    @classmethod
    def verify_state(cls, state: str) -> bool:
        """
        Verify state token exists and is not expired.
        
        Args:
            state: State token to verify
            
        Returns:
            True if valid, False otherwise
        """
        if state not in cls._states:
            return False
        
        # Check if expired (10 minutes)
        state_data = cls._states[state]
        age = datetime.now() - state_data["created_at"]
        
        if age > timedelta(minutes=10):
            cls.remove_state(state)
            return False
        
        return True
    
    @classmethod
    def remove_state(cls, state: str):
        """Remove state token after use."""
        cls._states.pop(state, None)
    
    @classmethod
    def cleanup_expired(cls):
        """Remove expired states (call periodically)."""
        now = datetime.now()
        expired = [
            state for state, data in cls._states.items()
            if now - data["created_at"] > timedelta(minutes=10)
        ]
        for state in expired:
            cls._states.pop(state, None)


# ============= User Creation/Update Helper =============

def normalize_email(email: str) -> str:
    """Normalize email to lowercase."""
    return email.lower().strip()


def generate_unique_username(base_username: str) -> str:
    """
    Generate unique username by appending numbers if needed.
    
    Args:
        base_username: Base username to start with
        
    Returns:
        Unique username
    """
    username = base_username.lower().replace(" ", "_")
    
    # Remove special characters
    username = "".join(c for c in username if c.isalnum() or c == "_")
    
    # Ensure it's not empty
    if not username:
        username = "user"
    
    # Check uniqueness
    if not User.objects.filter(username=username).exists():
        return username
    
    # Append numbers until unique
    counter = 1
    while True:
        new_username = f"{username}{counter}"
        if not User.objects.filter(username=new_username).exists():
            return new_username
        counter += 1


def create_or_update_oauth_user(
    provider: str,
    provider_user_id: str,
    subject: str,
    email: str,
    email_verified: bool = False,
    given_name: str = "",
    family_name: str = "",
    display_name: str = "",
    avatar_url: str = "",
    issuer: str = "",
    tenant_id: str = "",
    extra_data: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    expires_in: Optional[int] = None,
) -> tuple:
    """
    Create or update user from OAuth provider.
    
    Args:
        provider: OAuth provider name (e.g., 'microsoft', 'google')
        provider_user_id: User ID from provider (legacy, for compatibility)
        subject: Stable OIDC subject identifier ('sub' claim)
        email: User email
        email_verified: Whether email is verified by provider
        given_name: User given/first name
        family_name: User family/last name
        display_name: Full display name
        avatar_url: Profile picture URL
        issuer: OIDC issuer URI
        tenant_id: Provider tenant ID (e.g., Azure AD tenant)
        extra_data: Additional provider data
        access_token: OAuth access token
        refresh_token: OAuth refresh token
        expires_in: Token expiration time in seconds
        
    Returns:
        Tuple of (User, SocialAccount, created: bool)
    """
    from authentication.models import SocialAccount
    
    email = normalize_email(email)
    
    # Try to find existing social account
    try:
        social_account = SocialAccount.objects.select_related('user').get(
            provider=provider,
            subject=subject,
            is_revoked=False  # Only match active accounts
        )
        
        # Update existing user
        user = social_account.user
        user.email = email
        user.first_name = given_name or user.first_name
        user.last_name = family_name or user.last_name
        user.save()
        
        # Update social account
        social_account.provider_user_id = provider_user_id
        social_account.email = email
        social_account.email_verified = email_verified
        social_account.display_name = display_name
        social_account.given_name = given_name
        social_account.family_name = family_name
        social_account.avatar_url = avatar_url
        social_account.issuer = issuer
        social_account.tenant_id = tenant_id
        social_account.extra_data = extra_data or {}
        if access_token:
            social_account.access_token = access_token
        if refresh_token:
            social_account.refresh_token = refresh_token
        if expires_in:
            social_account.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        social_account.last_login = datetime.now()
        social_account.save()
        
        logger.info(f"Updated existing OAuth user: {user.username} ({provider})")
        return user, social_account, False
        
    except SocialAccount.DoesNotExist:
        # Try to find user by email
        try:
            user = User.objects.get(email=email)
            logger.info(f"Linking {provider} account to existing user: {user.username}")
            
        except User.DoesNotExist:
            # Create new user
            username = generate_unique_username(
                given_name or email.split('@')[0]
            )
            
            user = User.objects.create_user(
                username=username,
                email=email,
                first_name=given_name,
                last_name=family_name,
            )
            
            # OAuth users don't need password
            user.set_unusable_password()
            user.save()
            
            logger.info(f"Created new OAuth user: {user.username} ({provider})")
        
        # Check if this is the first social account for this provider
        is_primary = not SocialAccount.objects.filter(
            user=user,
            provider=provider,
            is_revoked=False
        ).exists()
        
        # Create social account
        social_account = SocialAccount.objects.create(
            user=user,
            provider=provider,
            provider_user_id=provider_user_id,
            subject=subject,
            issuer=issuer,
            tenant_id=tenant_id,
            email=email,
            email_verified=email_verified,
            display_name=display_name,
            given_name=given_name,
            family_name=family_name,
            avatar_url=avatar_url,
            is_primary=is_primary,
            access_token=access_token or "",
            refresh_token=refresh_token or "",
            token_expires_at=datetime.now() + timedelta(seconds=expires_in) if expires_in else None,
            extra_data=extra_data or {},
        )
        
        return user, social_account, True
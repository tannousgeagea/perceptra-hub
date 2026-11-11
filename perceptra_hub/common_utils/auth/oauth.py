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
from django.core.cache import cache
from memberships.models import OrganizationMembership

User = get_user_model()
logger = logging.getLogger(__name__)


# ============= Google OAuth =============

class GoogleOAuth:
    """Google OAuth 2.0 handler."""
    
    SCOPES = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ]
    
    @staticmethod
    def get_authorization_url(state: str) -> str:
        """
        Generate Google OAuth authorization URL.
        
        Args:
            state: Random state string for CSRF protection
            
        Returns:
            Authorization URL to redirect user to
        """
        from urllib.parse import urlencode
        
        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "scope": " ".join(GoogleOAuth.SCOPES),
            "state": state,
            "access_type": "offline",  # Get refresh token
            "prompt": "select_account",  # Let user choose account
        }
        
        query_string = urlencode(params)
        auth_url = f"{settings.GOOGLE_AUTHORIZATION_URL}?{query_string}"
        
        logger.info(f"Generated Google auth URL with state: {state[:20]}...")
        return auth_url
    
    @staticmethod
    async def exchange_code_for_token(code: str) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code from Google
            
        Returns:
            Token response dict or None if failed
        """
        token_data = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "code": code,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    settings.GOOGLE_TOKEN_URL,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Google token exchange failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None
    
    @staticmethod
    async def get_user_info(access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from Google UserInfo API.
        
        Args:
            access_token: Google access token
            
        Returns:
            User info dict or None if failed
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    settings.GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get Google user info: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    @staticmethod
    def extract_user_data(token_response: Dict[str, Any], user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize user data from Google OAuth response.
        
        Args:
            token_response: Token response from Google
            user_info: User info from Google UserInfo API
            
        Returns:
            Normalized user data dict
        """
        # Decode ID token to get OIDC claims (if available)
        id_token = token_response.get('id_token')
        id_token_claims = {}
        
        if id_token:
            try:
                import jwt
                # Decode without verification (we already trust the token from Google)
                id_token_claims = jwt.decode(id_token, options={"verify_signature": False})
            except Exception as e:
                logger.warning(f"Could not decode Google ID token: {e}")
        
        # Extract data with fallbacks
        return {
            "provider_user_id": user_info.get('id', ''),
            "subject": id_token_claims.get('sub') or user_info.get('id', ''),
            "email": user_info.get('email', ''),
            # Only an explicit true counts (userinfo v2 says "verified_email", v3 and the ID token "email_verified").
            "email_verified": any(
                value is True
                for value in (user_info.get('verified_email'), user_info.get('email_verified'),
                              id_token_claims.get('email_verified'))
            ),
            "given_name": user_info.get('given_name', ''),
            "family_name": user_info.get('family_name', ''),
            "display_name": user_info.get('name', ''),
            "avatar_url": user_info.get('picture', ''),
            "issuer": id_token_claims.get('iss', 'https://accounts.google.com'),
            "tenant_id": "",  # Google doesn't have tenant concept
            "extra_data": {
                "locale": user_info.get('locale', ''),
                "hd": user_info.get('hd', ''),  # Hosted domain (for Google Workspace)
                "id_token_claims": id_token_claims,
                "raw_user_info": user_info,
            }
        }

# ============= Microsoft OAuth =============

class MicrosoftOAuth:
    """Microsoft OAuth 2.0 handler."""
    
    SCOPES = ["openid", "profile", "email", "User.Read"]

    # Tenant of personal Microsoft accounts (outlook.com, hotmail.com, …),
    # whose email addresses Microsoft itself has verified.
    CONSUMER_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad"

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
        
        # Which email Microsoft vouches for. Graph's "mail" and
        # "userPrincipalName" are set by whoever runs the tenant, so they prove
        # nothing ("nOAuth"). Only the ID token's email claim counts, and only
        # for personal Microsoft accounts or when Microsoft marks the email's
        # domain as verified by the tenant (optional claim xms_edov).
        email_claim = id_token_claims.get('email', '')
        email_verified = bool(email_claim) and (
            id_token_claims.get('tid') == MicrosoftOAuth.CONSUMER_TENANT_ID
            or id_token_claims.get('xms_edov') in (True, 1, 'true', 'True', '1')
        )

        # Extract data with fallbacks
        return {
            "provider_user_id": user_info.get('id', ''),
            "subject": id_token_claims.get('sub') or id_token_claims.get('oid') or user_info.get('id', ''),
            "email": email_claim if email_verified else (
                user_info.get('mail') or user_info.get('userPrincipalName') or email_claim
            ),
            "email_verified": email_verified,
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
        cache.set(f"oauth_state:{state}", {
            "created_at": datetime.now().isoformat(),
            "data": user_data or {}
        }, timeout=600)
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
        
        state_data = cache.get(f"oauth_state:{state}")
        logger.warning(f"State Data: {state_data}")
        return state_data is not None
    
    @classmethod
    def remove_state(cls, state: str):
        """Remove state token after use."""
        cache.delete(f"oauth_state:{state}")
    
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

class OAuthEmailNotVerified(Exception):
    """The provider didn't confirm the email, so it can't name a Malumetrix account."""

    def __init__(self, provider: str):
        self.provider = provider
        label = {'microsoft': 'Microsoft', 'google': 'Google'}.get(provider, provider)
        super().__init__(
            f"{label} didn't confirm the email address of this account, so it can't be used to sign in. "
            "Sign in with your email and password, or use an account whose email is verified."
        )


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
    invitation=None,
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
        invitation: A pending invitation already checked for this email. An
            account with no organization joins it instead of getting a
            personal workspace.

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
        # Follow an email change at the provider only when it's verified and
        # not another account's (it is how people sign in with a password).
        if (email_verified and email and email != user.email
                and not User.objects.filter(email__iexact=email).exclude(pk=user.pk).exists()):
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
        # First sign-in with this provider account: the email decides which
        # Malumetrix account it becomes, so the provider must have verified it.
        # Otherwise anyone could claim someone else's address (and account).
        if not email_verified or not email:
            logger.warning(f"Refused {provider} sign-in: email not verified by the provider")
            raise OAuthEmailNotVerified(provider)

        # Try to find user by email
        try:
            user = User.objects.get(email__iexact=email)
            logger.info(f"Linking {provider} account to existing user: {user.username}")
            
        except User.DoesNotExist:
            # Create new user
            username = generate_unique_username(
                given_name or email.split('@')[0]
            )
            
            user = User.objects.create_user(
                username=username,
                email=email,
                password=None,
                first_name=given_name,
                last_name=family_name,
            )
            
            # OAuth users don't need password
            user.set_unusable_password()
            user.save()
            
            logger.info(f"Created new OAuth user: {user.username} ({provider})")
        
            # Signing in from an invitation: join that organization.
            if invitation is not None and not OrganizationMembership.objects.filter(user=user).exists():
                from memberships import invitations as invites

                try:
                    invites.accept(invitation, user)
                    logger.info(f"{user.username} joined organization {invitation.organization_id} by invitation")
                except invites.InvitationError as err:
                    # Used or withdrawn in the meantime: fall back to a workspace.
                    logger.info(f"Invitation not accepted for {user.username}: {err.code}")

            # Ensure user has at least one organization
            if not OrganizationMembership.objects.filter(user=user).exists():
                from organizations.workspaces import create_personal_workspace

                create_personal_workspace(user)
        
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
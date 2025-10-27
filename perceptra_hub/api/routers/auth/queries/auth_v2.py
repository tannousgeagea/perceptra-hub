"""
FastAPI routes for authentication.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from django.contrib.auth import get_user_model, authenticate
from django.db import IntegrityError
from typing import Optional
import logging
from asgiref.sync import sync_to_async

from api.routers.auth.schemas import (
    UserRegister,
    UserLogin,
    TokenResponse,
    TokenRefresh,
    PasswordChange,
    PasswordResetRequest,
    PasswordReset,
    UserProfile,
    UserProfileUpdate,
    UserContext,
    OrganizationInfo,
    MessageResponse,
)
from common_utils.auth.utils import (
    create_tokens_for_user,
    verify_refresh_token,
    verify_password_reset_token,
    generate_password_reset_token,
)
from api.dependencies import get_current_user, get_optional_user, fetch_user_from_db
from organizations.models import Organization
from memberships.models import OrganizationMembership

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============= Registration =============
@sync_to_async
def register_new_user(user_data: UserRegister):
    """
    Synchronous function to register a new user.
    """
    # Check if username exists
    if User.objects.filter(username=user_data.username).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    if User.objects.filter(email=user_data.email).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = User.objects.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
    )
    
    # Create tokens
    tokens = create_tokens_for_user(user)
    
    # Get user organizations (will be empty for new user)
    organizations = []
    
    return {
        "tokens": tokens,
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "organizations": organizations
        },
        "username": user.username  # For logging
    }

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Create a new user account"
)
async def register(user_data: UserRegister):
    """
    Register a new user.
    
    Creates a new user account and returns access/refresh tokens.
    """
    try:
        result = await register_new_user(user_data)
        
        logger.info(f"New user registered: {result['username']}")
        
        return TokenResponse(**result['tokens'], user=result['user'])
        
    except IntegrityError as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed. Username or email may already exist."
        )


# ============= Login =============
@sync_to_async
def authenticate_user(credentials: UserLogin):
    """
    Synchronous function to authenticate user and gather all data.
    """
    # Try to find user by username or email
    user = None
    
    # Check if input is email
    if '@' in credentials.username:
        try:
            user = User.objects.get(email=credentials.username, is_active=True)
        except User.DoesNotExist:
            pass
    else:
        try:
            user = User.objects.get(username=credentials.username, is_active=True)
        except User.DoesNotExist:
            pass
    
    # Authenticate
    if not user or not user.check_password(credentials.password):
        return None
    
    # Update last login
    user.last_login = datetime.now()
    user.save(update_fields=['last_login'])
    
    # Create tokens
    tokens = create_tokens_for_user(user)
    
    # Get user's organizations
    memberships = OrganizationMembership.objects.filter(
        user=user
    ).select_related('organization', 'role')
    
    organizations = [
        {
            "id": str(m.organization.org_id),
            "name": m.organization.name,
            "slug": m.organization.slug,
            "role": m.role.name
        }
        for m in memberships
    ]
    
    return {
        "tokens": tokens,
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "organizations": organizations
        }
    }


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login",
    description="Authenticate user and return tokens"
)
async def login(credentials: UserLogin):
    """
    Authenticate user with username/email and password.
    
    Returns access and refresh tokens on success.
    """
    # Try to find user by username or email
    result = await authenticate_user(credentials)
    
    if result:
        logger.info(f"User logged in: {result['user']['username']}")
        return TokenResponse(**result['tokens'], user=result['user'])
    
    # Authentication failed
    logger.warning(f"Failed login attempt for: {credentials.username}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username/email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ============= Token Refresh =============
@sync_to_async
def refresh_user_tokens(user_id: str):
    """
    Synchronous function to refresh tokens for a user.
    """
    # Get user
    try:
        user = User.objects.get(id=user_id, is_active=True)
    except User.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new tokens
    tokens = create_tokens_for_user(user)
    
    # Get organizations
    memberships = OrganizationMembership.objects.filter(
        user=user
    ).select_related('organization', 'role')
    
    organizations = [
        {
            "id": str(m.organization.org_id),
            "name": m.organization.name,
            "slug": m.organization.slug,
            "role": m.role.name
        }
        for m in memberships
    ]
    
    return {
        "tokens": tokens,
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "organizations": organizations
        }
    }

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Access Token",
    description="Get new access token using refresh token"
)
async def refresh_token(token_data: TokenRefresh):
    """
    Refresh access token using refresh token.
    """
    # Verify refresh token
    payload = verify_refresh_token(token_data.refresh_token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Refresh tokens and get user data
    result = await refresh_user_tokens(user_id)
    
    return TokenResponse(**result['tokens'], user=result['user'])



# ============= Logout =============

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="User Logout",
    description="Logout user (client should delete tokens)"
)
async def logout(user: User = Depends(get_current_user)):
    """
    Logout user.
    
    Note: JWT tokens are stateless, so logout is handled client-side
    by deleting the tokens. This endpoint is for logging purposes.
    """
    logger.info(f"User logged out: {user.username}")
    
    return MessageResponse(
        message="Logged out successfully. Please delete your tokens."
    )


# ============= User Profile =============
@sync_to_async
def get_user_profile_data(user: User):
    """
    Synchronous function to gather user profile data.
    """
    # Get user's organizations
    memberships = OrganizationMembership.objects.filter(
        user=user
    ).select_related('organization', 'role')
    
    organizations = [
        OrganizationInfo(
            id=str(m.organization.org_id),
            name=m.organization.name,
            slug=m.organization.slug,
            role=m.role.name,
            joined_at=m.joined_at
        )
        for m in memberships
    ]
    
    return UserContext(
        user=UserProfile(
            id=str(user.id),
            username=user.username,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            date_joined=user.date_joined,
            last_login=user.last_login,
            organizations=[org.dict() for org in organizations]
        ),
        current_organization=None,  # Will be set by frontend based on X-Organization-ID
        available_organizations=organizations
    )

@router.get(
    "/me",
    response_model=UserContext,
    summary="Get Current User",
    description="Get current authenticated user's profile"
)
async def get_current_user_profile(user: User = Depends(get_current_user)):
    """Get current user profile with organizations."""
    return await get_user_profile_data(user)


@sync_to_async
def update_user_profile_data(user: User, profile_data: UserProfileUpdate):
    """
    Synchronous function to update user profile.
    """
    # Update fields
    if profile_data.first_name is not None:
        user.first_name = profile_data.first_name
    if profile_data.last_name is not None:
        user.last_name = profile_data.last_name
    if profile_data.email is not None:
        # Check if email already exists
        if User.objects.filter(email=profile_data.email).exclude(id=user.id).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        user.email = profile_data.email
    
    user.save()
    
    # Get organizations
    memberships = OrganizationMembership.objects.filter(
        user=user
    ).select_related('organization', 'role')
    
    organizations = [
        {
            "id": str(m.organization.org_id),
            "name": m.organization.name,
            "slug": m.organization.slug,
            "role": m.role.name
        }
        for m in memberships
    ]
    
    return UserProfile(
        id=str(user.id),
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        date_joined=user.date_joined,
        last_login=user.last_login,
        organizations=organizations
    )

@router.put(
    "/me",
    response_model=UserProfile,
    summary="Update User Profile",
    description="Update current user's profile information"
)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    user: User = Depends(get_current_user)
):
    """Update user profile."""
    
    result = await update_user_profile_data(user, profile_data)
    
    logger.info(f"User profile updated: {result.username}")
    
    return result



# ============= Password Management =============
@sync_to_async
def change_user_password(user: User, password_data: PasswordChange):
    """
    Synchronous function to change user password.
    """
    # Verify old password
    if not user.check_password(password_data.old_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Set new password
    user.set_password(password_data.new_password)
    user.save()
    
    return user.username  # For logging

@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change Password",
    description="Change user's password"
)
async def change_password(
    password_data: PasswordChange,
    user: User = Depends(get_current_user)
):
    """Change user password."""
    
    username = await change_user_password(user, password_data)
    
    logger.info(f"Password changed for user: {username}")
    
    return MessageResponse(
        message="Password changed successfully"
    )


@sync_to_async
def process_password_reset_request(email: str):
    """
    Synchronous function to process password reset request.
    """
    try:
        user = User.objects.get(email=email, is_active=True)
        
        # Generate reset token
        reset_token = generate_password_reset_token(str(user.id))
        
        # TODO: Send email with reset link
        # send_password_reset_email(user.email, reset_token)
        
        return {
            "user_found": True,
            "email": user.email,
            "reset_token": reset_token  # For dev logging only
        }
        
    except User.DoesNotExist:
        # Don't reveal that email doesn't exist
        return {
            "user_found": False,
            "email": email,
            "reset_token": None
        }

@router.post(
    "/password-reset-request",
    response_model=MessageResponse,
    summary="Request Password Reset",
    description="Request password reset email"
)
async def request_password_reset(reset_data: PasswordResetRequest):
    """
    Request password reset.
    
    Sends password reset email with token.
    Note: Returns success even if email doesn't exist (security).
    """
    result = await process_password_reset_request(reset_data.email)
    
    if result["user_found"]:
        logger.info(f"Password reset requested for: {result['email']}")
        logger.info(f"Reset token (dev only): {result['reset_token']}")
    else:
        logger.warning(f"Password reset requested for non-existent email: {result['email']}")
    
    return MessageResponse(
        message="If the email exists, a password reset link has been sent"
    )

@sync_to_async
def reset_user_password(user_id: str, new_password: str):
    """
    Synchronous function to reset user password.
    """
    # Get user
    try:
        user = User.objects.get(id=user_id, is_active=True)
    except User.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found"
        )
    
    # Set new password
    user.set_password(new_password)
    user.save()
    
    return user.username  # For logging


@router.post(
    "/password-reset",
    response_model=MessageResponse,
    summary="Reset Password",
    description="Reset password using token from email"
)
async def reset_password(reset_data: PasswordReset):
    """Reset password using reset token."""
    
    # Verify token
    user_id = verify_password_reset_token(reset_data.token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Reset password
    username = await reset_user_password(user_id, reset_data.new_password)
    
    logger.info(f"Password reset completed for: {username}")
    
    return MessageResponse(
        message="Password reset successfully"
    )


# ============= Account Status =============

@router.get(
    "/verify",
    response_model=MessageResponse,
    summary="Verify Token",
    description="Verify if current token is valid"
)
async def verify_token(user: User = Depends(get_current_user)):
    """Verify if token is valid."""
    return MessageResponse(
        message=f"Token is valid for user: {user.username}"
    )


# Import datetime for last_login
from datetime import datetime
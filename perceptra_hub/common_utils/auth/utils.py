"""
JWT token utilities for authentication.
"""
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from django.conf import settings
from django.contrib.auth import get_user_model
import secrets

User = get_user_model()

# JWT Configuration
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days


def create_access_token(
    user_id: str,
    username: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        user_id: User UUID
        username: Username
        email: User email
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.now(tz=timezone.utc) + expires_delta
    else:
        expire = datetime.now(tz=timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "user_id": str(user_id),
        "username": username,
        "email": email,
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
        "type": "access"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """
    Create JWT refresh token.
    
    Args:
        user_id: User UUID
    
    Returns:
        Encoded JWT refresh token
    """
    expire = datetime.now(tz=timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "user_id": str(user_id),
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # Unique token ID
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload
    
    Raises:
        jwt.ExpiredSignatureError: Token has expired
        jwt.InvalidTokenError: Token is invalid
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify access token and return payload.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload or None if invalid
    """
    try:
        payload = decode_token(token)
        
        # Verify token type
        if payload.get("type") != "access":
            return None
        
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def verify_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify refresh token and return payload.
    
    Args:
        token: JWT refresh token string
    
    Returns:
        Decoded payload or None if invalid
    """
    try:
        payload = decode_token(token)
        
        # Verify token type
        if payload.get("type") != "refresh":
            return None
        
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Get token expiration datetime.
    
    Args:
        token: JWT token string
    
    Returns:
        Expiration datetime or None if invalid
    """
    try:
        payload = decode_token(token)
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            return datetime.fromtimestamp(exp_timestamp)
        return None
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def generate_password_reset_token(user_id: str) -> str:
    """
    Generate password reset token.
    
    Args:
        user_id: User UUID
    
    Returns:
        Password reset token (valid for 1 hour)
    """
    expire = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    
    payload = {
        "user_id": str(user_id),
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
        "type": "password_reset"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify password reset token.
    
    Args:
        token: Password reset token
    
    Returns:
        User ID if valid, None otherwise
    """
    try:
        payload = decode_token(token)
        
        if payload.get("type") != "password_reset":
            return None
        
        return payload.get("user_id")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def generate_email_verification_token(user_id: str, email: str) -> str:
    """
    Generate email verification token.
    
    Args:
        user_id: User UUID
        email: Email to verify
    
    Returns:
        Email verification token (valid for 24 hours)
    """
    expire = datetime.now(tz=timezone.utc) + timedelta(hours=24)
    
    payload = {
        "user_id": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
        "type": "email_verification"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_email_verification_token(token: str) -> Optional[Dict[str, str]]:
    """
    Verify email verification token.
    
    Args:
        token: Email verification token
    
    Returns:
        Dict with user_id and email if valid, None otherwise
    """
    try:
        payload = decode_token(token)
        
        if payload.get("type") != "email_verification":
            return None
        
        return {
            "user_id": payload.get("user_id"),
            "email": payload.get("email")
        }
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def create_tokens_for_user(user: User) -> Dict[str, Any]:
    """
    Create access and refresh tokens for a user.
    
    Args:
        user: Django User instance
    
    Returns:
        Dict with access_token, refresh_token, and expiry info
    """
    access_token = create_access_token(
        user_id=str(user.id),
        username=user.username,
        email=user.email
    )
    
    refresh_token = create_refresh_token(user_id=str(user.id))
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
    }
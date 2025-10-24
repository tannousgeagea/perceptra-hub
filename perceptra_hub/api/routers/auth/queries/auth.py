from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from django.contrib.auth import authenticate
from django.contrib.auth.models import update_last_login
from django.conf import settings
from datetime import datetime, timedelta, timezone
from jose import jwt
import os

# Django settings for JWT
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")  # Use Django's secret key or create a separate key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(user_id: int):
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM), expire

def create_refresh_token(user_id: int):
    expire = datetime.now(tz=timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM), expire

router = APIRouter()

# Login request model
class LoginRequest(BaseModel):
    email: str
    password: str

# Login response model
class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_at: str

# Login API
@router.post("/login", response_model=LoginResponse)
def login(data: LoginRequest):
    user = authenticate(username=data.email, password=data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    update_last_login(None, user)
    access_token, access_expiry  = create_access_token(user.id)
    refresh_token, _ = create_refresh_token(user.id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_at": access_expiry.isoformat()
    }


class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/refresh", response_model=LoginResponse)
def refresh_token(data: RefreshRequest):
    try:
        payload = jwt.decode(data.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid token type")
        user_id = payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Optionally check if user still exists
    from users.models import CustomUser
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        raise HTTPException(status_code=404, detail="User not found")

    new_access_token, access_expiry = create_access_token(user.id)
    new_refresh_token, _ = create_refresh_token(user.id)

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_at": access_expiry.isoformat()
    }

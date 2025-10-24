from fastapi import APIRouter, Depends
from .dependencies import get_current_user
from users.models import CustomUser as User
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class UserOut(BaseModel):
    id: int
    username: Optional[str] = None
    email: str
    avatar: str | None = None

@router.get("/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
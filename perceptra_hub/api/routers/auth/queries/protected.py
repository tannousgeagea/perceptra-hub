from fastapi import APIRouter, Depends
from .dependencies import get_current_user

router = APIRouter()

@router.get("/protected")
async def protected_route(user=Depends(get_current_user)):
    return {"message": f"Welcome {user.username}!"}


from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional, Dict, Any

from compute.models import Agent
from compute.services.agent_manager import AgentManager
from asgiref.sync import sync_to_async


# ============= Helper Functions =============

async def authenticate_agent_from_header(
    x_agent_key: Optional[str] = Header(None),
    x_agent_secret: Optional[str] = Header(None)
) -> Agent:
    """Authenticate agent from headers"""
    if not x_agent_key or not x_agent_secret:
        raise HTTPException(401, "Missing agent credentials")
    
    @sync_to_async
    def auth():
        return AgentManager.authenticate_agent(x_agent_key, x_agent_secret)
    
    agent = await auth()
    
    if not agent:
        raise HTTPException(401, "Invalid agent credentials")
    
    return agent
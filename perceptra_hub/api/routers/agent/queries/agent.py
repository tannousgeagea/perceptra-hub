"""
Agent Management API Endpoints
File: api/routers/agents.py
"""
from os import getenv as env
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from api.dependencies import get_request_context, RequestContext
from compute.models import Agent, AgentAPIKey, ComputeProvider
from compute.services.agent_manager import AgentManager
from django.db.models import Q
from asgiref.sync import sync_to_async
from api.routers.agent.schemas import *
from api.routers.agent.utils import *

router = APIRouter(prefix="/agents",)
BASE_URL = env('BASE_URL', 'http://localhost:8000')
# ============= User-Facing Endpoints (Require Auth) =============

@router.post("/register", response_model=RegisterAgentResponse)
async def register_agent(
    request: RegisterAgentRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Register a new on-premise agent.
    Returns API key (only shown once!) and install command.
    """
    # Convert Pydantic models to dicts
    gpu_info = [gpu.model_dump() for gpu in request.gpu_info]
    system_info = request.system_info.model_dump()
    
    @sync_to_async
    def register():
        return AgentManager.register_agent(
            organization_id=ctx.organization.id,
            name=request.name,
            gpu_info=gpu_info,
            system_info=system_info,
            created_by_id=ctx.user.id
        )
    
    agent, secret_key = await register()
    
    # Get API key
    @sync_to_async
    def get_key():
        return AgentAPIKey.objects.get(agent=agent, is_active=True)
    
    api_key = await get_key()
    
    # Generate install command
    install_command = _generate_install_command(
        api_url=BASE_URL,
        key_id=api_key.key_id,
        secret_key=secret_key,
        agent_id=agent.agent_id
    )
    
    return RegisterAgentResponse(
        agent_id=agent.agent_id,
        name=agent.name,
        api_key=api_key.key_id,
        secret_key=secret_key,
        install_command=install_command,
        status=agent.status
    )


@router.get("/", response_model=List[AgentListResponse])
async def list_agents(
    status: Optional[str] = None,
    ctx: RequestContext = Depends(get_request_context)
):
    """List all agents for organization"""
    
    @sync_to_async
    def get_agents():
        query = Agent.objects.filter(
            organization__org_id=ctx.organization.org_id
        ).order_by('-created_at')
        
        if status:
            query = query.filter(status=status)
        
        agents = list(query)
        
        # Get active jobs for each
        results = []
        for agent in agents:
            active_jobs = AgentManager.get_agent_active_jobs(agent)
            results.append(AgentListResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                status=agent.status,
                is_online=agent.is_online,
                gpu_count=agent.gpu_count,
                active_jobs=len(active_jobs),
                max_concurrent_jobs=agent.max_concurrent_jobs,
                last_heartbeat=agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                created_at=agent.created_at.isoformat()
            ))
        
        return results
    
    return await get_agents()


@router.get("/{agent_id}", response_model=AgentStatsResponse)
async def get_agent_stats(
    agent_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Get detailed stats for specific agent"""
    
    @sync_to_async
    def get_stats():
        try:
            agent = Agent.objects.get(
                agent_id=agent_id,
                organization__org_id=ctx.organization.org_id
            )
        except Agent.DoesNotExist:
            raise HTTPException(404, "Agent not found")
        
        stats = AgentManager.get_agent_stats(agent)
        
        return AgentStatsResponse(
            **stats,
            created_at=agent.created_at.isoformat()
        )
    
    return await get_stats()


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Delete agent (only if no active jobs)"""
    
    @sync_to_async
    def delete():
        try:
            agent = Agent.objects.get(
                agent_id=agent_id,
                organization__org_id=ctx.organization.org_id
            )
        except Agent.DoesNotExist:
            raise HTTPException(404, "Agent not found")
        
        # Check permissions
        if not ctx.has_role('admin', 'owner'):
            if agent.created_by_id != ctx.user.id:
                raise HTTPException(403, "Permission denied")
        
        # Check for active jobs
        active_jobs = AgentManager.get_agent_active_jobs(agent)
        if active_jobs:
            raise HTTPException(
                400,
                f"Cannot delete agent with {len(active_jobs)} active jobs"
            )
        
        # Deactivate API keys
        AgentAPIKey.objects.filter(agent=agent).update(is_active=False)
        
        # Delete agent
        agent_name = agent.name
        agent.delete()
        
        return {"message": f"Agent {agent_name} deleted"}
    
    return await delete()


@router.post("/keys/{key_id}/revoke")
async def revoke_api_key(
    key_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Revoke agent API key"""
    
    @sync_to_async
    def revoke():
        try:
            api_key = AgentAPIKey.objects.select_related('agent').get(
                key_id=key_id,
                organization__org_id=ctx.organization.org_id
            )
        except AgentAPIKey.DoesNotExist:
            raise HTTPException(404, "API key not found")
        
        # Check permissions
        if not ctx.has_role('admin', 'owner'):
            if api_key.created_by_id != ctx.user.id:
                raise HTTPException(403, "Permission denied")
        
        api_key.is_active = False
        api_key.save()
        
        # Mark agent offline
        if api_key.agent:
            api_key.agent.status = 'offline'
            api_key.agent.save()
        
        return {"message": f"API key {key_id} revoked"}
    
    return await revoke()


@router.post("/keys/{agent_id}/regenerate", response_model=RegisterAgentResponse)
async def regenerate_api_key(
    agent_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Regenerate API key for agent (revokes old key)"""
    
    @sync_to_async
    def regenerate():
        try:
            agent = Agent.objects.get(
                agent_id=agent_id,
                organization__org_id=ctx.organization.org_id
            )
        except Agent.DoesNotExist:
            raise HTTPException(404, "Agent not found")
        
        # Check permissions
        if not ctx.has_role('admin', 'owner'):
            if agent.created_by_id != ctx.user.id:
                raise HTTPException(403, "Permission denied")
        
        # Revoke old keys
        AgentAPIKey.objects.filter(agent=agent).update(is_active=False)
        
        # Generate new key
        key_id, secret_key, key_hash = AgentAPIKey.generate_key()
        
        api_key = AgentAPIKey.objects.create(
            key_id=key_id,
            key_hash=key_hash,
            organization=ctx.organization,
            agent=agent,
            name=f"{agent.name} Key (Regenerated)",
            created_by_id=ctx.user.id
        )
        
        return agent, api_key, secret_key
    
    agent, api_key, secret_key = await regenerate()
    
    # Generate new install command
    install_command = _generate_install_command(
        api_url=BASE_URL,
        key_id=api_key.key_id,
        secret_key=secret_key,
        agent_id=agent.agent_id
    )
    
    return RegisterAgentResponse(
        agent_id=agent.agent_id,
        name=agent.name,
        api_key=api_key.key_id,
        secret_key=secret_key,
        install_command=install_command,
        status=agent.status
    )


# ============= Agent-Facing Endpoints (Agent Auth Required) =============

@router.post("/heartbeat")
async def agent_heartbeat(
    request: HeartbeatRequest,
    agent: Agent = Depends(authenticate_agent_from_header)
):
    """Agent sends heartbeat (every 30 seconds)"""
    
    gpu_info = [gpu.model_dump() for gpu in request.gpu_info] if request.gpu_info else None
    system_info = request.system_info.model_dump() if request.system_info else None
    
    @sync_to_async
    def handle():
        AgentManager.handle_heartbeat(
            agent=agent,
            status=request.status,
            gpu_info=gpu_info,
            system_info=system_info
        )
    
    await handle()
    
    return {"status": "ok", "message": "Heartbeat received"}


@router.get("/poll/job", response_model=PollJobResponse)
async def poll_for_job(
    agent: Agent = Depends(authenticate_agent_from_header)
):
    """
    Agent polls for next job to execute.
    Returns job details or null if no jobs available.
    """
    
    import logging
    logging.error(f"Polling Agent")
    @sync_to_async
    def poll():
        return AgentManager.poll_job(agent)
    
    
    job_assignment = await poll()
    
    if not job_assignment:
        return PollJobResponse()
    
    return PollJobResponse(**job_assignment)


@router.post("/jobs/progress")
async def report_job_progress(
    request: JobProgressRequest,
    agent: Agent = Depends(authenticate_agent_from_header)
):
    """Agent reports training progress"""
    
    @sync_to_async
    def update():
        AgentManager.update_job_progress(
            job_id=request.job_id,
            status=request.status,
            progress=request.progress,
            metrics=request.metrics,
            error=request.error
        )
    
    await update()
    
    return {"status": "ok", "message": "Progress updated"}


@router.post("/jobs/complete")
async def complete_job(
    request: JobCompleteRequest,
    agent: Agent = Depends(authenticate_agent_from_header)
):
    """Agent reports job completion"""
    
    @sync_to_async
    def complete():
        AgentManager.complete_job(
            job_id=request.job_id,
            success=request.success,
            artifacts=request.artifacts,
            final_metrics=request.final_metrics,
            error=request.error
        )
    
    await complete()
    
    return {"status": "ok", "message": "Job completed"}


# ============= Utility Functions =============

def _generate_install_command(
    api_url: str,
    key_id: str,
    secret_key: str,
    agent_id: str
) -> str:
    """Generate one-liner install command for agent"""
    from scripts.generate_agent_install import generate_install_command
    # Clean API URL
    api_url = api_url.rstrip('/')
    
    commands = generate_install_command(
        api_url=api_url,
        agent_key=key_id,
        agent_secret=secret_key,
        agent_id=agent_id
    )
    
    # Return docker run as primary command
    return commands['docker_run']
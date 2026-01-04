
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============= Request/Response Models =============

class GPUInfo(BaseModel):
    name: str
    memory_total: int  # MB
    memory_free: int  # MB
    uuid: str
    cuda_compute_capability: Optional[str] = None

class SystemInfo(BaseModel):
    os: str
    cpu_count: int
    memory_total: int  # MB
    python_version: str
    cuda_version: Optional[str] = None
    docker_version: Optional[str] = None

class RegisterAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    gpu_info: List[GPUInfo]
    system_info: SystemInfo

class RegisterAgentResponse(BaseModel):
    agent_id: str
    name: str
    api_key: str  # key_id
    secret_key: str  # Only returned once!
    install_command: str
    status: str

class AgentStatsResponse(BaseModel):
    agent_id: str
    name: str
    status: str
    is_online: bool
    gpu_count: int
    gpu_info: List[Dict]
    system_info: Dict
    active_jobs: int
    max_concurrent_jobs: int
    completed_jobs: int
    failed_jobs: int
    last_heartbeat: Optional[str]
    uptime_seconds: Optional[float]
    created_at: str

class AgentListResponse(BaseModel):
    agent_id: str
    name: str
    status: str
    is_online: bool
    gpu_count: int
    gpu_info: List[Dict]
    system_info: Dict
    active_jobs: int
    max_concurrent_jobs: int
    last_heartbeat: Optional[str]
    uptime_seconds: Optional[float]
    created_at: str

class HeartbeatRequest(BaseModel):
    status: str  # ready, busy, error
    gpu_info: Optional[List[GPUInfo]] = None
    system_info: Optional[SystemInfo] = None

class JobProgressRequest(BaseModel):
    job_id: str
    status: str  # running, completed, failed
    progress: float  # 0-100
    metrics: Dict[str, Any]
    error: Optional[str] = None

class JobCompleteRequest(BaseModel):
    job_id: str
    success: bool
    artifacts: Dict[str, str]  # checkpoint_key, onnx_key, logs_key
    final_metrics: Dict[str, Any]
    error: Optional[str] = None

class PollJobResponse(BaseModel):
    job_id: Optional[str] = None
    training_session_id: Optional[str] = None
    model_version_id: Optional[str] = None
    organization_id: Optional[str] = None
    framework: Optional[str] = None
    task: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    dataset_version_id: Optional[str] = None
    storage_profile_id: Optional[str] = None
    assigned_at: Optional[str] = None

class RevokeKeyRequest(BaseModel):
    key_id: str

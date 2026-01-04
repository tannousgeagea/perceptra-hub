"""
Agent Manager Service - Handles agent lifecycle and job assignment.
File: compute/services/agent_manager.py
"""
import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import timedelta

from django.utils import timezone
from django.core.cache import cache
from django.db import transaction

from compute.models import Agent, AgentAPIKey, TrainingJob
from training.models import TrainingSession

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Manages on-premise training agents.
    
    Responsibilities:
    - Agent registration and authentication
    - Job assignment to available agents
    - Heartbeat monitoring
    - Status tracking
    """
    
    # Redis keys
    JOBS_QUEUE_KEY = "agent:jobs:queue"  # List of pending job IDs
    JOB_ASSIGNMENT_KEY = "agent:job:{job_id}:assignment"  # Hash of job assignment
    AGENT_STATUS_KEY = "agent:{agent_id}:status"  # Hash of agent status
    
    # Timeouts
    HEARTBEAT_TIMEOUT = 120  # 2 minutes
    JOB_CLAIM_TIMEOUT = 300  # 5 minutes
    
    @classmethod
    def register_agent(
        cls,
        organization_id: str,
        name: str,
        gpu_info: List[Dict],
        system_info: Dict,
        created_by_id: int
    ) -> tuple[Agent, str]:
        """
        Register a new agent and generate API key.
        
        Returns:
            (Agent instance, secret_key)
        """
        from organizations.models import Organization
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        
        with transaction.atomic():
            # Create agent
            agent = Agent.objects.create(
                agent_id=f"agent_{uuid.uuid4().hex[:16]}",
                organization_id=organization_id,
                name=name,
                status='pending',
                gpu_info=gpu_info,
                system_info=system_info,
                created_by_id=created_by_id,
                max_concurrent_jobs=len(gpu_info) if gpu_info else 1
            )
            
            # Generate API key
            key_id, secret_key, key_hash = AgentAPIKey.generate_key()
            
            api_key = AgentAPIKey.objects.create(
                key_id=key_id,
                key_hash=key_hash,
                organization_id=organization_id,
                agent=agent,
                name=f"{name} Key",
                created_by_id=created_by_id
            )
            
            logger.info(f"Registered agent {agent.agent_id} for org {organization_id}")
            
            return agent, secret_key
    
    @classmethod
    def authenticate_agent(cls, key_id: str, secret_key: str) -> Optional[Agent]:
        """
        Authenticate agent using API key.
        
        Returns:
            Agent instance if valid, None otherwise
        """
        try:
            api_key = AgentAPIKey.objects.select_related('agent').get(
                key_id=key_id,
                is_active=True
            )
            
            # Check expiration
            if api_key.is_expired:
                logger.warning(f"Agent key {key_id} expired")
                return None
            
            # Verify secret
            if not api_key.verify_secret(secret_key):
                logger.warning(f"Invalid secret for key {key_id}")
                return None
            
            # Update last used
            api_key.last_used = timezone.now()
            api_key.save(update_fields=['last_used'])
            
            agent = api_key.agent
            
            # Update agent status if offline
            if agent.status == 'offline':
                agent.status = 'ready'
                agent.save(update_fields=['status'])
            
            logger.debug(f"Authenticated agent {agent.agent_id}")
            return agent
            
        except AgentAPIKey.DoesNotExist:
            logger.warning(f"Invalid key_id: {key_id}")
            return None
    
    @classmethod
    def handle_heartbeat(
        cls,
        agent: Agent,
        status: str,
        gpu_info: Optional[List[Dict]] = None,
        system_info: Optional[Dict] = None
    ):
        """
        Process agent heartbeat.
        Updates agent status and checks for stale agents.
        """
        from django.db.models import F
        
        update_fields = ['last_heartbeat', 'updated_at']
        
        # Update heartbeat
        agent.last_heartbeat = timezone.now()
        
        # Update status if changed
        if status != agent.status:
            agent.status = status
            update_fields.append('status')
        
        # Update GPU info if provided
        if gpu_info is not None:
            agent.gpu_info = gpu_info
            update_fields.append('gpu_info')
        
        # Update system info if provided
        if system_info is not None:
            agent.system_info = system_info
            update_fields.append('system_info')
        
        agent.save(update_fields=update_fields)
        
        # Cache agent status for quick checks
        cache.set(
            cls.AGENT_STATUS_KEY.format(agent_id=agent.agent_id),
            {
                'status': status,
                'gpu_count': agent.gpu_count,
                'last_heartbeat': agent.last_heartbeat.isoformat()
            },
            timeout=cls.HEARTBEAT_TIMEOUT
        )
        
        logger.info(f"Heartbeat from agent {agent.agent_id}: {status}")
    
    @classmethod
    def assign_job_to_agent(
        cls,
        training_job: TrainingJob,
        agent: Agent
    ) -> Dict[str, Any]:
        """
        Assign training job to specific agent.
        Creates Redis entry for agent to pick up.
        
        Returns:
            Job assignment dict
        """
        from django.core.cache import cache
        import json
        
        # Build job config
        job_assignment = {
            'job_id': training_job.job_id,
            'training_session_id': training_job.training_session.session_id,
            'model_version_id': training_job.training_session.model_version.version_id,
            'organization_id': training_job.training_session.model_version.model.organization.org_id,
            'framework': training_job.training_session.model_version.model.framework.name,
            'task': training_job.training_session.model_version.model.task.name,
            'config': training_job.training_session.config,
            'dataset_version_id': training_job.training_session.model_version.dataset_version.id if training_job.training_session.model_version.dataset_version else None,
            'storage_profile_id': training_job.training_session.model_version.storage_profile.profile_id,
            'assigned_at': timezone.now().isoformat(),
            'agent_id': agent.agent_id
        }
        
        # Store in Redis for agent to pick up
        assignment_key = cls.JOB_ASSIGNMENT_KEY.format(job_id=training_job.job_id)
        cache.set(assignment_key, job_assignment, timeout=86400)  # 24 hours
        
        # Add to agent's job queue
        queue_key = f"agent:{agent.agent_id}:jobs"
        cache.lpush(queue_key, training_job.job_id)
        
        # Update training session
        training_job.training_session.status = 'queued'
        training_job.training_session.compute_resource = f"agent/{agent.name}"
        training_job.training_session.save()
        
        # Update job
        training_job.external_job_id = f"agent:{agent.agent_id}:{training_job.job_id}"
        training_job.save()
        
        logger.info(f"Assigned job {training_job.job_id} to agent {agent.agent_id}")
        
        return job_assignment
    
    @classmethod
    def get_available_agent(
        cls,
        organization_id: str,
        required_gpus: int = 1
    ) -> Optional[Agent]:
        """
        Find available agent for organization.
        
        Priority:
        1. Ready status
        2. GPU availability
        3. Least loaded
        """
        # Get online agents for org
        cutoff = timezone.now() - timedelta(seconds=cls.HEARTBEAT_TIMEOUT)
        
        agents = Agent.objects.filter(
            organization__org_id=organization_id,
            status__in=['ready', 'busy'],
            last_heartbeat__gte=cutoff
        ).order_by('status', 'id')
        
        for agent in agents:
            # Check GPU count
            if agent.gpu_count < required_gpus:
                continue
            
            # Check if agent has capacity
            active_jobs = cls.get_agent_active_jobs(agent)
            if len(active_jobs) < agent.max_concurrent_jobs:
                logger.debug(f"Found available agent: {agent.agent_id}")
                return agent
        
        logger.warning(f"No available agents for org {organization_id}")
        return None
    
    @classmethod
    def get_agent_active_jobs(cls, agent: Agent) -> List[str]:
        """Get list of active job IDs for agent"""
        queue_key = f"agent:{agent.agent_id}:jobs"
        job_ids = cache.get(queue_key, 0, -1) or []
        return [jid.decode() if isinstance(jid, bytes) else jid for jid in job_ids]
    
    @classmethod
    def poll_job(cls, agent: Agent) -> Optional[Dict[str, Any]]:
        """
        Agent polls for next job to execute.
        Returns job assignment or None.
        """
        queue_key = f"agent:{agent.agent_id}:jobs"
        
        # Pop job from queue (blocking pop with timeout)
        job_id = cache.get(queue_key)
        
        if not job_id:
            return None
        
        if isinstance(job_id, bytes):
            job_id = job_id.decode()
        
        # Get job assignment
        assignment_key = cls.JOB_ASSIGNMENT_KEY.format(job_id=job_id)
        assignment = cache.get(assignment_key)
        
        if not assignment:
            logger.error(f"Job assignment not found for {job_id}")
            return None
        
        logger.info(f"Agent {agent.agent_id} claimed job {job_id}")
        return assignment
    
    @classmethod
    def update_job_progress(
        cls,
        job_id: str,
        status: str,
        progress: float,
        metrics: Dict[str, Any],
        error: Optional[str] = None
    ):
        """
        Update training job progress from agent.
        """
        try:
            training_job = TrainingJob.objects.select_related(
                'training_session'
            ).get(job_id=job_id)
            
            session = training_job.training_session
            
            # Update session
            update_fields = []
            
            if session.status != status:
                session.status = status
                update_fields.append('status')
            
            if session.progress != progress:
                session.progress = progress
                update_fields.append('progress')
            
            if metrics:
                session.metrics = {**session.metrics, **metrics}
                update_fields.append('metrics')
            
            if error and session.error_message != error:
                session.error_message = error
                update_fields.append('error_message')
            
            if status == 'running' and not session.started_at:
                session.started_at = timezone.now()
                update_fields.append('started_at')
            
            if status in ['completed', 'failed', 'cancelled'] and not session.completed_at:
                session.completed_at = timezone.now()
                update_fields.append('completed_at')
            
            if update_fields:
                update_fields.append('updated_at')
                session.save(update_fields=update_fields)
            
            logger.debug(f"Updated job {job_id}: {status} ({progress}%)")
            
        except TrainingJob.DoesNotExist:
            logger.error(f"Training job not found: {job_id}")
    
    @classmethod
    def complete_job(
        cls,
        job_id: str,
        success: bool,
        artifacts: Dict[str, str],
        final_metrics: Dict[str, Any],
        error: Optional[str] = None
    ):
        """
        Mark job as complete and store artifacts.
        """
        try:
            training_job = TrainingJob.objects.select_related(
                'training_session',
                'training_session__model_version'
            ).get(job_id=job_id)
            
            session = training_job.training_session
            model_version = session.model_version
            
            with transaction.atomic():
                # Update session
                session.status = 'completed' if success else 'failed'
                session.progress = 100.0 if success else session.progress
                session.completed_at = timezone.now()
                session.metrics = {**session.metrics, **final_metrics}
                
                if error:
                    session.error_message = error
                
                session.save()
                
                # Update model version with artifacts
                if success and artifacts:
                    model_version.checkpoint_key = artifacts.get('checkpoint_key', '')
                    model_version.onnx_model_key = artifacts.get('onnx_key', '')
                    model_version.training_logs_key = artifacts.get('logs_key', '')
                    model_version.status = 'trained'
                    model_version.metrics = final_metrics
                    model_version.save()
                    
                    logger.info(f"Job {job_id} completed successfully")
                else:
                    model_version.status = 'failed'
                    model_version.error_message = error or 'Training failed'
                    model_version.save()
                    
                    logger.error(f"Job {job_id} failed: {error}")
                
                # Update job
                training_job.completed_at = timezone.now()
                training_job.save()
                
                # Clean up Redis
                assignment_key = cls.JOB_ASSIGNMENT_KEY.format(job_id=job_id)
                cache.delete(assignment_key)
                
        except TrainingJob.DoesNotExist:
            logger.error(f"Training job not found: {job_id}")
    
    @classmethod
    def mark_stale_agents_offline(cls):
        """
        Mark agents as offline if no heartbeat received.
        Should be run periodically (e.g., every minute via Celery beat).
        """
        cutoff = timezone.now() - timedelta(seconds=cls.HEARTBEAT_TIMEOUT)
        
        stale_agents = Agent.objects.filter(
            status__in=['ready', 'busy'],
            last_heartbeat__lt=cutoff
        )
        
        count = stale_agents.update(status='offline')
        
        if count > 0:
            logger.warning(f"Marked {count} agents as offline")
        
        return count
    
    @classmethod
    def get_agent_stats(cls, agent: Agent) -> Dict[str, Any]:
        """Get agent statistics"""
        active_jobs = cls.get_agent_active_jobs(agent)
        
        # Get completed jobs count
        completed_count = TrainingJob.objects.filter(
            actual_provider__provider_type='on-premise-agent',
            external_job_id__contains=agent.agent_id,
            training_session__status='completed'
        ).count()
        
        # Get failed jobs count
        failed_count = TrainingJob.objects.filter(
            actual_provider__provider_type='on-premise-agent',
            external_job_id__contains=agent.agent_id,
            training_session__status='failed'
        ).count()
        
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'status': agent.status,
            'is_online': agent.is_online,
            'gpu_count': agent.gpu_count,
            'gpu_info': agent.gpu_info,
            'system_info': agent.system_info,
            'active_jobs': len(active_jobs),
            'max_concurrent_jobs': agent.max_concurrent_jobs,
            'completed_jobs': completed_count,
            'failed_jobs': failed_count,
            'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            'uptime_seconds': (timezone.now() - agent.last_heartbeat).total_seconds() if agent.last_heartbeat else None
        }
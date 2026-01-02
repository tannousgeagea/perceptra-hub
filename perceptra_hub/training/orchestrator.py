"""
Production-grade training orchestrator with intelligent routing and fallback.
Handles provider selection, failure recovery, and cost optimization.
"""
import uuid
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

from django.db import transaction
from django.utils import timezone
from django.core.cache import cache

from ml_models.models import ModelVersion
from training.models import TrainingSession
from compute.models import (
    ComputeProfile, ComputeProvider, TrainingJob, ComputeFallback
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates training job submission with intelligent routing.
    
    Key responsibilities:
    - Select optimal compute provider based on strategy
    - Implement fallback logic on provider failure
    - Enforce cost/resource limits
    - Track job lifecycle
    """
    
    def __init__(self, model_version: ModelVersion):
        self.model_version = model_version
        self.organization = model_version.model.organization
        self.model = model_version.model
        
    def submit_training(
        self,
        training_session: TrainingSession,
        compute_profile_id: Optional[str] = None
    ) -> TrainingJob:
        """
        Submit training job to appropriate compute resource.
        
        Flow:
        1. Select compute profile (specified or default)
        2. Validate profile and check availability
        3. Select provider based on strategy
        4. Submit job with fallback on failure
        5. Record job in database
        
        Returns:
            TrainingJob instance
            
        Raises:
            ValueError: If no valid compute profile
            RuntimeError: If all providers fail
        """
        logger.info(
            f"Orchestrating training for model_version={self.model_version.version_id}, "
            f"session={training_session.session_id}"
        )
        
        # Step 1: Select compute profile
        compute_profile = self._select_compute_profile(compute_profile_id)
        logger.info(f"Selected compute profile: {compute_profile.name}")
        
        # Step 2: Validate and prepare
        self._validate_profile(compute_profile, training_session)
        
        # Step 3: Select provider with fallback support
        provider, instance_type = self._select_provider_with_fallback(
            compute_profile,
            training_session
        )
        logger.info(f"Selected provider: {provider.name} ({instance_type})")
        
        # Step 4: Create job record (atomic operation)
        with transaction.atomic():
            training_job = TrainingJob.objects.create(
                job_id=str(uuid.uuid4()),
                training_session=training_session,
                compute_profile=compute_profile,
                actual_provider=provider,
                instance_type=instance_type,
                estimated_cost=self._estimate_cost(provider, instance_type, training_session)
            )
            
            # Update session status
            training_session.status = 'queued'
            training_session.compute_resource = f"{provider.name}/{instance_type}"
            training_session.save()
        
        # Step 5: Submit to provider (outside transaction)
        try:
            external_job_id = self._submit_to_provider(
                provider,
                training_job,
                compute_profile
            )
            
            # Update with external ID
            training_job.external_job_id = external_job_id
            training_job.started_at = timezone.now()
            training_job.save()
            
            logger.info(
                f"Successfully submitted job {training_job.job_id} "
                f"to {provider.name} (external_id={external_job_id})"
            )
            
        except Exception as e:
            logger.error(f"Failed to submit job to {provider.name}: {e}")
            # Mark job as failed
            training_job.training_session.status = 'failed'
            training_job.training_session.error_message = f"Submission failed: {str(e)}"
            training_job.training_session.save()
            raise RuntimeError(f"Job submission failed: {str(e)}")
        
        return training_job
    
    def _select_compute_profile(self, profile_id: Optional[str]) -> ComputeProfile:
        """
        Select compute profile to use.
        Priority: specified > default > first active
        """
        if profile_id:
            try:
                profile = ComputeProfile.objects.select_related('provider').get(
                    profile_id=profile_id,
                    organization=self.organization,
                    is_active=True
                )
                logger.debug(f"Using specified profile: {profile.name}")
                return profile
            except ComputeProfile.DoesNotExist:
                raise ValueError(f"Compute profile {profile_id} not found or inactive")
        
        # Try default profile
        profile = ComputeProfile.objects.filter(
            organization=self.organization,
            is_default=True,
            is_active=True
        ).select_related('provider').first()
        
        if profile:
            logger.debug(f"Using default profile: {profile.name}")
            return profile
        
        # Fallback to any active profile
        profile = ComputeProfile.objects.filter(
            organization=self.organization,
            is_active=True
        ).select_related('provider').first()
        
        if profile:
            logger.warning(f"No default profile, using: {profile.name}")
            return profile
        
        raise ValueError("No active compute profile configured for organization")
    
    def _validate_profile(
        self,
        profile: ComputeProfile,
        session: TrainingSession
    ):
        """Validate profile configuration and limits"""
        
        # Check if credentials required and present
        if profile.provider.requires_user_credentials:
            if not profile.user_credentials:
                raise ValueError(
                    f"Provider {profile.provider.name} requires credentials but none configured"
                )
        
        # Check concurrent job limit
        active_jobs_count = self._get_active_jobs_count(profile)
        if active_jobs_count >= profile.max_concurrent_jobs:
            logger.warning(
                f"Profile {profile.name} at capacity: "
                f"{active_jobs_count}/{profile.max_concurrent_jobs} jobs"
            )
            # Don't fail - queue the job and let provider handle queuing
        
        # Validate training config doesn't exceed limits
        if session.config.get('max_epochs'):
            estimated_hours = session.config['max_epochs'] * 0.5  # Rough estimate
            if estimated_hours > profile.max_training_hours:
                raise ValueError(
                    f"Training may exceed max hours limit: "
                    f"{estimated_hours:.1f}h > {profile.max_training_hours}h"
                )
    
    def _select_provider_with_fallback(
        self,
        profile: ComputeProfile,
        session: TrainingSession
    ) -> Tuple[ComputeProvider, str]:
        """
        Select provider based on strategy with fallback support.
        
        Strategies:
        - queue: Always use platform GPU (wait if busy)
        - cheapest: Select cheapest available provider
        - fastest: Select fastest GPU available
        - preferred: Use primary, fallback if unavailable
        """
        strategy = profile.strategy
        logger.debug(f"Using strategy: {strategy}")
        
        if strategy == 'queue':
            # Always use primary provider (platform GPU), queue if needed
            return profile.provider, profile.default_instance_type
        
        # For other strategies, try providers in order
        providers_to_try = self._get_providers_by_strategy(profile, strategy)
        
        for provider, instance_type in providers_to_try:
            if self._is_provider_available(provider, profile):
                logger.debug(f"Provider {provider.name} is available")
                return provider, instance_type
            else:
                logger.debug(f"Provider {provider.name} not available, trying next")
        
        # All providers unavailable - use primary anyway (will queue)
        logger.warning("All providers unavailable, using primary with queueing")
        return profile.provider, profile.default_instance_type
    
    def _get_providers_by_strategy(
        self,
        profile: ComputeProfile,
        strategy: str
    ) -> list[Tuple[ComputeProvider, str]]:
        """
        Get ordered list of providers to try based on strategy.
        Returns list of (provider, instance_type) tuples.
        """
        providers = []
        
        if strategy == 'preferred':
            # Primary first, then fallbacks in order
            providers.append((profile.provider, profile.default_instance_type))
            
            fallbacks = ComputeFallback.objects.filter(
                profile=profile
            ).select_related('provider').order_by('priority')
            
            for fallback in fallbacks:
                if fallback.provider.is_active:
                    providers.append((fallback.provider, profile.default_instance_type))
        
        elif strategy == 'cheapest':
            # Sort by cost (platform GPU first - free/cheap)
            all_providers = [profile.provider]
            fallbacks = ComputeFallback.objects.filter(
                profile=profile
            ).select_related('provider')
            all_providers.extend([f.provider for f in fallbacks if f.provider.is_active])
            
            # Simple cost heuristic: platform < cloud
            all_providers.sort(key=lambda p: (
                0 if 'platform' in p.provider_type else 1
            ))
            providers = [(p, profile.default_instance_type) for p in all_providers]
        
        elif strategy == 'fastest':
            # Sort by GPU power (cloud GPUs typically faster)
            all_providers = [profile.provider]
            fallbacks = ComputeFallback.objects.filter(
                profile=profile
            ).select_related('provider')
            all_providers.extend([f.provider for f in fallbacks if f.provider.is_active])
            
            # Simple speed heuristic: cloud > platform
            all_providers.sort(key=lambda p: (
                1 if 'platform' in p.provider_type else 0
            ))
            providers = [(p, profile.default_instance_type) for p in all_providers]
        
        return providers
    
    def _is_provider_available(
        self,
        provider: ComputeProvider,
        profile: ComputeProfile
    ) -> bool:
        """
        Check if provider is available for immediate use.
        Uses caching to avoid repeated checks.
        """
        cache_key = f"provider_available:{provider.id}:{profile.id}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Check provider active
        if not provider.is_active:
            cache.set(cache_key, False, 60)
            return False
        
        # Check credentials if required
        if provider.requires_user_credentials and not profile.user_credentials:
            cache.set(cache_key, False, 300)  # Cache longer for config issues
            return False
        
        # For platform providers, check capacity
        if provider.provider_type in ['platform-gpu', 'platform-cpu']:
            active_jobs = self._get_active_jobs_count(profile, provider)
            available = active_jobs < profile.max_concurrent_jobs
            cache.set(cache_key, available, 30)  # Short cache for capacity
            return available
        
        # For cloud providers, assume available (they handle queuing)
        cache.set(cache_key, True, 120)
        return True
    
    def _get_active_jobs_count(
        self,
        profile: ComputeProfile,
        provider: Optional[ComputeProvider] = None
    ) -> int:
        """Get count of active training jobs for profile/provider"""
        query = TrainingJob.objects.filter(
            compute_profile=profile,
            training_session__status__in=['queued', 'initializing', 'running']
        )
        
        if provider:
            query = query.filter(actual_provider=provider)
        
        return query.count()
    
    def _estimate_cost(
        self,
        provider: ComputeProvider,
        instance_type: str,
        session: TrainingSession
    ) -> Optional[Decimal]:
        """
        Estimate training cost based on provider and expected duration.
        Returns None if cost can't be estimated.
        """
        # Platform providers - free or fixed cost
        if provider.provider_type in ['platform-gpu', 'platform-cpu']:
            return Decimal('0.00')
        
        # Get instance cost from provider config
        instances = provider.available_instances
        instance_info = next(
            (i for i in instances if i.get('name') == instance_type),
            None
        )
        
        if not instance_info or 'cost_per_hour' not in instance_info:
            return None
        
        cost_per_hour = Decimal(str(instance_info['cost_per_hour']))
        
        # Estimate hours (rough heuristic based on epochs)
        epochs = session.config.get('max_epochs', 100)
        estimated_hours = Decimal(str(epochs * 0.5))  # 30 min per epoch average
        
        return cost_per_hour * estimated_hours
    
    def _submit_to_provider(
        self,
        provider: ComputeProvider,
        training_job: TrainingJob,
        compute_profile: ComputeProfile
    ) -> str:
        """
        Submit job to specific provider using appropriate adapter.
        Returns external job ID from provider.
        """
        from compute.adapters import get_adapter_for_provider
        
        adapter = get_adapter_for_provider(provider)
        
        # Prepare job configuration
        job_config = {
            'job_id': training_job.job_id,
            'model_version_id': self.model_version.version_id,
            'model_name': self.model.name,
            'training_session_id': training_job.training_session.session_id,
            'instance_type': training_job.instance_type,
            'storage_profile': self.model_version.storage_profile,
            'dataset_version': self.model_version.dataset_version,
            'config': training_job.training_session.config,
            'organization_id': self.organization.org_id,
            'framework': self.model.framework.name,
            'task': self.model.task.name,
            'parent_version_id': (
                self.model_version.parent_version.version_id 
                if self.model_version.parent_version 
                else None
            )
        }
        
        # Decrypt credentials if needed
        credentials = None
        if provider.requires_user_credentials:
            credentials = self._decrypt_credentials(compute_profile.user_credentials)
        
        # Submit to provider
        external_job_id = adapter.submit_job(job_config, credentials)
        
        return external_job_id
    
    def _decrypt_credentials(self, encrypted_creds: Dict) -> Dict:
        """Decrypt stored credentials"""
        from cryptography.fernet import Fernet
        from django.conf import settings
        import json
        
        cipher = Fernet(settings.COMPUTE_CREDENTIALS_KEY.encode())
        decrypted = cipher.decrypt(encrypted_creds['encrypted'].encode())
        return json.loads(decrypted)
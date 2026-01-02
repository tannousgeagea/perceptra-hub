"""
On-Premise Training Agent - Lightweight worker that runs on user's GPU machine.
File: agent/main.py

This agent:
1. Authenticates with platform via API key
2. Sends heartbeat every 30 seconds
3. Polls for training jobs
4. Executes training using modular trainers
5. Reports progress and uploads artifacts
"""
import os
import sys
import time
import json
import logging
import signal
import requests
import GPUtil
import psutil
import platform
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('agent')

# Agent configuration from environment
API_URL = os.getenv('API_URL', 'http://localhost:8000')
AGENT_KEY = os.getenv('AGENT_KEY')
AGENT_SECRET = os.getenv('AGENT_SECRET')
AGENT_ID = os.getenv('AGENT_ID')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '10'))  # seconds
HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL', '30'))  # seconds

# Paths
WORK_DIR = Path('/tmp/agent-work')
DATASETS_DIR = WORK_DIR / 'datasets'
OUTPUTS_DIR = WORK_DIR / 'outputs'

# Global state
shutdown_requested = False


class AgentClient:
    """Client for communicating with platform API"""
    
    def __init__(self, api_url: str, key: str, secret: str):
        self.api_url = api_url.rstrip('/')
        self.key = key
        self.secret = secret
        self.headers = {
            'X-Agent-Key': key,
            'X-Agent-Secret': secret,
            'Content-Type': 'application/json'
        }
    
    def heartbeat(self, status: str, gpu_info: list, system_info: dict) -> bool:
        """Send heartbeat to platform"""
        try:
            response = requests.post(
                f'{self.api_url}/api/v1/agents/heartbeat',
                headers=self.headers,
                json={
                    'status': status,
                    'gpu_info': gpu_info,
                    'system_info': system_info
                },
                timeout=10
            )
            response.raise_for_status()
            logger.info('Heartbeat sent successfully')
            return True
        except Exception as e:
            logger.error(f'Failed to send heartbeat: {e}')
            return False
    
    def poll_job(self) -> Optional[Dict[str, Any]]:
        """Poll for next job"""
        try:
            response = requests.get(
                f'{self.api_url}/api/v1/agents/poll/job',
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Check if job available
            if data.get('job_id'):
                logger.info(f"Received job: {data['job_id']}")
                return data
            
            return None
        except Exception as e:
            logger.error(f'Failed to poll for job: {e}')
            return None
    
    def report_progress(
        self,
        job_id: str,
        status: str,
        progress: float,
        metrics: dict,
        error: Optional[str] = None
    ) -> bool:
        """Report job progress"""
        try:
            response = requests.post(
                f'{self.api_url}/api/v1/agents/jobs/progress',
                headers=self.headers,
                json={
                    'job_id': job_id,
                    'status': status,
                    'progress': progress,
                    'metrics': metrics,
                    'error': error
                },
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f'Progress reported: {status} ({progress}%)')
            return True
        except Exception as e:
            logger.error(f'Failed to report progress: {e}')
            return False
    
    def complete_job(
        self,
        job_id: str,
        success: bool,
        artifacts: dict,
        final_metrics: dict,
        error: Optional[str] = None
    ) -> bool:
        """Report job completion"""
        try:
            response = requests.post(
                f'{self.api_url}/api/v1/agents/jobs/complete',
                headers=self.headers,
                json={
                    'job_id': job_id,
                    'success': success,
                    'artifacts': artifacts,
                    'final_metrics': final_metrics,
                    'error': error
                },
                timeout=30
            )
            response.raise_for_status()
            logger.info(f'Job completion reported: success={success}')
            return True
        except Exception as e:
            logger.error(f'Failed to report completion: {e}')
            return False
    
    def download_dataset(self, dataset_version_id: str, storage_profile_id: str, output_path: Path) -> bool:
        """Download dataset from storage"""
        # TODO: Implement dataset download from storage
        # For now, assume dataset is pre-downloaded or accessible
        logger.warning('Dataset download not implemented, assuming data exists')
        return True
    
    def upload_artifact(
        self,
        file_path: Path,
        storage_key: str,
        storage_profile_id: str
    ) -> bool:
        """Upload training artifact to storage"""
        # TODO: Implement artifact upload to storage
        # For now, just log
        logger.warning(f'Artifact upload not implemented: {storage_key}')
        return True


class SystemMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_gpu_info() -> list:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'uuid': gpu.uuid,
                    'cuda_compute_capability': f"{gpu.driver}"
                }
                for gpu in gpus
            ]
        except Exception as e:
            logger.error(f'Failed to get GPU info: {e}')
            return []
    
    @staticmethod
    def get_system_info() -> dict:
        """Get system information"""
        try:
            return {
                'os': platform.platform(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total // (1024 * 1024),  # MB
                'python_version': platform.python_version(),
                'cuda_version': os.getenv('CUDA_VERSION', 'unknown'),
                'docker_version': 'inside-container'
            }
        except Exception as e:
            logger.error(f'Failed to get system info: {e}')
            return {}


class JobExecutor:
    """Executes training jobs"""
    
    def __init__(self, client: AgentClient):
        self.client = client
    
    def execute(self, job: Dict[str, Any]) -> tuple[bool, dict, dict, Optional[str]]:
        """
        Execute training job.
        
        Returns:
            (success, artifacts, metrics, error)
        """
        job_id = job['job_id']
        framework = job['framework']
        task = job['task']
        config = job['config']
        
        logger.info(f"Starting job {job_id}: {framework}/{task}")
        
        # Setup directories
        job_work_dir = OUTPUTS_DIR / job_id
        job_work_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_dir = DATASETS_DIR / job.get('dataset_version_id', 'default')
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Report job started
            self.client.report_progress(
                job_id=job_id,
                status='running',
                progress=0.0,
                metrics={}
            )
            
            # Download dataset (if needed)
            if job.get('dataset_version_id'):
                logger.info('Downloading dataset...')
                self.client.download_dataset(
                    dataset_version_id=job['dataset_version_id'],
                    storage_profile_id=job['storage_profile_id'],
                    output_path=dataset_dir
                )
            
            # Get trainer
            from training.trainers.factory import get_trainer
            
            trainer = get_trainer(
                framework=framework,
                task=task,
                dataset_path=str(dataset_dir),
                output_dir=str(job_work_dir),
                **config
            )
            
            # Setup progress callback
            def progress_callback(epoch: int, total_epochs: int, metrics: dict):
                progress = (epoch / total_epochs) * 100
                self.client.report_progress(
                    job_id=job_id,
                    status='running',
                    progress=progress,
                    metrics=metrics
                )
            
            trainer.set_progress_callback(progress_callback)
            
            # Train
            logger.info('Starting training...')
            result = trainer.train()
            
            # Upload artifacts
            artifacts = {}
            
            if result.get('checkpoint_path'):
                checkpoint_path = Path(result['checkpoint_path'])
                if checkpoint_path.exists():
                    storage_key = f"organizations/{job['organization_id']}/models/{job['model_version_id']}/checkpoint.pt"
                    if self.client.upload_artifact(
                        checkpoint_path,
                        storage_key,
                        job['storage_profile_id']
                    ):
                        artifacts['checkpoint_key'] = storage_key
            
            if result.get('onnx_path'):
                onnx_path = Path(result['onnx_path'])
                if onnx_path.exists():
                    storage_key = f"organizations/{job['organization_id']}/models/{job['model_version_id']}/model.onnx"
                    if self.client.upload_artifact(
                        onnx_path,
                        storage_key,
                        job['storage_profile_id']
                    ):
                        artifacts['onnx_key'] = storage_key
            
            # Training logs
            logs_path = job_work_dir / 'training.log'
            if logs_path.exists():
                storage_key = f"organizations/{job['organization_id']}/models/{job['model_version_id']}/training.log"
                if self.client.upload_artifact(
                    logs_path,
                    storage_key,
                    job['storage_profile_id']
                ):
                    artifacts['logs_key'] = storage_key
            
            logger.info(f'Job {job_id} completed successfully')
            return True, artifacts, result.get('metrics', {}), None
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Report error
            self.client.report_progress(
                job_id=job_id,
                status='failed',
                progress=0.0,
                metrics={},
                error=error_msg
            )
            
            return False, {}, {}, error_msg


class Agent:
    """Main agent class"""
    
    def __init__(self):
        self.client = AgentClient(API_URL, AGENT_KEY, AGENT_SECRET)
        self.executor = JobExecutor(self.client)
        self.monitor = SystemMonitor()
        self.last_heartbeat = 0
        self.current_job = None
    
    def run(self):
        """Main agent loop"""
        logger.info('Agent starting...')
        logger.info(f'API URL: {API_URL}')
        logger.info(f'Agent ID: {AGENT_ID}')
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initial heartbeat
        self._send_heartbeat('ready')
        
        logger.info('Agent ready, starting main loop')
        
        while not shutdown_requested:
            try:
                # Send heartbeat if needed
                if time.time() - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                    status = 'busy' if self.current_job else 'ready'
                    self._send_heartbeat(status)
                
                # Poll for job if not busy
                if not self.current_job:
                    job = self.client.poll_job()
                    
                    if job:
                        self._execute_job(job)
                    else:
                        # No job available, wait
                        time.sleep(POLL_INTERVAL)
                else:
                    # Job in progress, wait
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f'Error in main loop: {e}', exc_info=True)
                time.sleep(5)
        
        logger.info('Agent shutting down')
    
    def _send_heartbeat(self, status: str):
        """Send heartbeat to platform"""
        gpu_info = self.monitor.get_gpu_info()
        system_info = self.monitor.get_system_info()
        
        success = self.client.heartbeat(status, gpu_info, system_info)
        if success:
            self.last_heartbeat = time.time()
    
    def _execute_job(self, job: Dict[str, Any]):
        """Execute training job"""
        job_id = job['job_id']
        self.current_job = job_id
        
        try:
            logger.info(f'Executing job: {job_id}')
            
            # Execute
            success, artifacts, metrics, error = self.executor.execute(job)
            
            # Report completion
            self.client.complete_job(
                job_id=job_id,
                success=success,
                artifacts=artifacts,
                final_metrics=metrics,
                error=error
            )
            
        except Exception as e:
            logger.error(f'Failed to execute job: {e}', exc_info=True)
            self.client.complete_job(
                job_id=job_id,
                success=False,
                artifacts={},
                final_metrics={},
                error=str(e)
            )
        finally:
            self.current_job = None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        global shutdown_requested
        logger.info(f'Received signal {signum}, shutting down gracefully...')
        shutdown_requested = True


def validate_config():
    """Validate agent configuration"""
    if not AGENT_KEY:
        logger.error('AGENT_KEY environment variable not set')
        return False
    
    if not AGENT_SECRET:
        logger.error('AGENT_SECRET environment variable not set')
        return False
    
    if not AGENT_ID:
        logger.error('AGENT_ID environment variable not set')
        return False
    
    return True


def main():
    """Main entry point"""
    logger.info('='*60)
    logger.info('CV Training Agent')
    logger.info('='*60)
    
    # Validate config
    if not validate_config():
        logger.error('Configuration validation failed')
        sys.exit(1)
    
    # Create work directories
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start agent
    agent = Agent()
    
    try:
        agent.run()
    except Exception as e:
        logger.error(f'Agent crashed: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
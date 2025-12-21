"""
GPU monitoring and metrics collection.
Location: training/monitoring.py
"""
import psutil
import logging
from celery import shared_task
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False


class GPUMonitor:
    """Monitor GPU usage and health"""
    
    @staticmethod
    def get_gpu_stats() -> List[Dict]:
        """Get detailed GPU statistics"""
        gpus = []
        
        if not NVML_AVAILABLE:
            logger.warning("NVML not available, GPU stats unavailable")
            return gpus
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = pynvml.nvmlDeviceGetName(handle)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_total = mem_info.total / 1e9  # GB
                mem_used = mem_info.used / 1e9
                mem_free = mem_info.free / 1e9
                mem_util = (mem_used / mem_total) * 100
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temp = None
                
                # Power
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                except:
                    power_usage = None
                    power_limit = None
                
                # Processes
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    process_count = len(processes)
                    process_memory = sum(p.usedGpuMemory for p in processes) / 1e9
                except:
                    process_count = 0
                    process_memory = 0
                
                gpus.append({
                    'index': i,
                    'name': name.decode() if isinstance(name, bytes) else name,
                    'memory': {
                        'total_gb': round(mem_total, 2),
                        'used_gb': round(mem_used, 2),
                        'free_gb': round(mem_free, 2),
                        'utilization_percent': round(mem_util, 1)
                    },
                    'utilization_percent': gpu_util,
                    'temperature_c': temp,
                    'power': {
                        'usage_watts': round(power_usage, 1) if power_usage else None,
                        'limit_watts': round(power_limit, 1) if power_limit else None
                    },
                    'processes': {
                        'count': process_count,
                        'memory_gb': round(process_memory, 2)
                    }
                })
            
            return gpus
            
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return []
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Get system resource statistics"""
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / 1e9, 2),
                'used_gb': round(psutil.virtual_memory().used / 1e9, 2),
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total_gb': round(psutil.disk_usage('/').total / 1e9, 2),
                'used_gb': round(psutil.disk_usage('/').used / 1e9, 2),
                'percent': psutil.disk_usage('/').percent
            }
        }
    
    @staticmethod
    def check_gpu_health() -> Dict:
        """Check GPU health and return issues"""
        issues = []
        warnings = []
        
        gpus = GPUMonitor.get_gpu_stats()
        
        for gpu in gpus:
            gpu_id = gpu['index']
            
            # Check memory
            mem_util = gpu['memory']['utilization_percent']
            if mem_util > 95:
                issues.append(f"GPU {gpu_id}: Memory critically high ({mem_util}%)")
            elif mem_util > 85:
                warnings.append(f"GPU {gpu_id}: Memory high ({mem_util}%)")
            
            # Check temperature
            temp = gpu.get('temperature_c')
            if temp:
                if temp > 85:
                    issues.append(f"GPU {gpu_id}: Temperature critically high ({temp}°C)")
                elif temp > 75:
                    warnings.append(f"GPU {gpu_id}: Temperature high ({temp}°C)")
            
            # Check power
            power = gpu.get('power', {})
            if power.get('usage_watts') and power.get('limit_watts'):
                power_percent = (power['usage_watts'] / power['limit_watts']) * 100
                if power_percent > 95:
                    warnings.append(f"GPU {gpu_id}: Power usage high ({power_percent:.0f}%)")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat()
        }


# ============= Celery Tasks =============

@shared_task(name='training.monitoring.report_gpu_stats')
def report_gpu_stats():
    """
    Periodic task to report GPU statistics.
    Stores metrics for monitoring/alerting.
    """
    try:
        gpu_stats = GPUMonitor.get_gpu_stats()
        system_stats = GPUMonitor.get_system_stats()
        health = GPUMonitor.check_gpu_health()
        
        # Store in database for historical tracking
        from training.models import GPUMetrics
        
        for gpu in gpu_stats:
            GPUMetrics.objects.create(
                gpu_index=gpu['index'],
                gpu_name=gpu['name'],
                memory_used_gb=gpu['memory']['used_gb'],
                memory_total_gb=gpu['memory']['total_gb'],
                utilization_percent=gpu['utilization_percent'],
                temperature_c=gpu.get('temperature_c'),
                power_usage_watts=gpu.get('power', {}).get('usage_watts'),
                process_count=gpu['processes']['count']
            )
        
        # Log warnings/issues
        if health['warnings']:
            for warning in health['warnings']:
                logger.warning(warning)
        
        if health['issues']:
            for issue in health['issues']:
                logger.error(issue)
            
            # Send alerts (implement your alert system)
            # send_slack_alert(health['issues'])
            # send_email_alert(health['issues'])
        
        return {
            'gpu_stats': gpu_stats,
            'system_stats': system_stats,
            'health': health
        }
        
    except Exception as e:
        logger.error(f"Failed to report GPU stats: {e}")
        return {'error': str(e)}


@shared_task(name='training.monitoring.cleanup_old_metrics')
def cleanup_old_metrics():
    """Clean up old GPU metrics (keep last 7 days)"""
    from training.models import GPUMetrics
    
    cutoff = datetime.now() - timedelta(days=7)
    deleted_count, _ = GPUMetrics.objects.filter(
        created_at__lt=cutoff
    ).delete()
    
    logger.info(f"Cleaned up {deleted_count} old GPU metrics")
    return {'deleted': deleted_count}


# ============= Model for storing metrics =============
"""
Add to training/models.py:

class GPUMetrics(models.Model):
    '''Store GPU metrics for monitoring'''
    gpu_index = models.IntegerField()
    gpu_name = models.CharField(max_length=255)
    memory_used_gb = models.FloatField()
    memory_total_gb = models.FloatField()
    utilization_percent = models.FloatField()
    temperature_c = models.FloatField(null=True, blank=True)
    power_usage_watts = models.FloatField(null=True, blank=True)
    process_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'gpu_metrics'
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['gpu_index', 'created_at']),
        ]
"""


# ============= API Endpoint for GPU Stats =============
"""
Add to your API:

@router.get("/monitoring/gpu")
async def get_gpu_stats():
    '''Get current GPU statistics'''
    return {
        'gpus': GPUMonitor.get_gpu_stats(),
        'system': GPUMonitor.get_system_stats(),
        'health': GPUMonitor.check_gpu_health()
    }
"""
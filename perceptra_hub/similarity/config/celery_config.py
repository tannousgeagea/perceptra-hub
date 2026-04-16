# celery_config.py

import os
from kombu import Queue
from celery.schedules import crontab


def route_task(name, args, kwargs, options, task=None, **kw):
    """Route tasks to queues based on task name prefix (e.g., 'alarm:task' -> 'alarm' queue)"""
    if ":" in name:
        queue, _ = name.split(":", 1)
        return {"queue": queue}
    return {"queue": "celery"}


# Environment variables
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = os.getenv('RABBITMQ_PORT', '5672')
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.getenv('RABBITMQ_PASS', 'guest')
CELERY_ENV = os.getenv('CELERY_CONFIG', 'development')

class BaseConfig:
    """Base Celery configuration"""
    
    # Broker & Backend
    CELERY_BROKER_URL = os.getenv(
        "CELERY_BROKER_URL",
        f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}//"
    )
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "rpc://")
    
    # Serialization
    accept_content = ['json', 'pickle']
    task_serializer = 'pickle'
    result_serializer = 'pickle'
    
    # Timezone
    timezone = 'UTC'
    CELERY_ENABLE_UTC = True
    
    # Task routing
    CELERY_TASK_ROUTES = (route_task,)
    
    # Queues - Add your custom queues here
    CELERY_TASK_QUEUES = (
        Queue("celery"),           # Default queue
        Queue("cleanup"),          # Cleanup tasks
        Queue("similarity"),       # Similarity 
        Queue("maintenance"),      # Maintenance tasks
    )
    
    # Performance
    CELERY_WORKER_PREFETCH_MULTIPLIER = 1
    CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000
    
    # Task results
    CELERY_RESULT_EXPIRES = 3600  # 1 hour
    result_persistent = False
    CELERY_TASK_TRACK_STARTED = True
    
    # Events
    CELERY_WORKER_SEND_TASK_EVENTS = False
    CELERY_TASK_SEND_SENT_EVENT = False
    
    # Time limits (seconds)
    CELERY_TASK_SOFT_TIME_LIMIT = 300   # 5 minutes
    CELERY_TASK_TIME_LIMIT = 600        # 10 minutes
    
    # Task execution
    CELERY_TASK_ACKS_LATE = True
    CELERY_TASK_REJECT_ON_WORKER_LOST = True
    
    # Beat schedule
    CELERY_BEAT_SCHEDULE = {}


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    CELERY_TASK_ALWAYS_EAGER = False  # Set True to run tasks synchronously in dev
    CELERY_TASK_EAGER_PROPAGATES = True


class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    # More conservative settings for production
    CELERY_WORKER_PREFETCH_MULTIPLIER = 4
    CELERY_WORKER_MAX_TASKS_PER_CHILD = 500
    CELERY_WORKER_SEND_TASK_EVENTS = True  # Enable for monitoring
    CELERY_TASK_SEND_SENT_EVENT = True


class TestConfig(BaseConfig):
    """Test environment configuration"""
    CELERY_TASK_ALWAYS_EAGER = True  # Run tasks synchronously in tests
    CELERY_TASK_EAGER_PROPAGATES = True
    CELERY_RESULT_BACKEND = 'cache+memory://'


def get_config():
    """Get configuration based on environment"""
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'test': TestConfig,
    }
    return config_map.get(CELERY_ENV, DevelopmentConfig)()


# Export settings
settings = get_config()


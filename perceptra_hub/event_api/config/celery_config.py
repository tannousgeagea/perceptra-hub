import os
import celery
from functools import lru_cache
from kombu import Queue

RABBITMQ_PORT = os.environ.get('RABBITMQ_PORT', "5672")
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', "localhost")

def route_task(name, args, kwargs, options, task=None, **kw):
    print(name)
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "celery"}

class BaseConfig:
    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", f"amqp://guest:guest@{RABBITMQ_HOST}:{RABBITMQ_PORT}//")
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", "rpc://")

    CELERY_TASK_QUEUES: list = (
        # default queue
        Queue("celery"),
        # custom queue
    )

    CELERY_TASK_ROUTES = (route_task,)
    ACCEPT_CONTENT = ['json', 'pickle']
    TASK_SERIALIZE = 'pickle'
    RESULT_SERIALIZE = 'pickle'
    TIMEZONE = 'UTC'
    ENABLE_UTC = True 

class DevelopmentConfig(BaseConfig):
    pass


@lru_cache()
def get_settings():
    config_cls_dict = {
        "development": DevelopmentConfig,
    }
    config_name = os.environ.get("CELERY_CONFIG", "development")
    config_cls = config_cls_dict[config_name]
    return config_cls()


settings = get_settings()

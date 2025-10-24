import django
django.setup()
from django.core.cache import cache


def track_progress(
        task_id:str, 
        percentage:int,
        status:str,
        message:str=None
        ):
    cache.set(
        task_id,
        {
            "percentage": percentage,
            "status": status,
            "message": message,
            "isComplete": False if percentage < 100 else True
        }
    )

def get_progress(task_id):
    return cache.get(task_id)
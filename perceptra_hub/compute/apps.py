from django.apps import AppConfig
from common_utils.cache.cache_service import initialize_cache

# Call once on startup
initialize_cache()

class ComputeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "compute"

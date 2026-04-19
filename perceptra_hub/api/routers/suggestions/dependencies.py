import os
from typing import Optional

from .inference_client import SegInferenceClient
from .schemas import ModelConfig
from .segmentation_service_mock import MockSegmentationService
from .services import SuggestionService


def get_segmentation_service():
    """Return real inference client if SEG_INFERENCE_URL is set, otherwise mock."""
    url = os.getenv("SEG_INFERENCE_URL", "").strip()
    if not url:
        return MockSegmentationService()
    api_key = os.getenv("SEG_INFERENCE_API_KEY", "")
    timeout = float(os.getenv("SEG_INFERENCE_TIMEOUT", "10"))
    return SegInferenceClient(base_url=url, timeout_s=timeout, api_key=api_key)


def get_suggestion_service() -> SuggestionService:
    return SuggestionService()


def get_model_config(config: Optional[ModelConfig] = None) -> ModelConfig:
    return config or ModelConfig()

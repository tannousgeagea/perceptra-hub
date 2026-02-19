from typing import Optional
from fastapi import Depends
from .segmentation_service import SegmentationService
from .segmentation_service_mock import MockSegmentationService
from .services import SuggestionService
from .schemas import ModelConfig

def get_segmentation_service() -> MockSegmentationService:
    """Singleton service instance."""
    return MockSegmentationService()

def get_suggestion_service() -> SuggestionService:
    return SuggestionService()

def get_model_config(config: Optional[ModelConfig] = None) -> ModelConfig:
    """Provide default or user-specified model config."""
    return config or ModelConfig()
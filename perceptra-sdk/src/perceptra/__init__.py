"""
Perceptra — Official Python SDK for the Perceptra Hub CV MLOps Platform.

Usage::

    import perceptra

    client = perceptra.Perceptra(api_key="ph_live_abc123...")

    # Create a project
    project = client.projects.create(name="My Project", project_type_id=1)

    # Upload images
    client.images.upload(file="./photo.jpg", project_id=project.project_id)

    # Train a model
    result = client.models.train(model_id, dataset_version_id)
"""
from perceptra._version import __version__
from perceptra._client import Perceptra
from perceptra._async_client import AsyncPerceptra
from perceptra._exceptions import (
    PerceptraError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
)

__all__ = [
    "__version__",
    "Perceptra",
    "AsyncPerceptra",
    # Exceptions
    "PerceptraError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
]

"""
Asynchronous Perceptra client.
"""
from __future__ import annotations

from typing import Any, Optional, Type, TypeVar

import httpx

from perceptra._base_client import _BaseClient, _parse_response
from perceptra._exceptions import APIConnectionError, APITimeoutError
from perceptra._transport._auth import APIKeyAuth
from perceptra._transport._retry import should_retry, async_sleep_for_retry
from perceptra._utils import strip_none

T = TypeVar("T")


class AsyncPerceptra(_BaseClient):
    """
    Asynchronous client for the Perceptra Hub API.

    Usage::

        import perceptra

        async with perceptra.AsyncPerceptra(api_key="ph_live_abc123...") as client:
            projects = await client.projects.list()
    """

    _http: httpx.AsyncClient

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._http = httpx.AsyncClient(
            auth=APIKeyAuth(self.api_key),
            headers=self._build_headers(),
            timeout=httpx.Timeout(self.timeout),
        )
        self._init_resources()

    def _init_resources(self) -> None:
        from perceptra.resources.projects import AsyncProjects
        from perceptra.resources.images import AsyncImages
        from perceptra.resources.annotations import AsyncAnnotations
        from perceptra.resources.models import AsyncModels
        from perceptra.resources.training import AsyncTraining
        from perceptra.resources.versions import AsyncVersions
        from perceptra.resources.organizations import AsyncOrganizations
        from perceptra.resources.jobs import AsyncJobs
        from perceptra.resources.classes import AsyncClasses
        from perceptra.resources.tags import AsyncTags
        from perceptra.resources.api_keys import AsyncAPIKeys
        from perceptra.resources.storage import AsyncStorage

        self.projects = AsyncProjects(self)
        self.images = AsyncImages(self)
        self.annotations = AsyncAnnotations(self)
        self.models = AsyncModels(self)
        self.training = AsyncTraining(self)
        self.versions = AsyncVersions(self)
        self.organizations = AsyncOrganizations(self)
        self.jobs = AsyncJobs(self)
        self.classes = AsyncClasses(self)
        self.tags = AsyncTags(self)
        self.api_keys = AsyncAPIKeys(self)
        self.storage = AsyncStorage(self)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        query: Optional[dict[str, Any]] = None,
        files: Any = None,
        data: Optional[dict[str, Any]] = None,
        cast_to: Type[T] = dict,  # type: ignore
        stream: bool = False,
    ) -> T:
        url = self._build_url(path)
        params = strip_none(query) if query else None

        kwargs: dict[str, Any] = {"method": method, "url": url, "params": params}
        if files is not None:
            kwargs["files"] = files
            if data:
                kwargs["data"] = strip_none(data)
        elif body is not None:
            kwargs["json"] = body

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.request(**kwargs)
            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    continue
                raise APITimeoutError(f"Request to {path} timed out") from e
            except httpx.ConnectError as e:
                raise APIConnectionError(f"Failed to connect to {url}", cause=e) from e

            if should_retry(response) and attempt < self.max_retries:
                await async_sleep_for_retry(attempt, response)
                continue

            return _parse_response(response, cast_to)

        raise APIConnectionError("Max retries exceeded")

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncPerceptra:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

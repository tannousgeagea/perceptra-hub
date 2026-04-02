"""
Synchronous Perceptra client.
"""
from __future__ import annotations

from typing import Any, Optional, Type, TypeVar

import httpx

from perceptra._base_client import _BaseClient, _parse_response
from perceptra._exceptions import APIConnectionError, APITimeoutError
from perceptra._transport._auth import APIKeyAuth
from perceptra._transport._retry import should_retry, sleep_for_retry
from perceptra._utils import strip_none

T = TypeVar("T")


class Perceptra(_BaseClient):
    """
    Synchronous client for the Perceptra Hub API.

    Usage::

        import perceptra

        client = perceptra.Perceptra(api_key="ph_live_abc123...")
        projects = client.projects.list()
    """

    _http: httpx.Client

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._http = httpx.Client(
            auth=APIKeyAuth(self.api_key),
            headers=self._build_headers(),
            timeout=httpx.Timeout(self.timeout),
        )
        self._init_resources()

    def _init_resources(self) -> None:
        from perceptra.resources.projects import Projects
        from perceptra.resources.images import Images
        from perceptra.resources.annotations import Annotations
        from perceptra.resources.models import Models
        from perceptra.resources.training import Training
        from perceptra.resources.versions import Versions
        from perceptra.resources.organizations import Organizations
        from perceptra.resources.jobs import Jobs
        from perceptra.resources.classes import Classes
        from perceptra.resources.tags import Tags
        from perceptra.resources.api_keys import APIKeys
        from perceptra.resources.storage import Storage

        self.projects = Projects(self)
        self.images = Images(self)
        self.annotations = Annotations(self)
        self.models = Models(self)
        self.training = Training(self)
        self.versions = Versions(self)
        self.organizations = Organizations(self)
        self.jobs = Jobs(self)
        self.classes = Classes(self)
        self.tags = Tags(self)
        self.api_keys = APIKeys(self)
        self.storage = Storage(self)

    def _request(
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

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.request(**kwargs)
            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    continue
                raise APITimeoutError(f"Request to {path} timed out") from e
            except httpx.ConnectError as e:
                raise APIConnectionError(f"Failed to connect to {url}", cause=e) from e

            if should_retry(response) and attempt < self.max_retries:
                sleep_for_retry(attempt, response)
                continue

            return _parse_response(response, cast_to)

        raise APIConnectionError("Max retries exceeded")  # Should not reach here

    def _stream(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = self._build_url(path)
        return self._http.stream(method, url, **kwargs)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Perceptra:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

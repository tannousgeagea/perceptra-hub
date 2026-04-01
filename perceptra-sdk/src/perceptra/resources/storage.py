from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Storage(SyncAPIResource):

    def create_profile(
        self,
        name: str,
        backend: str,
        config: dict,
        *,
        region: Optional[str] = None,
        is_default: Optional[bool] = None,
        credential_ref_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "backend": backend, "config": config}
        if region is not None: body["region"] = region
        if is_default is not None: body["is_default"] = is_default
        if credential_ref_id is not None: body["credential_ref_id"] = credential_ref_id
        return self._client._request("POST", "/storage/profiles", body=body, cast_to=dict)

    def list_profiles(self, *, skip: int = 0, limit: int = 100) -> list:
        return self._client._request(
            "GET", "/storage/profiles",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    def retrieve_profile(self, profile_id: str) -> dict:
        return self._client._request("GET", f"/storage/profiles/{profile_id}", cast_to=dict)

    def update_profile(
        self,
        profile_id: str,
        *,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        region: Optional[str] = None,
        is_default: Optional[bool] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if config is not None: body["config"] = config
        if region is not None: body["region"] = region
        if is_default is not None: body["is_default"] = is_default
        return self._client._request("PUT", f"/storage/profiles/{profile_id}", body=body, cast_to=dict)

    def delete_profile(self, profile_id: str) -> None:
        self._client._request("DELETE", f"/storage/profiles/{profile_id}", cast_to=dict)

    def test_connection(self, profile_id: str) -> dict:
        return self._client._request("POST", f"/storage/profiles/{profile_id}/test", cast_to=dict)


class AsyncStorage(AsyncAPIResource):

    async def create_profile(
        self,
        name: str,
        backend: str,
        config: dict,
        *,
        region: Optional[str] = None,
        is_default: Optional[bool] = None,
        credential_ref_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "backend": backend, "config": config}
        if region is not None: body["region"] = region
        if is_default is not None: body["is_default"] = is_default
        if credential_ref_id is not None: body["credential_ref_id"] = credential_ref_id
        return await self._client._request("POST", "/storage/profiles", body=body, cast_to=dict)

    async def list_profiles(self, *, skip: int = 0, limit: int = 100) -> list:
        return await self._client._request(
            "GET", "/storage/profiles",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    async def retrieve_profile(self, profile_id: str) -> dict:
        return await self._client._request("GET", f"/storage/profiles/{profile_id}", cast_to=dict)

    async def update_profile(
        self,
        profile_id: str,
        *,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        region: Optional[str] = None,
        is_default: Optional[bool] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if config is not None: body["config"] = config
        if region is not None: body["region"] = region
        if is_default is not None: body["is_default"] = is_default
        return await self._client._request("PUT", f"/storage/profiles/{profile_id}", body=body, cast_to=dict)

    async def delete_profile(self, profile_id: str) -> None:
        await self._client._request("DELETE", f"/storage/profiles/{profile_id}", cast_to=dict)

    async def test_connection(self, profile_id: str) -> dict:
        return await self._client._request("POST", f"/storage/profiles/{profile_id}/test", cast_to=dict)

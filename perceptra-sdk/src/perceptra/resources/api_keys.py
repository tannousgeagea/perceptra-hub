from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class APIKeys(SyncAPIResource):

    def create(
        self,
        name: str,
        *,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None,
        scope: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name}
        if permissions is not None: body["permissions"] = permissions
        if expires_in_days is not None: body["expires_in_days"] = expires_in_days
        if scopes is not None: body["scopes"] = scopes
        if allowed_ips is not None: body["allowed_ips"] = allowed_ips
        if rate_limit_per_minute is not None: body["rate_limit_per_minute"] = rate_limit_per_minute
        if rate_limit_per_hour is not None: body["rate_limit_per_hour"] = rate_limit_per_hour
        if scope is not None: body["scope"] = scope
        if user_id is not None: body["user_id"] = user_id
        return self._client._request("POST", "/api-keys", body=body, cast_to=dict)

    def list(self, *, is_active: Optional[bool] = None, scope: Optional[str] = None) -> list:
        query: dict[str, Any] = {}
        if is_active is not None: query["is_active"] = is_active
        if scope is not None: query["scope"] = scope
        return self._client._request("GET", "/api-keys", query=query or None, cast_to=list)

    def retrieve(self, key_id: str) -> dict:
        return self._client._request("GET", f"/api-keys/{key_id}", cast_to=dict)

    def update(
        self,
        key_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        scopes: Optional[List[str]] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if description is not None: body["description"] = description
        if permissions is not None: body["permissions"] = permissions
        if is_active is not None: body["is_active"] = is_active
        if scopes is not None: body["scopes"] = scopes
        if allowed_ips is not None: body["allowed_ips"] = allowed_ips
        if rate_limit_per_minute is not None: body["rate_limit_per_minute"] = rate_limit_per_minute
        if rate_limit_per_hour is not None: body["rate_limit_per_hour"] = rate_limit_per_hour
        return self._client._request("PATCH", f"/api-keys/{key_id}", body=body, cast_to=dict)

    def revoke(self, key_id: str) -> dict:
        return self._client._request("POST", f"/api-keys/{key_id}/revoke", cast_to=dict)

    def rotate(self, key_id: str, *, grace_period_hours: Optional[int] = None) -> dict:
        body: dict[str, Any] = {}
        if grace_period_hours is not None: body["grace_period_hours"] = grace_period_hours
        return self._client._request("POST", f"/api-keys/{key_id}/rotate", body=body or None, cast_to=dict)

    def delete(self, key_id: str) -> None:
        self._client._request("DELETE", f"/api-keys/{key_id}", cast_to=dict)

    def usage(self, key_id: str, *, days: Optional[int] = None) -> dict:
        query: dict[str, Any] = {}
        if days is not None: query["days"] = days
        return self._client._request("GET", f"/api-keys/{key_id}/usage", query=query or None, cast_to=dict)


class AsyncAPIKeys(AsyncAPIResource):

    async def create(
        self,
        name: str,
        *,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None,
        scope: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name}
        if permissions is not None: body["permissions"] = permissions
        if expires_in_days is not None: body["expires_in_days"] = expires_in_days
        if scopes is not None: body["scopes"] = scopes
        if allowed_ips is not None: body["allowed_ips"] = allowed_ips
        if rate_limit_per_minute is not None: body["rate_limit_per_minute"] = rate_limit_per_minute
        if rate_limit_per_hour is not None: body["rate_limit_per_hour"] = rate_limit_per_hour
        if scope is not None: body["scope"] = scope
        if user_id is not None: body["user_id"] = user_id
        return await self._client._request("POST", "/api-keys", body=body, cast_to=dict)

    async def list(self, *, is_active: Optional[bool] = None, scope: Optional[str] = None) -> list:
        query: dict[str, Any] = {}
        if is_active is not None: query["is_active"] = is_active
        if scope is not None: query["scope"] = scope
        return await self._client._request("GET", "/api-keys", query=query or None, cast_to=list)

    async def retrieve(self, key_id: str) -> dict:
        return await self._client._request("GET", f"/api-keys/{key_id}", cast_to=dict)

    async def update(
        self,
        key_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        scopes: Optional[List[str]] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if description is not None: body["description"] = description
        if permissions is not None: body["permissions"] = permissions
        if is_active is not None: body["is_active"] = is_active
        if scopes is not None: body["scopes"] = scopes
        if allowed_ips is not None: body["allowed_ips"] = allowed_ips
        if rate_limit_per_minute is not None: body["rate_limit_per_minute"] = rate_limit_per_minute
        if rate_limit_per_hour is not None: body["rate_limit_per_hour"] = rate_limit_per_hour
        return await self._client._request("PATCH", f"/api-keys/{key_id}", body=body, cast_to=dict)

    async def revoke(self, key_id: str) -> dict:
        return await self._client._request("POST", f"/api-keys/{key_id}/revoke", cast_to=dict)

    async def rotate(self, key_id: str, *, grace_period_hours: Optional[int] = None) -> dict:
        body: dict[str, Any] = {}
        if grace_period_hours is not None: body["grace_period_hours"] = grace_period_hours
        return await self._client._request("POST", f"/api-keys/{key_id}/rotate", body=body or None, cast_to=dict)

    async def delete(self, key_id: str) -> None:
        await self._client._request("DELETE", f"/api-keys/{key_id}", cast_to=dict)

    async def usage(self, key_id: str, *, days: Optional[int] = None) -> dict:
        query: dict[str, Any] = {}
        if days is not None: query["days"] = days
        return await self._client._request("GET", f"/api-keys/{key_id}/usage", query=query or None, cast_to=dict)

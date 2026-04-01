from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra.types.organization import OrganizationDetails

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Organizations(SyncAPIResource):

    def retrieve(self) -> OrganizationDetails:
        return self._client._request("GET", "/organizations/details", cast_to=OrganizationDetails)

    def list_members(self, *, skip: int = 0, limit: int = 100) -> list:
        return self._client._request(
            "GET", "/organizations/members",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    def list_projects(self, *, skip: int = 0, limit: int = 100) -> list:
        return self._client._request(
            "GET", "/organizations/projects",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )


class AsyncOrganizations(AsyncAPIResource):

    async def retrieve(self) -> OrganizationDetails:
        return await self._client._request("GET", "/organizations/details", cast_to=OrganizationDetails)

    async def list_members(self, *, skip: int = 0, limit: int = 100) -> list:
        return await self._client._request(
            "GET", "/organizations/members",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    async def list_projects(self, *, skip: int = 0, limit: int = 100) -> list:
        return await self._client._request(
            "GET", "/organizations/projects",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

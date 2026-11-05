"""
The signed-in user's notifications in the current organization.

GET    /notifications               newest first (by last update), paginated with `before`
GET    /notifications/unread-count
POST   /notifications/{id}/read
POST   /notifications/read-all
DELETE /notifications               clear them all

Live delivery: the "notification" event on the user's realtime channel.
"""
from datetime import datetime
from typing import List, Optional

from asgiref.sync import sync_to_async
from django.utils import timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.dependencies import RequestContext, get_request_context
from notifications.models import Notification
from notifications.services import serialize

router = APIRouter()


class NotificationOut(BaseModel):
    id: int
    kind: str
    title: str
    body: str
    url: str
    count: int
    read: bool
    organization_id: str
    project_id: Optional[str] = None
    actor: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class NotificationPage(BaseModel):
    items: List[NotificationOut]
    unread: int
    # Pass as `before` to get the next page; null when there is none.
    next_before: Optional[datetime] = None


class UnreadCount(BaseModel):
    unread: int


def _mine(ctx: RequestContext):
    if ctx.user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="A user is required")
    return Notification.objects.filter(recipient=ctx.user, organization=ctx.organization)


@router.get("/notifications", response_model=NotificationPage, summary="My notifications")
async def list_notifications(
    limit: int = Query(30, ge=1, le=100),
    before: Optional[datetime] = Query(None, description="Only notices last updated before this time"),
    unread_only: bool = Query(False),
    ctx: RequestContext = Depends(get_request_context),
):
    @sync_to_async
    def fetch():
        qs = _mine(ctx)
        unread = qs.filter(read_at__isnull=True).count()
        if unread_only:
            qs = qs.filter(read_at__isnull=True)
        if before is not None:
            qs = qs.filter(updated_at__lt=before)
        rows = list(
            qs.select_related('organization', 'project', 'actor').order_by('-updated_at', '-id')[: limit + 1]
        )
        more = len(rows) > limit
        rows = rows[:limit]
        return {
            'items': [serialize(n) for n in rows],
            'unread': unread,
            'next_before': rows[-1].updated_at if more and rows else None,
        }

    return await fetch()


@router.get("/notifications/unread-count", response_model=UnreadCount, summary="My unread notifications")
async def unread_count(ctx: RequestContext = Depends(get_request_context)):
    @sync_to_async
    def count():
        return _mine(ctx).filter(read_at__isnull=True).count()

    return {'unread': await count()}


@router.post("/notifications/read-all", response_model=UnreadCount, summary="Mark all as read")
async def read_all(ctx: RequestContext = Depends(get_request_context)):
    @sync_to_async
    def mark():
        _mine(ctx).filter(read_at__isnull=True).update(read_at=timezone.now())

    await mark()
    return {'unread': 0}


@router.post("/notifications/{notification_id}/read", response_model=UnreadCount, summary="Mark one as read")
async def read_one(notification_id: int, ctx: RequestContext = Depends(get_request_context)):
    @sync_to_async
    def mark():
        qs = _mine(ctx)
        if not qs.filter(pk=notification_id).exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
        qs.filter(pk=notification_id, read_at__isnull=True).update(read_at=timezone.now())
        return qs.filter(read_at__isnull=True).count()

    return {'unread': await mark()}


@router.delete("/notifications", response_model=UnreadCount, summary="Clear my notifications")
async def clear(ctx: RequestContext = Depends(get_request_context)):
    @sync_to_async
    def delete():
        _mine(ctx).delete()

    await delete()
    return {'unread': 0}

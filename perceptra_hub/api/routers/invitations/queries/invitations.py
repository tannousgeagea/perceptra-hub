"""
Organization invitations.

Admins (of the current organization):
    POST   /api/v1/organizations/invitations                 invite an email with a role
    GET    /api/v1/organizations/invitations?status=pending  list
    POST   /api/v1/organizations/invitations/{id}/resend     email it again (renews 7 days)
    DELETE /api/v1/organizations/invitations/{id}            withdraw

Invitee:
    GET    /api/v1/invitations/{token}          what the link is for (public: the token is the secret)
    POST   /api/v1/invitations/{token}/accept   join, signed in with the invited email

Signing up through an invitation goes through POST /auth/register with invite_token.
"""
from typing import List, Literal, Optional

from asgiref.sync import sync_to_async
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from api.dependencies import RequestContext, get_request_context
from memberships import invitations as svc
from memberships.models import OrganizationInvitation
from users.auth import get_current_user

router = APIRouter()

ERROR_STATUS = {
    'bad_role': status.HTTP_400_BAD_REQUEST,
    'already_member': status.HTTP_409_CONFLICT,
    'not_pending': status.HTTP_409_CONFLICT,
    'not_found': status.HTTP_404_NOT_FOUND,
    'expired': status.HTTP_410_GONE,
    'revoked': status.HTTP_410_GONE,
    'accepted': status.HTTP_409_CONFLICT,
    'wrong_account': status.HTTP_403_FORBIDDEN,
}


def _http(err: svc.InvitationError) -> HTTPException:
    return HTTPException(
        status_code=ERROR_STATUS.get(err.code, status.HTTP_400_BAD_REQUEST),
        detail={'code': err.code, 'message': err.message},
    )


class InviteRequest(BaseModel):
    email: EmailStr
    role: Literal['viewer', 'editor', 'admin'] = 'viewer'
    message: Optional[str] = Field(None, max_length=1000)


# ─── Admin side ──────────────────────────────────────────────────────────────

@router.post("/organizations/invitations", status_code=status.HTTP_201_CREATED)
async def create_invitation(
    data: InviteRequest,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def run():
        invitation = svc.invite(ctx.organization, data.email, data.role, ctx.user, data.message or '')
        invitation = OrganizationInvitation.objects.select_related('role', 'invited_by').get(pk=invitation.pk)
        return invitation.pk, svc.serialize_for_admin(invitation)

    try:
        invitation_id, body = await run()
    except svc.InvitationError as err:
        raise _http(err)
    background_tasks.add_task(svc.send_invitation, invitation_id)
    return body


@router.get("/organizations/invitations")
async def list_invitations(
    status_filter: Literal['pending', 'all'] = Query('pending', alias='status'),
    ctx: RequestContext = Depends(get_request_context),
) -> List[dict]:
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def run():
        qs = OrganizationInvitation.objects.filter(organization=ctx.organization).select_related('role', 'invited_by')
        if status_filter == 'pending':
            qs = qs.filter(status=OrganizationInvitation.Status.PENDING)
        return [svc.serialize_for_admin(i) for i in qs.order_by('-created_at')[:200]]

    return await run()


@sync_to_async
def _get_own(ctx: RequestContext, invitation_id: int) -> OrganizationInvitation:
    invitation = (
        OrganizationInvitation.objects.select_related('role', 'invited_by', 'organization')
        .filter(pk=invitation_id, organization=ctx.organization).first()
    )
    if invitation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Invitation not found')
    return invitation


@router.post("/organizations/invitations/{invitation_id}/resend")
async def resend_invitation(
    invitation_id: int,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role('admin', 'owner')
    invitation = await _get_own(ctx, invitation_id)
    try:
        await sync_to_async(svc.renew_for_resend)(invitation)
    except svc.InvitationError as err:
        raise _http(err)
    background_tasks.add_task(svc.send_invitation, invitation.pk)
    return await sync_to_async(svc.serialize_for_admin)(invitation)


@router.delete("/organizations/invitations/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invitation(
    invitation_id: int,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role('admin', 'owner')
    invitation = await _get_own(ctx, invitation_id)
    try:
        await sync_to_async(svc.revoke)(invitation)
    except svc.InvitationError as err:
        raise _http(err)


# ─── Invitee side ────────────────────────────────────────────────────────────

@router.get("/invitations/{token}")
async def get_invitation(token: str):
    """What the invitation is for. Public: holding the link is the authorization."""
    invitation = await sync_to_async(svc.get_by_token)(token)
    if invitation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={'code': 'not_found', 'message': 'This invitation link is not valid'},
        )
    return await sync_to_async(svc.preview)(invitation)


@router.post("/invitations/{token}/accept")
async def accept_invitation(token: str, user=Depends(get_current_user)):
    """Join the organization. The signed-in account must have the invited email."""

    @sync_to_async
    def run():
        invitation = svc.check_usable(svc.get_by_token(token), user.email)
        membership = svc.accept(invitation, user)
        organization = membership.organization
        return {
            'organization': {
                'id': str(organization.org_id),
                'name': organization.name,
                'slug': organization.slug,
                'role': membership.role.name,
            },
        }

    try:
        return await run()
    except svc.InvitationError as err:
        raise _http(err)

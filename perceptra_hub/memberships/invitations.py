"""
Inviting people into an organization by email.

    invite()            an admin invites an email with a role (and a note)
    send_invitation()   the email with the link (after the response)
    preview()           what the link shows: who invites you, to what, as what
    accept()            a signed-in user with that email joins the organization

The link carries a random token; it is the only credential, valid for
INVITATION_DAYS and once. Accepting needs an account with the invited email,
so a forwarded link is useless to anyone else.
"""
import logging
import os
from datetime import timedelta
from typing import Optional
from urllib.parse import quote

from django.db import transaction
from django.utils import timezone

from memberships.models import OrganizationInvitation, OrganizationMembership, Role

logger = logging.getLogger(__name__)

INVITATION_DAYS = 7
INVITABLE_ROLES = ('viewer', 'editor', 'admin')

# Shown on the invitation page and in the email: what joining means.
ROLE_INFO = {
    'viewer': {
        'label': 'Viewer',
        'summary': 'See projects, datasets and results',
        'abilities': [
            "Browse the organization's projects and images",
            'View datasets, models and analytics',
        ],
    },
    'editor': {
        'label': 'Editor',
        'summary': 'Annotate, review and build datasets',
        'abilities': [
            'Annotate and review images in projects',
            'Create dataset versions and train models',
            'Everything a viewer can do',
        ],
    },
    'admin': {
        'label': 'Admin',
        'summary': 'Run the organization',
        'abilities': [
            'Invite and manage members',
            'Manage projects, storage and compute',
            'Everything an editor can do',
        ],
    },
}


class InvitationError(Exception):
    """A user-facing reason the invitation can't be used (code + message)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def frontend_url() -> str:
    return (os.getenv('FRONTEND_ENDPOINT') or 'http://localhost:5173').rstrip('/')


def invitation_url(invitation: OrganizationInvitation) -> str:
    return f"{frontend_url()}/invite/{quote(invitation.token)}"


def display_name(user) -> str:
    if user is None:
        return 'Someone'
    full = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return full or user.username or user.email


def org_name(organization) -> str:
    # The name the console shows everywhere (sidebar, switcher, invite dialog).
    return organization.name


def state_of(invitation: OrganizationInvitation) -> str:
    """pending | expired | accepted | revoked"""
    if invitation.status == OrganizationInvitation.Status.PENDING and invitation.is_expired:
        return 'expired'
    return invitation.status


# ─── Admin side ──────────────────────────────────────────────────────────────

def invite(organization, email: str, role_name: str, invited_by, message: str = '') -> OrganizationInvitation:
    """
    Create (or renew) the pending invitation for this email. Raises
    InvitationError if the role can't be granted or they are already a member.
    """
    from django.contrib.auth import get_user_model

    email = email.strip().lower()
    role_name = role_name.strip().lower()
    if role_name not in INVITABLE_ROLES:
        raise InvitationError('bad_role', f"Invitations can grant: {', '.join(INVITABLE_ROLES)}")
    role = Role.objects.filter(name=role_name).first()
    if role is None:
        raise InvitationError('bad_role', f"Role '{role_name}' does not exist")

    existing_user = get_user_model().objects.filter(email__iexact=email).first()
    if existing_user and OrganizationMembership.objects.filter(user=existing_user, organization=organization).exists():
        raise InvitationError('already_member', f"{email} is already a member of {org_name(organization)}")

    with transaction.atomic():
        invitation = (
            OrganizationInvitation.objects.select_for_update()
            .filter(organization=organization, email__iexact=email, status=OrganizationInvitation.Status.PENDING)
            .first()
        )
        expires = timezone.now() + timedelta(days=INVITATION_DAYS)
        if invitation is None:
            invitation = OrganizationInvitation.objects.create(
                organization=organization, email=email, role=role, invited_by=invited_by,
                message=(message or '').strip()[:1000], expires_at=expires,
            )
        else:
            # Inviting again updates the role/note and restarts the clock; same link.
            invitation.role = role
            invitation.invited_by = invited_by
            invitation.message = (message or '').strip()[:1000]
            invitation.expires_at = expires
            invitation.save(update_fields=['role', 'invited_by', 'message', 'expires_at'])
    return invitation


def revoke(invitation: OrganizationInvitation) -> None:
    if invitation.status != OrganizationInvitation.Status.PENDING:
        raise InvitationError('not_pending', 'Only pending invitations can be revoked')
    invitation.status = OrganizationInvitation.Status.REVOKED
    invitation.save(update_fields=['status'])


def renew_for_resend(invitation: OrganizationInvitation) -> None:
    if invitation.status != OrganizationInvitation.Status.PENDING:
        raise InvitationError('not_pending', 'Only pending invitations can be resent')
    invitation.expires_at = timezone.now() + timedelta(days=INVITATION_DAYS)
    invitation.save(update_fields=['expires_at'])


def send_invitation(invitation_id: int) -> None:
    """Email the invitation. Runs after the response; never raises, never logs the link."""
    from common_utils.emails.utils import send_invitation_email

    try:
        invitation = OrganizationInvitation.objects.select_related(
            'organization', 'role', 'invited_by'
        ).get(pk=invitation_id)
        if invitation.status != OrganizationInvitation.Status.PENDING:
            return
        send_invitation_email(invitation, invitation_url(invitation), email_context(invitation))
        invitation.last_sent_at = timezone.now()
        invitation.send_count += 1
        invitation.save(update_fields=['last_sent_at', 'send_count'])
        logger.info('Invitation %s emailed', invitation_id)
    except Exception as exc:
        logger.error('Invitation %s email failed: %s', invitation_id, type(exc).__name__)


def serialize_for_admin(invitation: OrganizationInvitation) -> dict:
    return {
        'id': invitation.id,
        'email': invitation.email,
        'role': invitation.role.name,
        'status': state_of(invitation),
        'message': invitation.message,
        'invited_by': display_name(invitation.invited_by) if invitation.invited_by_id else None,
        'created_at': invitation.created_at.isoformat(),
        'expires_at': invitation.expires_at.isoformat(),
        'last_sent_at': invitation.last_sent_at.isoformat() if invitation.last_sent_at else None,
        'send_count': invitation.send_count,
        'link': invitation_url(invitation),
    }


# ─── Invitee side ────────────────────────────────────────────────────────────

def _org_facts(organization) -> dict:
    from projects.models import Project

    return {
        'members': OrganizationMembership.objects.filter(organization=organization, status='active').count(),
        'projects': Project.objects.filter(organization=organization, is_active=True, is_deleted=False).count(),
    }


def email_context(invitation: OrganizationInvitation) -> dict:
    organization = invitation.organization
    role = ROLE_INFO.get(invitation.role.name, {'label': invitation.role.name.title(), 'summary': '', 'abilities': []})
    return {
        'org_name': org_name(organization),
        'org_initials': ''.join(w[0] for w in org_name(organization).split()[:2]).upper() or 'O',
        'org_description': organization.description or '',
        'inviter_name': display_name(invitation.invited_by),
        'inviter_email': invitation.invited_by.email if invitation.invited_by_id else '',
        'role_label': role['label'],
        'role_summary': role['summary'],
        'role_abilities': role['abilities'],
        'message': invitation.message,
        'email': invitation.email,
        'expires_at': invitation.expires_at,
        'valid_days': INVITATION_DAYS,
        **_org_facts(organization),
    }


def get_by_token(token: str) -> Optional[OrganizationInvitation]:
    if not token or len(token) > 64:
        return None
    return (
        OrganizationInvitation.objects.select_related('organization', 'role', 'invited_by')
        .filter(token=token).first()
    )


def preview(invitation: OrganizationInvitation) -> dict:
    """What the invitation page shows. Only for whoever holds the link."""
    from django.contrib.auth import get_user_model

    ctx = email_context(invitation)
    organization = invitation.organization
    return {
        'state': state_of(invitation),
        'email': invitation.email,
        'account_exists': get_user_model().objects.filter(email__iexact=invitation.email, is_active=True).exists(),
        'organization': {
            'id': str(organization.org_id),
            'name': ctx['org_name'],
            'description': ctx['org_description'],
            'logo_url': organization.logo_url or None,
            'members': ctx['members'],
            'projects': ctx['projects'],
        },
        'role': {
            'name': invitation.role.name,
            'label': ctx['role_label'],
            'summary': ctx['role_summary'],
            'abilities': ctx['role_abilities'],
        },
        'invited_by': {'name': ctx['inviter_name'], 'email': ctx['inviter_email']} if invitation.invited_by_id else None,
        'message': invitation.message,
        'created_at': invitation.created_at.isoformat(),
        'expires_at': invitation.expires_at.isoformat(),
    }


def check_usable(invitation: Optional[OrganizationInvitation], email: str) -> OrganizationInvitation:
    """Raise InvitationError unless this email can use the invitation now."""
    if invitation is None:
        raise InvitationError('not_found', 'This invitation link is not valid')
    state = state_of(invitation)
    if state == 'expired':
        raise InvitationError('expired', 'This invitation has expired. Ask for a new one.')
    if state == 'revoked':
        raise InvitationError('revoked', 'This invitation was withdrawn')
    if state == 'accepted':
        raise InvitationError('accepted', 'This invitation has already been used')
    if (email or '').strip().lower() != invitation.email.lower():
        raise InvitationError(
            'wrong_account',
            f"This invitation is for {invitation.email}. Sign in with that email to accept it.",
        )
    return invitation


def accept(invitation: OrganizationInvitation, user) -> OrganizationMembership:
    """Join the organization (in the caller's transaction or its own)."""
    with transaction.atomic():
        invitation = OrganizationInvitation.objects.select_for_update().select_related(
            'organization', 'role'
        ).get(pk=invitation.pk)
        check_usable(invitation, user.email)
        membership, created = OrganizationMembership.objects.get_or_create(
            user=user,
            organization=invitation.organization,
            defaults={'role': invitation.role, 'invited_by': invitation.invited_by, 'status': 'active'},
        )
        invitation.status = OrganizationInvitation.Status.ACCEPTED
        invitation.accepted_at = timezone.now()
        invitation.accepted_by = user
        invitation.save(update_fields=['status', 'accepted_at', 'accepted_by'])
    logger.info('Invitation %s accepted by user %s (new membership: %s)', invitation.pk, user.id, created)
    return membership

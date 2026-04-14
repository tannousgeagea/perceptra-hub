# apps/billing/tasks.py
from celery import shared_task
from django.db import transaction
from activity.models import ActivityEvent, ActivityEventType
from billing.models import BillableAction, BillingRateCard
from memberships.models import ProjectMembership, OrganizationMembership
from annotations.models import Annotation
import logging

logger = logging.getLogger(__name__)


def get_rate_card_for_event(event):
    """
    Get appropriate rate card with proper hierarchy.
    
    Priority:
    1. Project membership rate card (user-specific for this project)
    2. Organization membership rate card (user-specific for this org)
    3. Project-specific rate card (all users in this project)
    4. Organization default rate card (fallback)
    """
    
    # Priority 1: Project membership rate card
    if event.user and event.project:
        try:
            project_membership = ProjectMembership.objects.get(
                user=event.user,
                project=event.project
            )
            if project_membership.billing_rate_card:
                return project_membership.billing_rate_card
        except ProjectMembership.DoesNotExist:
            pass
    
    # Priority 2: Organization membership rate card
    if event.user:
        try:
            org_membership = OrganizationMembership.objects.get(
                user=event.user,
                organization=event.organization,
                status='active'
            )
            if org_membership.billing_rate_card:
                return org_membership.billing_rate_card
        except OrganizationMembership.DoesNotExist:
            pass
    
    # Priority 3: Project-specific rate card
    if event.project:
        rate_card = BillingRateCard.objects.filter(
            organization=event.organization,
            project=event.project,
            is_active=True
        ).first()
        if rate_card:
            return rate_card
    
    # Priority 4: Organization default rate card
    return BillingRateCard.objects.filter(
        organization=event.organization,
        project__isnull=True,
        is_active=True
    ).first()

@shared_task(
    bind=True, 
    max_retries=3, 
    queue="activity", 
    name="activity:create_billable_action_from_event"
)
def create_billable_action_from_event(self, event_id):
    """
    Convert ActivityEvent into BillableAction(s) for billing.
    """
    try:
        event = ActivityEvent.objects.get(event_id=event_id)
        
        # Get applicable rate card (project-specific or default)
        rate_card = get_rate_card_for_event(event)
        
        if not rate_card:
            rate_card = BillingRateCard.objects.filter(
                organization=event.organization,
                project__isnull=True,
                is_active=True
            ).first()
        
        if not rate_card:
            logger.warning(f"No rate card found for org {event.organization_id}")
            return
        
        # Determine action type and rate
        action_type, unit_rate = determine_billable_action(event, rate_card)
        
        if not action_type:
            return  # Not billable
        
        # Create billable action
        with transaction.atomic():
            billable = BillableAction.objects.create(
                organization=event.organization,
                project=event.project,
                user=event.user,
                activity_event=event,
                rate_card=rate_card,
                action_type=action_type,
                quantity=1,
                unit_rate=unit_rate,
                currency=rate_card.currency,
                metadata=event.metadata
            )
            
            logger.info(f"Created billable action {billable.action_id} for event {event_id}")
        
    except Exception as exc:
        logger.error(f"Failed to create billable action: {exc}")
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


def determine_billable_action(event, rate_card):
    """
    Map ActivityEvent to BillableAction type and rate.
    
    Returns: (action_type, unit_rate) or (None, None) if not billable
    """
    event_type = event.event_type
    metadata = event.metadata
    
    # Annotation Creation
    if event_type == ActivityEventType.ANNOTATION_CREATE:
        annotation_source = metadata.get('annotation_source')
        
        if annotation_source == 'manual':
            # Check if this is a missed object (added where no prediction existed)
            # This requires additional context - for now, treat as new annotation
            return (BillableAction.ActionType.NEW_ANNOTATION, rate_card.rate_new_annotation)
        
        elif annotation_source == 'prediction':
            # AI prediction - check if untouched or edited later
            # This is determined on ANNOTATION_UPDATE events
            return (BillableAction.ActionType.UNTOUCHED_PREDICTION, rate_card.rate_untouched_prediction)
    
    # Annotation Update (Edit)
    elif event_type == ActivityEventType.ANNOTATION_UPDATE:
        edit_type = metadata.get('edit_type')
        
        if edit_type == 'minor':
            return (BillableAction.ActionType.MINOR_EDIT, rate_card.rate_minor_edit)
        
        elif edit_type == 'major':
            return (BillableAction.ActionType.MAJOR_EDIT, rate_card.rate_major_edit)
        
        elif edit_type == 'class_change':
            return (BillableAction.ActionType.CLASS_CHANGE, rate_card.rate_class_change)
    
    # Annotation Deletion
    elif event_type == ActivityEventType.ANNOTATION_DELETE:
        return (BillableAction.ActionType.DELETION, rate_card.rate_deletion)
    
    # Prediction Edits (specific)
    elif event_type == ActivityEventType.PREDICTION_EDIT:
        edit_type = metadata.get('edit_type')
        
        if edit_type == Annotation.EditType.MINOR:
            return (BillableAction.ActionType.MINOR_EDIT, rate_card.rate_minor_edit)
        elif edit_type == Annotation.EditType.MAJOR:
            return (BillableAction.ActionType.MAJOR_EDIT, rate_card.rate_major_edit)
        elif edit_type == Annotation.EditType.CLASS_CHANGE:
            return (BillableAction.ActionType.CLASS_CHANGE, rate_card.rate_class_change)
    
    # Reviews
    elif event_type == ActivityEventType.IMAGE_REVIEW:
        return (BillableAction.ActionType.IMAGE_REVIEW, rate_card.rate_image_review)
    
    elif event_type == ActivityEventType.ANNOTATION_REVIEW:
        return (BillableAction.ActionType.ANNOTATION_REVIEW, rate_card.rate_annotation_review)
    
    # Not billable
    return (None, None)

@shared_task(bind=True)
def backfill_billing_for_user_in_org(self, user_id, org_id, start_date=None, end_date=None):
    """
    Backfill billable actions for a user in a specific organization.
    
    **Use Case**: User just marked as external in this org, backfill their work.
    """
    from django.contrib.auth import get_user_model
    from activity.models import ActivityEvent
    from memberships.models import OrganizationMembership
    from organizations.models import Organization
    from django.utils import timezone
    
    User = get_user_model()
    
    try:
        user = User.objects.get(id=user_id)
        org = Organization.objects.get(id=org_id)
        membership = OrganizationMembership.objects.get(
            user=user,
            organization=org
        )
    except (User.DoesNotExist, Organization.DoesNotExist, OrganizationMembership.DoesNotExist) as e:
        logger.error(f"Entity not found: {e}")
        return
    
    if not membership.billing_enabled:
        logger.warning(f"Billing not enabled for {user.username} in {org.name}")
        return
    
    # Get date range
    if not start_date:
        first_event = ActivityEvent.objects.filter(
            user=user,
            organization=org
        ).order_by('timestamp').first()
        start_date = first_event.timestamp if first_event else timezone.now()
    
    if not end_date:
        end_date = timezone.now()
    
    # Find unbilled events
    unbilled_events = ActivityEvent.objects.filter(
        user=user,
        organization=org,
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).exclude(
        billable_actions__isnull=False
    ).order_by('timestamp')
    
    total_events = unbilled_events.count()
    logger.info(f"Backfilling {total_events} events for {user.username} in {org.name}")
    
    created_count = 0
    skipped_count = 0
    
    for event in unbilled_events.iterator(chunk_size=100):
        try:
            # Get rate card using hierarchy
            rate_card = get_rate_card_for_event(event)
            
            if not rate_card:
                logger.warning(f"No rate card for event {event.event_id}")
                skipped_count += 1
                continue
            
            action_type, unit_rate = determine_billable_action(event, rate_card)
            
            if not action_type:
                skipped_count += 1
                continue
            
            BillableAction.objects.create(
                organization=org,
                project=event.project,
                user=user,
                activity_event=event,
                rate_card=rate_card,
                action_type=action_type,
                quantity=1,
                unit_rate=unit_rate,
                currency=rate_card.currency,
                metadata=event.metadata
            )
            
            created_count += 1
            
        except Exception as e:
            logger.error(f"Failed to create billing for event {event.event_id}: {e}")
            skipped_count += 1
    
    logger.info(
        f"Backfill complete for {user.username} in {org.name}: "
        f"{created_count} created, {skipped_count} skipped"
    )
    
    return {
        'user_id': user_id,
        'username': user.username,
        'organization_id': org_id,
        'organization_name': org.name,
        'total_events': total_events,
        'created': created_count,
        'skipped': skipped_count
    }


@shared_task(bind=True)
def backfill_billing_for_user_in_project(self, user_id, project_id, start_date=None, end_date=None):
    """
    Backfill billable actions for a user in a specific project.
    
    **Use Case**: User marked as external for this project only.
    """
    from django.contrib.auth import get_user_model
    from projects.models import Project
    from memberships.models import ProjectMembership
    from activity.models import ActivityEvent
    from django.utils import timezone
    
    User = get_user_model()
    
    try:
        user = User.objects.get(id=user_id)
        project = Project.objects.get(id=project_id)
        membership = ProjectMembership.objects.get(
            user=user,
            project=project
        )
    except (User.DoesNotExist, Project.DoesNotExist, ProjectMembership.DoesNotExist) as e:
        logger.error(f"Entity not found: {e}")
        return
    
    if not membership.billing_enabled:
        logger.warning(f"Billing not enabled for {user.username} in project {project.name}")
        return
    
    # Get date range
    if not start_date:
        first_event = ActivityEvent.objects.filter(
            user=user,
            project=project
        ).order_by('timestamp').first()
        start_date = first_event.timestamp if first_event else timezone.now()
    
    if not end_date:
        end_date = timezone.now()
    
    # Find unbilled events in this project
    unbilled_events = ActivityEvent.objects.filter(
        user=user,
        project=project,
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).exclude(
        billable_actions__isnull=False
    ).order_by('timestamp')
    
    total_events = unbilled_events.count()
    logger.info(f"Backfilling {total_events} events for {user.username} in project {project.name}")
    
    created_count = 0
    skipped_count = 0
    
    for event in unbilled_events.iterator(chunk_size=100):
        try:
            rate_card = get_rate_card_for_event(event)
            
            if not rate_card:
                skipped_count += 1
                continue
            
            action_type, unit_rate = determine_billable_action(event, rate_card)
            
            if not action_type:
                skipped_count += 1
                continue
            
            BillableAction.objects.create(
                organization=project.organization,
                project=project,
                user=user,
                activity_event=event,
                rate_card=rate_card,
                action_type=action_type,
                quantity=1,
                unit_rate=unit_rate,
                currency=rate_card.currency,
                metadata=event.metadata
            )
            
            created_count += 1
            
        except Exception as e:
            logger.error(f"Failed to create billing for event {event.event_id}: {e}")
            skipped_count += 1
    
    logger.info(
        f"Backfill complete for {user.username} in {project.name}: "
        f"{created_count} created, {skipped_count} skipped"
    )
    
    return {
        'user_id': user_id,
        'username': user.username,
        'project_id': project_id,
        'project_name': project.name,
        'total_events': total_events,
        'created': created_count,
        'skipped': skipped_count
    }


@shared_task(bind=True)
def backfill_billing_for_organization(self, org_id, start_date=None, end_date=None):
    """
    Backfill billable actions for entire organization.
    
    **Use Case**: Organization was marked as vendor retroactively.
    """
    from organizations.models import Organization
    from activity.models import ActivityEvent
    from billing.models import BillableAction
    from django.utils import timezone
    
    try:
        org = Organization.objects.get(id=org_id)
    except Organization.DoesNotExist:
        logger.error(f"Organization {org_id} not found")
        return
    
    if not org.is_vendor:
        logger.warning(f"Organization {org.name} is not marked as vendor")
        return
    
    # Get date range
    if not start_date:
        first_event = ActivityEvent.objects.filter(organization=org).order_by('timestamp').first()
        start_date = first_event.timestamp if first_event else timezone.now()
    
    if not end_date:
        end_date = timezone.now()
    
    # Find all unbilled events
    unbilled_events = ActivityEvent.objects.filter(
        organization=org,
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).exclude(
        billable_actions__isnull=False
    )
    
    total_events = unbilled_events.count()
    logger.info(f"Backfilling {total_events} events for organization {org.name}")
    
    created_count = 0
    skipped_count = 0
    
    # Get org's default rate card
    default_rate_card = BillingRateCard.objects.filter(
        organization=org,
        project__isnull=True,
        is_active=True
    ).first()
    
    if not default_rate_card:
        logger.error(f"No default rate card found for {org.name}")
        return
    
    for event in unbilled_events.iterator(chunk_size=100):
        try:
            action_type, unit_rate = determine_billable_action(event, default_rate_card)
            
            if not action_type:
                skipped_count += 1
                continue
            
            # Check for user-specific or project-specific rate card
            rate_card = default_rate_card
            if event.user and event.user.billing_rate_card:
                rate_card = event.user.billing_rate_card
            elif event.project:
                project_rate = BillingRateCard.objects.filter(
                    organization=org,
                    project=event.project,
                    is_active=True
                ).first()
                if project_rate:
                    rate_card = project_rate
            
            BillableAction.objects.create(
                organization=org,
                project=event.project,
                user=event.user,
                activity_event=event,
                rate_card=rate_card,
                action_type=action_type,
                quantity=1,
                unit_rate=unit_rate,
                currency=rate_card.currency,
                metadata=event.metadata
            )
            
            created_count += 1
            
        except Exception as e:
            logger.error(f"Failed to create billing for event {event.event_id}: {e}")
            skipped_count += 1
    
    logger.info(
        f"Backfill complete for {org.name}: "
        f"{created_count} created, {skipped_count} skipped"
    )
    
    return {
        'organization_id': org_id,
        'organization_name': org.name,
        'total_events': total_events,
        'created': created_count,
        'skipped': skipped_count
    }
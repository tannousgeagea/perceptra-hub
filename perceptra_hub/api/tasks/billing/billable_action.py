# apps/billing/tasks.py
from celery import shared_task
from django.db import transaction
from activity.models import ActivityEvent, ActivityEventType
from billing.models import BillableAction, BillingRateCard
from annotations.models import Annotation
import logging

logger = logging.getLogger(__name__)


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
        rate_card = BillingRateCard.objects.filter(
            organization=event.organization,
            project=event.project,
            is_active=True
        ).first()
        
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
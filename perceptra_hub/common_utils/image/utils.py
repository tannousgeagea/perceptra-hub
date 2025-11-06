"""
Enhanced image search with query-based filtering.

Add this to images.py to replace the existing list_images endpoint.
"""

from typing import Optional, List
from uuid import UUID
from fastapi import Query
from django.db import models


def parse_search_query(query: str) -> dict:
    """
    Parse search query string into filters.
    
    Examples:
        "tag:car min-width:1920 class:vehicle"
        "split:train filename:IMG max-height:1080"
        "min-annotations:5 sort:size"
    
    Returns:
        Dictionary with parsed filters
    """
    filters = {
        'tags': [],
        'classes': [],
        'splits': [],
        'filename': None,
        'min_width': None,
        'max_width': None,
        'min_height': None,
        'max_height': None,
        'min_annotations': None,
        'max_annotations': None,
        'job': None,
        'like_image': None,
        'sort': None,
        'text_search': []  # Free text not matching any filter
    }
    
    if not query:
        return filters
    
    # Split by spaces but respect quotes
    import re
    tokens = re.findall(r'(?:[^\s"]|"(?:\\.|[^"])*")+', query)
    
    for token in tokens:
        token = token.strip('"')
        
        if ':' in token:
            key, value = token.split(':', 1)
            key = key.lower()
            
            if key == 'tag':
                filters['tags'].append(value)
            elif key == 'class':
                filters['classes'].append(value)
            elif key == 'split':
                filters['splits'].append(value)
            elif key == 'filename':
                filters['filename'] = value
            elif key == 'min-width':
                filters['min_width'] = int(value)
            elif key == 'max-width':
                filters['max_width'] = int(value)
            elif key == 'min-height':
                filters['min_height'] = int(value)
            elif key == 'max-height':
                filters['max_height'] = int(value)
            elif key == 'min-annotations':
                filters['min_annotations'] = int(value)
            elif key == 'max-annotations':
                filters['max_annotations'] = int(value)
            elif key == 'job':
                filters['job'] = value
            elif key == 'like-image':
                filters['like_image'] = value
            elif key == 'sort':
                filters['sort'] = value
            else:
                # Unknown filter, treat as text search
                filters['text_search'].append(token)
        else:
            # No colon, treat as text search
            filters['text_search'].append(token)
    
    return filters


def apply_image_filters(queryset, filters: dict):
    """
    Apply parsed filters to Image queryset.
    
    Args:
        queryset: Django queryset
        filters: Parsed filters from parse_search_query
    
    Returns:
        Filtered queryset
    """
    # Tag filters
    for tag in filters['tags']:
        queryset = queryset.filter(tags__name__iexact=tag)
    
    # Class filters (assuming you have annotations with classes)
    if filters['classes']:
        queryset = queryset.filter(
            annotations__annotation_class__name__in=filters['classes']
        ).distinct()
    
    # Split filters (assuming ProjectImage has split field)
    if filters['splits']:
        queryset = queryset.filter(
            project_images__split__in=filters['splits']
        ).distinct()
    
    # Filename filter
    if filters['filename']:
        queryset = queryset.filter(
            models.Q(name__icontains=filters['filename']) |
            models.Q(original_filename__icontains=filters['filename'])
        )
    
    # Dimension filters
    if filters['min_width']:
        queryset = queryset.filter(width__gte=filters['min_width'])
    if filters['max_width']:
        queryset = queryset.filter(width__lte=filters['max_width'])
    if filters['min_height']:
        queryset = queryset.filter(height__gte=filters['min_height'])
    if filters['max_height']:
        queryset = queryset.filter(height__lte=filters['max_height'])
    
    # Annotation count filters
    if filters['min_annotations']:
        queryset = queryset.annotate(
            annotation_count=models.Count('annotations')
        ).filter(annotation_count__gte=filters['min_annotations'])
    
    if filters['max_annotations']:
        queryset = queryset.annotate(
            annotation_count=models.Count('annotations')
        ).filter(annotation_count__lte=filters['max_annotations'])
    
    # Job filter
    if filters['job']:
        queryset = queryset.filter(annotations__job__name=filters['job']).distinct()
    
    # Text search (free text)
    if filters['text_search']:
        search_query = ' '.join(filters['text_search'])
        queryset = queryset.filter(
            models.Q(name__icontains=search_query) |
            models.Q(original_filename__icontains=search_query) |
            models.Q(source_of_origin__icontains=search_query)
        )
    
    # Sorting
    sort_mapping = {
        'size': '-file_size',
        'name': 'name',
        'date': '-created_at',
        'width': '-width',
        'height': '-height',
        'annotations': '-annotation_count'
    }
    
    if filters['sort']:
        sort_field = sort_mapping.get(filters['sort'], '-created_at')
        
        # Add annotation count if sorting by it
        if sort_field == '-annotation_count':
            queryset = queryset.annotate(
                annotation_count=models.Count('annotations')
            )
        
        queryset = queryset.order_by(sort_field)
    else:
        # Default sort by creation date
        queryset = queryset.order_by('-created_at')
    
    return queryset

def parse_project_image_query(query: str) -> dict:
    """Parse project image search query."""
    import re
    
    filters = {
        'status': None,
        'split': None,
        'annotated': None,
        'reviewed': None,
        'marked_null': None,
        'job_status': None,
        'tags': [],
        'classes': [],
        'filename': None,
        'min_width': None,
        'max_width': None,
        'min_height': None,
        'max_height': None,
        'min_annotations': None,
        'max_annotations': None,
        'sort': None,
        'text_search': []
    }
    
    if not query:
        return filters
    
    tokens = re.findall(r'(?:[^\s"]|"(?:\\.|[^"])*")+', query)
    
    for token in tokens:
        token = token.strip('"')
        
        if ':' in token:
            key, value = token.split(':', 1)
            key = key.lower()
            
            if key == 'status':
                filters['status'] = value
            elif key == 'annotated':
                filters['annotated'] = value.lower() == 'true'
            elif key == 'reviewed':
                filters['reviewed'] = value.lower() == 'true'
            elif key == 'marked-null':
                filters['marked_null'] = value.lower() == 'true'
            elif key == 'job-status':
                filters['job_status'] = value
            elif key == 'split':
                filters['split'] = value
            elif key == 'tag':
                filters['tags'].append(value)
            elif key == 'class':
                filters['classes'].append(value)
            elif key == 'filename':
                filters['filename'] = value
            elif key == 'min-width':
                filters['min_width'] = int(value)
            elif key == 'max-width':
                filters['max_width'] = int(value)
            elif key == 'min-height':
                filters['min_height'] = int(value)
            elif key == 'max-height':
                filters['max_height'] = int(value)
            elif key == 'min-annotations':
                filters['min_annotations'] = int(value)
            elif key == 'max-annotations':
                filters['max_annotations'] = int(value)
            elif key == 'sort':
                filters['sort'] = value
            else:
                filters['text_search'].append(token)
        else:
            filters['text_search'].append(token)
    
    return filters


def apply_project_image_filters(queryset, filters: dict):
    """Apply parsed filters to ProjectImage queryset."""
    from django.db import models
    
    # Status filters
    if filters['status']:
        queryset = queryset.filter(status=filters['status'])
    
    if filters['annotated'] is not None:
        queryset = queryset.filter(annotated=filters['annotated'])
    
    if filters['reviewed'] is not None:
        queryset = queryset.filter(reviewed=filters['reviewed'])
    
    if filters['marked_null'] is not None:
        queryset = queryset.filter(marked_as_null=filters['marked_null'])
    
    if filters['job_status']:
        queryset = queryset.filter(job_assignment_status=filters['job_status'])
    
    if filters['split']:
        queryset = queryset.filter(mode__mode=filters['split'])
    
    
    # Tag filters (through image)
    for tag in filters['tags']:
        queryset = queryset.filter(image__tags__name__iexact=tag)
    
    for cls in filters['classes']:
        queryset = queryset.filter(annotations__annotation_class__name__iexact=cls)
    
    # Filename filter
    if filters['filename']:
        queryset = queryset.filter(
            models.Q(image__name__icontains=filters['filename']) |
            models.Q(image__original_filename__icontains=filters['filename'])
        )
    
    # Dimension filters (through image)
    if filters['min_width']:
        queryset = queryset.filter(image__width__gte=filters['min_width'])
    if filters['max_width']:
        queryset = queryset.filter(image__width__lte=filters['max_width'])
    if filters['min_height']:
        queryset = queryset.filter(image__height__gte=filters['min_height'])
    if filters['max_height']:
        queryset = queryset.filter(image__height__lte=filters['max_height'])
    
    # Annotation count filters
    if filters['min_annotations']:
        queryset = queryset.annotate(
            annotation_count=models.Count('annotations', filter=models.Q(annotations__is_active=True))
        ).filter(annotation_count__gte=filters['min_annotations'])
    
    if filters['max_annotations']:
        queryset = queryset.annotate(
            annotation_count=models.Count('annotations', filter=models.Q(annotations__is_active=True))
        ).filter(annotation_count__lte=filters['max_annotations'])
    
    # Text search
    if filters['text_search']:
        search_query = ' '.join(filters['text_search'])
        queryset = queryset.filter(
            models.Q(image__name__icontains=search_query) |
            models.Q(image__original_filename__icontains=search_query)
        )
    
    # Sorting
    sort_mapping = {
        'size': '-image__file_size',
        'name': 'image__name',
        'date': '-added_at',
        'width': '-image__width',
        'height': '-image__height',
        'priority': '-priority',
        'annotations': '-annotation_count'
    }
    
    if filters['sort']:
        sort_field = sort_mapping.get(filters['sort'], '-priority')
        
        # Add annotation count if sorting by it
        if sort_field == '-annotation_count':
            queryset = queryset.annotate(
                annotation_count=models.Count('annotations', filter=models.Q(annotations__is_active=True))
            )
        
        queryset = queryset.order_by(sort_field, '-added_at')
    else:
        # Default sort
        queryset = queryset.order_by('-priority', '-added_at')
    
    return queryset

def apply_version_image_filters(queryset, filters: dict):
    """Apply parsed filters to ProjectImage queryset."""
    from django.db import models
    
    # Status filters
    if filters['status']:
        queryset = queryset.filter(project_image__status=filters['status'])
    
    if filters['annotated'] is not None:
        queryset = queryset.filter(project_image__annotated=filters['annotated'])
    
    if filters['reviewed'] is not None:
        queryset = queryset.filter(project_image__reviewed=filters['reviewed'])
    
    if filters['marked_null'] is not None:
        queryset = queryset.filter(project_image__marked_as_null=filters['marked_null'])
    
    if filters['job_status']:
        queryset = queryset.filter(project_image__job_assignment_status=filters['job_status'])
    
    if filters['split']:
        queryset = queryset.filter(project_image__mode__mode=filters['split'])
    
    
    # Tag filters (through image)
    for tag in filters['tags']:
        queryset = queryset.filter(project_image__image__tags__name__iexact=tag)
    
    for cls in filters['classes']:
        queryset = queryset.filter(project_image__annotations__annotation_class__name__iexact=cls)
    
    # Filename filter
    if filters['filename']:
        queryset = queryset.filter(
            models.Q(project_image__image__name__icontains=filters['filename']) |
            models.Q(project_image__image__original_filename__icontains=filters['filename'])
        )
    
    # Dimension filters (through image)
    if filters['min_width']:
        queryset = queryset.filter(project_image__image__width__gte=filters['min_width'])
    if filters['max_width']:
        queryset = queryset.filter(project_image__image__width__lte=filters['max_width'])
    if filters['min_height']:
        queryset = queryset.filter(project_image__image__height__gte=filters['min_height'])
    if filters['max_height']:
        queryset = queryset.filter(project_image__image__height__lte=filters['max_height'])
    
    # Annotation count filters
    if filters['min_annotations']:
        queryset = queryset.filter(annotation_count__gte=filters['min_annotations'])
    
    if filters['max_annotations']:
        queryset = queryset.filter(annotation_count__lte=filters['max_annotations'])
    
    # Text search
    if filters['text_search']:
        search_query = ' '.join(filters['text_search'])
        queryset = queryset.filter(
            models.Q(project_image__image__name__icontains=search_query) |
            models.Q(project_image__image__original_filename__icontains=search_query)
        )
    
    # Sorting
    sort_mapping = {
        'size': '-image__file_size',
        'name': 'image__name',
        'date': '-added_at',
        'width': '-image__width',
        'height': '-image__height',
        'priority': '-priority',
        'annotations': '-annotation_count'
    }
    
    if filters['sort']:
        sort_field = sort_mapping.get(filters['sort'], '-priority')
        
        # Add annotation count if sorting by it
        if sort_field == '-annotation_count':
            queryset = queryset.annotate(
                annotation_count=models.Count('annotations', filter=models.Q(project_image__annotations__is_active=True))
            )
        
        queryset = queryset.order_by(sort_field, '-added_at')
    else:
        # Default sort
        queryset = queryset.order_by('-project_image__priority', '-added_at')
    
    return queryset
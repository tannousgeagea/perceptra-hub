# api/routers/suggestions/services.py

from typing import List, Optional
from uuid import UUID
import uuid as uuid_lib
import json

import numpy as np
from pathlib import Path
from asgiref.sync import sync_to_async
from django.db import transaction

from annotations.models import Annotation, AnnotationClass, AnnotationType, SuggestionSession
from projects.models import Project, ProjectImage
from .schemas import *

class SuggestionService:
    """
    Handles suggestion lifecycle:
    - Create session (DB) + suggestion cache (Redis)
    - Run AI inference
    - Accept/reject/clear
    """
    
    def __init__(self, redis_client=None):
        # In production: inject Redis. Suggestions stored as:
        # key: f"suggestions:{session_id}"
        # value: JSON list of Suggestion objects
        # TTL: 1 hour
        self.redis = redis_client or self._get_redis()
        self.suggestion_ttl = 3600
    
    def _get_redis(self):
        # Placeholder - wire up your Redis connection
        from django.core.cache import cache
        return cache
    
    def _cache_key(self, session_id: UUID) -> str:
        return f"suggestions:{session_id}"
    
    # --------------------------------------------------------
    # Session Management
    # --------------------------------------------------------
    
    @sync_to_async
    def create_session(
        self,
        project: Project,
        image_id: str,
        user,
        config: ModelConfig,
    ) -> UUID:
        image = ProjectImage.objects.get(id=image_id, project=project)
        session, _ = SuggestionSession.objects.get_or_create(
            project_image=image,
            created_by=user,
            model_name=config.model,
            model_device=config.device,
            model_precision=config.precision,
        )
        # Initialize empty cache
        self.redis.set(self._cache_key(session.suggestion_id), json.dumps([]), self.suggestion_ttl)
        return session.suggestion_id
    
    @sync_to_async
    def get_session(self, session_id: UUID) -> SuggestionSession:
        """Get session with model config."""
        return SuggestionSession.objects.get(suggestion_id=session_id)
    
    @sync_to_async
    def update_session_model(
        self, 
        session_id: UUID, 
        config: ModelConfig
    ) -> None:
        """Update model configuration for existing session."""
        SuggestionSession.objects.filter(suggestion_id=session_id).update(
            model_name=config.model,
            model_device=config.device,
            model_precision=config.precision
        )
    
    @sync_to_async
    def update_session_source_type(self, session_id: UUID, source_type: SuggestionSourceType):
        """Track what operation was last performed."""
        SuggestionSession.objects.filter(suggestion_id=session_id).update(
            source_type=source_type.value
        )
        
    def store_suggestions(self, session_id: UUID, suggestions: List[Suggestion]):
        data = [s.model_dump() for s in suggestions]
        self.redis.set(self._cache_key(session_id), json.dumps(data), self.suggestion_ttl)
    
        # Update session count in DB
        self._update_session_count(session_id, len(suggestions))
    
    @sync_to_async
    def load_image(self, image_id: int) -> np.ndarray:
        """Load image as numpy array from storage."""
        from PIL import Image as PILImage
        import requests
        from io import BytesIO
        
        project_image = ProjectImage.objects.select_related(
            'image__storage_profile'
        ).get(id=image_id)
        
        # Get presigned URL from storage
        presigned_url = project_image.image.get_download_url(expiration=300)
        
        # Download and convert to numpy
        response = requests.get(presigned_url, timeout=30)
        response.raise_for_status()
        
        pil_img = PILImage.open(BytesIO(response.content)).convert("RGB")
        return np.array(pil_img)
    
    
    @sync_to_async
    def _get_session_db(self, session_id: UUID) -> SuggestionSession:
        return SuggestionSession.objects.get(suggestion_id=session_id)
    
    def _store_suggestions(self, session_id: UUID, suggestions: List[Suggestion]):
        data = [s.model_dump() for s in suggestions]
        self.redis.set(self._cache_key(session_id), json.dumps(data), self.suggestion_ttl)
    
    def _get_suggestions(self, session_id: UUID) -> List[dict]:
        cached = self.redis.get(self._cache_key(session_id))
        return json.loads(cached) if cached else []
    
    # --------------------------------------------------------
    # AI Inference Methods (integrate your SAM package here)
    # --------------------------------------------------------
    
    async def run_sam_auto(self, session_id: UUID, image_id: int, config: SAMAutoRequest):
        """Background task: run SAM auto-segmentation."""
        # 1. Load image
        image_path = await self._get_image_path(image_id)
        
        # 2. Call SAM service (your external package)
        # results = sam_service.auto_segment(image_path, config.points_per_side)
        results = []  # Placeholder
        
        # 3. Convert to suggestions
        suggestions = [
            Suggestion(
                suggestion_id=str(uuid_lib.uuid4()),
                bbox=BoundingBox(x=r['bbox'][0], y=r['bbox'][1], 
                                 width=r['bbox'][2], height=r['bbox'][3]),
                mask=r.get('mask_rle'),
                confidence=r['confidence']
            )
            for r in results
            if (r['bbox'][2] * r['bbox'][3]) >= config.min_area
        ]
        
        # 4. Cache results
        self._store_suggestions(session_id, suggestions)
        
        # 5. Update session metrics
        await self._update_session_count(session_id, len(suggestions))
    
    async def run_sam_point(
        self, session_id: UUID, image_id: int, points: List[PointPrompt]
    ) -> List[Suggestion]:
        """Synchronous SAM point prompt."""
        image_path = await self._get_image_path(image_id)
        
        # Call SAM with points
        # result = sam_service.point_segment(image_path, [(p.x, p.y, p.label) for p in points])
        result = {}  # Placeholder
        
        suggestions = []
        if result:
            suggestions.append(Suggestion(
                suggestion_id=str(uuid_lib.uuid4()),
                bbox=BoundingBox(x=result['bbox'][0], y=result['bbox'][1],
                                 width=result['bbox'][2], height=result['bbox'][3]),
                mask=result.get('mask_rle'),
                confidence=result.get('confidence', 0.9)
            ))
        
        self._store_suggestions(session_id, suggestions)
        await self._update_session_count(session_id, len(suggestions))
        return suggestions
    
    async def find_similar_objects(
        self,
        session_id: UUID,
        image_id: int,
        reference_uid: str,
        max_suggestions: int,
        min_similarity: float
    ) -> List[Suggestion]:
        """Find visually similar regions using embeddings."""
        # 1. Get reference annotation bbox
        ref_ann = await self._get_annotation(reference_uid)
        
        # 2. Extract embedding of reference region
        # ref_embedding = embedding_service.extract(image, ref_ann.data)
        
        # 3. Run sliding window / SAM proposals, compare embeddings
        # similar_regions = embedding_service.find_similar(image, ref_embedding, min_similarity)
        similar_regions = []  # Placeholder
        
        suggestions = [
            Suggestion(
                id=str(uuid_lib.uuid4()),
                bbox=BoundingBox(x=r['bbox'][0], y=r['bbox'][1],
                                 width=r['bbox'][2], height=r['bbox'][3]),
                confidence=r['similarity'],
                suggested_class_id=ref_ann.annotation_class.class_id,
                suggested_class_name=ref_ann.annotation_class.name,
                type="propagated",
                status="pending",
            )
            for r in similar_regions[:max_suggestions]
        ]
        
        self._store_suggestions(session_id, suggestions)
        await self._update_session_count(session_id, len(suggestions))
        return suggestions
    
    async def propagate_annotations(
        self,
        session_id: UUID,
        target_image_id: int,
        source_image_id: int,
        annotation_uids: Optional[List[str]]
    ) -> List[Suggestion]:
        """Copy annotations from previous frame as suggestions."""
        source_anns = await self._get_image_annotations(source_image_id, annotation_uids)
        
        suggestions = [
            Suggestion(
                suggestion_id=str(uuid_lib.uuid4()),
                bbox=BoundingBox(
                    x=ann.data[0], y=ann.data[1],
                    width=ann.data[2] - ann.data[0],
                    height=ann.data[3] - ann.data[1]
                ),
                confidence=0.95,  # High confidence for direct copy
                suggested_class_id=ann.annotation_class.class_id,
                suggested_class_name=ann.annotation_class.name
            )
            for ann in source_anns
        ]
        
        self._store_suggestions(session_id, suggestions)
        await self._update_session_count(session_id, len(suggestions))
        return suggestions
    
    async def suggest_labels(
        self,
        session_id: UUID,
        project: Project,
        image_id: int,
        bbox: BoundingBox,
        top_k: int
    ) -> List[Suggestion]:
        """Suggest class labels for a region using CLIP or classifier."""
        # 1. Get project classes
        classes = await self._get_project_classes(project)
        
        # 2. Extract region, run classifier
        # scores = classifier.predict(image, bbox, [c.name for c in classes])
        scores = []  # Placeholder: [(class_id, class_name, score), ...]
        
        suggestions = [
            Suggestion(
                suggestion_id=str(uuid_lib.uuid4()),
                bbox=bbox,
                confidence=score,
                suggested_class_id=class_id,
                suggested_class_name=class_name
            )
            for class_id, class_name, score in scores[:top_k]
        ]
        
        self._store_suggestions(session_id, suggestions)
        await self._update_session_count(session_id, len(suggestions))
        return suggestions
    
    # --------------------------------------------------------
    # Accept / Reject
    # --------------------------------------------------------
    
    async def accept_suggestions(
        self,
        session_id: UUID,
        suggestion_ids: List[str],
        class_id_override: Optional[str],
        class_name_override: Optional[str],
        user,
        project: Project,
        image_id: int
    ) -> List[str]:
        """Convert accepted suggestions into real annotations."""
        suggestions = self._get_suggestions(session_id)
        to_accept = {s['suggestion_id']: s for s in suggestions if s['suggestion_id'] in suggestion_ids}
        
        created_uids = []
        
        for sid, sugg in to_accept.items():
            # Handle both class_id and class_name
            if class_id_override:
                class_id = int(class_id_override)
            elif class_name_override:
                class_obj = await self._get_class_by_name(project, class_name_override)
                class_id = class_obj.class_id
            else:
                class_id = sugg.get('suggested_class_id')
            
            
            # Create annotation via existing logic
            uid = await self._create_annotation(
                project=project,
                image_id=image_id,
                bbox=sugg['bbox'],
                class_id=class_id,
                confidence=sugg['confidence'],
                source='prediction',
                user=user
            )
            created_uids.append(uid)
        
        # Update metrics
        await self._update_session_accepted(session_id, len(created_uids))
        
        # Remove accepted from cache
        remaining = [s for s in suggestions if s['suggestion_id'] not in suggestion_ids]
        self.redis.set(self._cache_key(session_id), json.dumps(remaining), self.suggestion_ttl)
        
        return created_uids
    
    @sync_to_async
    def _get_class_by_name(self, project: Project, name: str) -> AnnotationClass:
        return AnnotationClass.objects.get(
            name=name,
            annotation_group__project=project
        )
    
    async def reject_suggestions(self, session_id: UUID, suggestion_ids: List[str]):
        """Mark as rejected, remove from cache."""
        suggestions = self._get_suggestions(session_id)
        remaining = [s for s in suggestions if s['suggestion_id'] not in suggestion_ids]
        
        self.redis.set(self._cache_key(session_id), json.dumps(remaining), self.suggestion_ttl)
        await self._update_session_rejected(session_id, len(suggestion_ids))
    
    async def clear_session(self, session_id: UUID):
        """Remove all suggestions."""
        self.redis.delete(self._cache_key(session_id))
    
    # --------------------------------------------------------
    # Helpers (implement these with your ORM)
    # --------------------------------------------------------
    
    @sync_to_async
    def _get_image_path(self, image_id: int) -> str:
        img = ProjectImage.objects.get(id=image_id)
        return img.image.path
    
    @sync_to_async
    def _get_annotation(self, uid: str) -> Annotation:
        return Annotation.objects.select_related('annotation_class').get(
            annotation_uid=uid
        )
    
    @sync_to_async
    def _get_image_annotations(
        self, 
        image_id: int, 
        uids: Optional[List[str]] = None,
    ) -> List[Annotation]:
        qs = Annotation.objects.filter(
            project_image_id=image_id, 
            is_active=True
        ).select_related('annotation_class')
        if uids:
            qs = qs.filter(annotation_uid__in=uids)
        return list(qs)
    
    @sync_to_async
    def _get_project_classes(self, project: Project) -> List[AnnotationClass]:
        return list(AnnotationClass.objects.filter(annotation_group__project=project))
    
    @sync_to_async
    def _create_annotation(self, project, image_id, bbox, class_id, confidence, source, user) -> str:
        """Reuse existing annotation creation logic."""
        from annotations.models import Annotation, AnnotationClass, AnnotationType
        
        image = ProjectImage.objects.get(id=image_id)
        ann_class = AnnotationClass.objects.get(
            class_id=class_id,
            annotation_group__project=project
        )
        ann_type = AnnotationType.objects.get(name='Bounding Box')
        
        uid = str(uuid_lib.uuid4())
        
        with transaction.atomic():
            ann = Annotation.objects.create(
                project_image=image,
                annotation_type=ann_type,
                annotation_class=ann_class,
                data=[bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
                annotation_uid=uid,
                annotation_source=source,
                confidence=confidence,
                created_by=user,
                updated_by=user
            )
            if not image.annotated:
                image.annotated = True
                image.status = 'annotated'
                image.save(update_fields=['annotated', 'status'])
                
        return uid
    
    @sync_to_async
    def _update_session_count(self, session_id: UUID, count: int):
        SuggestionSession.objects.filter(suggestion_id=session_id).update(suggestions_generated=count)
    
    @sync_to_async
    def _update_session_accepted(self, session_id: UUID, count: int):
        from django.db.models import F
        SuggestionSession.objects.filter(suggestion_id=session_id).update(
            suggestions_accepted=F('suggestions_accepted') + count
        )
    
    @sync_to_async
    def _update_session_rejected(self, session_id: UUID, count: int):
        from django.db.models import F
        SuggestionSession.objects.filter(suggestion_id=session_id).update(
            suggestions_rejected=F('suggestions_rejected') + count
        )
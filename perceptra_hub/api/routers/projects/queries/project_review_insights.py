"""
GET /api/v1/projects/{project_id}/review-insights?days=30

Why work gets sent back in a project: send-back rate, the most common
reasons, the rate per annotator, and a weekly trend, from ImageReview.
Feeds labelling guidelines ("most send-backs are missed small objects").
"""
from collections import Counter, defaultdict
from datetime import timedelta
from typing import List, Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from django.db.models.functions import TruncWeek
from django.utils import timezone
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.dependencies import ProjectContext, get_project_context
from projects.models import ImageReview

router = APIRouter(prefix="/projects")

# The Send back dialog's one-click reasons (ReviewFeedback.tsx QUICK_REASONS).
# A note is counted under each quick reason it contains; anything else is "Other".
QUICK_REASONS = [
    "Missing objects",
    "Wrong class",
    "Boxes not tight",
    "Should be marked empty",
    "Not empty: objects missed",
]
OTHER = "Other"


class ReasonCount(BaseModel):
    reason: str
    count: int
    share: float                 # of send-backs, in %
    examples: List[str] = []     # recent free-text notes (Other) or details added to a quick reason


class AnnotatorQuality(BaseModel):
    user_id: str
    name: str
    reviews: int
    sent_back: int
    rate: float                  # %


class WeekPoint(BaseModel):
    week: str                    # ISO date of the week's Monday
    approved: int
    sent_back: int


class ReviewInsights(BaseModel):
    days: int
    reviews: int
    sent_back: int
    approved: int
    rate: Optional[float]        # sent back / reviews, %
    reasons: List[ReasonCount]
    annotators: List[AnnotatorQuality]
    weekly: List[WeekPoint]


def _reasons_in(note: str) -> tuple:
    """Quick reasons a note contains, and what is left over (free text)."""
    found, rest = [], note
    for quick in QUICK_REASONS:
        if quick.lower() in rest.lower():
            found.append(quick)
            start = rest.lower().index(quick.lower())
            rest = rest[:start] + rest[start + len(quick):]
    leftover = " ".join(part.strip(" ;,.") for part in rest.split(";")).strip(" ;,.")
    return found, leftover


@router.get("/{project_id}/review-insights", response_model=ReviewInsights, summary="Review Insights")
async def review_insights(
    project_id: UUID,
    days: int = Query(30, ge=1, le=365),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_project_access()

    @sync_to_async
    def compute():
        since = timezone.now() - timedelta(days=days)
        reviews = ImageReview.objects.filter(project=project_ctx.project, created_at__gte=since)
        totals = reviews.aggregate(total=Count("id"), sent_back=Count("id", filter=Q(approved=False)))
        total, sent_back = totals["total"], totals["sent_back"]

        # Reasons: newest first, so examples are recent.
        counts, examples = Counter(), defaultdict(list)
        for note in reviews.filter(approved=False).order_by("-created_at").values_list("reason", flat=True):
            found, leftover = _reasons_in(note or "")
            for quick in found:
                counts[quick] += 1
                if leftover and len(examples[quick]) < 3:
                    examples[quick].append(leftover[:200])
            if not found:
                counts[OTHER] += 1
                if leftover and len(examples[OTHER]) < 5:
                    examples[OTHER].append(leftover[:200])
        reasons = [
            ReasonCount(
                reason=reason,
                count=n,
                share=round(n / sent_back * 100, 1) if sent_back else 0.0,
                examples=examples.get(reason, []),
            )
            for reason, n in counts.most_common()
        ]

        per_annotator = (
            reviews.filter(annotator__isnull=False)
            .values("annotator_id", "annotator__username", "annotator__first_name", "annotator__last_name")
            .annotate(n=Count("id"), back=Count("id", filter=Q(approved=False)))
            .order_by("-back", "-n")
        )
        annotators = [
            AnnotatorQuality(
                user_id=str(row["annotator_id"]),
                name=(f"{row['annotator__first_name'] or ''} {row['annotator__last_name'] or ''}".strip()
                      or row["annotator__username"]),
                reviews=row["n"],
                sent_back=row["back"],
                rate=round(row["back"] / row["n"] * 100, 1) if row["n"] else 0.0,
            )
            for row in per_annotator
        ]

        weekly = [
            WeekPoint(week=row["week"].date().isoformat(), approved=row["n"] - row["back"], sent_back=row["back"])
            for row in reviews.annotate(week=TruncWeek("created_at"))
            .values("week")
            .annotate(n=Count("id"), back=Count("id", filter=Q(approved=False)))
            .order_by("week")
        ]

        return ReviewInsights(
            days=days,
            reviews=total,
            sent_back=sent_back,
            approved=total - sent_back,
            rate=round(sent_back / total * 100, 1) if total else None,
            reasons=reasons,
            annotators=annotators,
            weekly=weekly,
        )

    return await compute()

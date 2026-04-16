"""
similarity/admin.py
===================
Django admin registrations for the similarity subsystem.

All list views are read-only by default to prevent accidental mutations
of scan results through the admin interface.
"""

from django.contrib import admin
from django.utils.html import format_html
from unfold.admin import ModelAdmin
from similarity.models import (
    ImageHash,
    SimilarityScan,
    SimilarityCluster,
    SimilarityClusterMember,
    ClusterAction,
)


# ---------------------------------------------------------------------------

@admin.register(ImageHash)
class ImageHashAdmin(ModelAdmin):
    list_display  = ("image", "algorithm", "short_hash", "computed_at")
    list_filter   = ("algorithm",)
    search_fields = ("image__image_id", "hash_value")
    readonly_fields = ("image", "algorithm", "hash_value", "computed_at")

    def short_hash(self, obj):
        return obj.hash_value[:16] + "…"
    short_hash.short_description = "Hash (truncated)"


# ---------------------------------------------------------------------------

@admin.register(SimilarityScan)
class SimilarityScanAdmin(ModelAdmin):
    list_display  = (
        "scan_id", "organization", "scope", "algorithm",
        "similarity_threshold", "status", "progress_display",
        "clusters_found", "created_at",
    )
    list_filter   = ("status", "algorithm", "scope", "organization")
    search_fields = ("scan_id", "organization__name", "project__name")
    readonly_fields = (
        "scan_id", "organization", "project", "scope", "algorithm",
        "threshold", "similarity_threshold", "status",
        "total_images", "hashed_images", "clusters_found",
        "task_id", "initiated_by",
        "started_at", "completed_at", "created_at", "updated_at",
        "error_log",
    )

    def progress_display(self, obj):
        pct = obj.progress_pct
        color = "#22c55e" if pct == 100 else "#3b82f6"
        return format_html(
            '<div style="width:100px;background:#e5e7eb;border-radius:4px;">'
            '<div style="width:{pct}%;background:{color};height:8px;border-radius:4px;"></div>'
            '</div> {pct}%',
            pct=pct, color=color,
        )
    progress_display.short_description = "Progress"


# ---------------------------------------------------------------------------

class SimilarityClusterMemberInline(admin.TabularInline):
    model         = SimilarityClusterMember
    extra         = 0
    readonly_fields = ("image", "role", "similarity_score", "added_at")
    can_delete    = False


class ClusterActionInline(admin.TabularInline):
    model         = ClusterAction
    extra         = 0
    readonly_fields = ("action_type", "performed_by", "image_ids", "performed_at")
    can_delete    = False


@admin.register(SimilarityCluster)
class SimilarityClusterAdmin(ModelAdmin):
    list_display  = (
        "cluster_id", "scan", "member_count",
        "avg_similarity_display", "status", "created_at",
    )
    list_filter   = ("status", "scan__organization")
    search_fields = ("cluster_id", "scan__scan_id")
    readonly_fields = (
        "cluster_id", "scan", "representative", "member_count",
        "avg_similarity", "max_similarity", "status",
        "reviewed_by", "reviewed_at", "created_at",
    )
    inlines = [SimilarityClusterMemberInline, ClusterActionInline]

    def avg_similarity_display(self, obj):
        if obj.avg_similarity is None:
            return "—"
        pct = round(obj.avg_similarity * 100, 1)
        return f"{pct}%"
    avg_similarity_display.short_description = "Avg similarity"


# ---------------------------------------------------------------------------

@admin.register(SimilarityClusterMember)
class SimilarityClusterMemberAdmin(ModelAdmin):
    list_display  = ("image", "cluster", "role", "similarity_score", "added_at")
    list_filter   = ("role",)
    search_fields = ("image__image_id", "cluster__cluster_id")
    readonly_fields = ("image", "cluster", "role", "similarity_score", "added_at")


# ---------------------------------------------------------------------------

@admin.register(ClusterAction)
class ClusterActionAdmin(ModelAdmin):
    list_display  = ("action_id", "cluster", "action_type", "performed_by", "performed_at")
    list_filter   = ("action_type",)
    search_fields = ("cluster__cluster_id", "performed_by__username")
    readonly_fields = (
        "action_id", "cluster", "action_type",
        "performed_by", "image_ids", "meta", "performed_at",
    )
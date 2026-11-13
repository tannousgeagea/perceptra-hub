"""
What a dataset version contains versus what its export file contains.

A version's total_images / total_annotations / train_count / val_count /
test_count are written by the exporter and describe the exported file:
  - only images with ground-truth labels (manual or imported annotations;
    model predictions too when the export config has include_predictions),
  - plus augmented copies, which count in total_images but in no split.

These helpers work out the version's own numbers next to that, from data
already stored (no extra fields).
"""
from datetime import timedelta

from django.db.models import Count, Q, QuerySet

from projects.models import Version

# Annotation sources the exporter treats as ground truth
# (common_utils/dataset_export/streaming/base.py get_annotations).
GROUND_TRUTH_SOURCES = ("manual", "imported")

# Finishing an export saves the version just after setting exported_at, so
# updated_at is a few milliseconds later; anything beyond this is a change.
_EXPORT_SAVE_SLACK = timedelta(seconds=5)


def with_export_breakdown(queryset: QuerySet) -> QuerySet:
    """Annotate versions with the counts export_breakdown() needs (one query)."""
    active = Q(version_images__project_image__annotations__is_active=True)
    return queryset.annotate(
        images_in_version=Count("version_images", distinct=True),
        images_with_annotations=Count("version_images", filter=active, distinct=True),
        images_with_labels=Count(
            "version_images",
            filter=active & Q(
                version_images__project_image__annotations__annotation_source__in=GROUND_TRUTH_SOURCES
            ),
            distinct=True,
        ),
    )


def export_breakdown(version: Version) -> dict:
    """The version-side fields of VersionResponse, for a version from with_export_breakdown()."""
    include_predictions = bool((version.export_config or {}).get("include_predictions"))
    exported = version.export_status == "completed" and version.exported_at is not None

    no_labels = version.images_in_version - version.images_with_annotations
    predictions_only = version.images_with_annotations - version.images_with_labels
    split_total = version.train_count + version.val_count + version.test_count

    return {
        "images_in_version": version.images_in_version,
        "augmented_images": max(0, version.total_images - split_total) if exported else 0,
        # Based on the labels as they are now: what the next export would skip.
        "not_exported_no_labels": no_labels,
        "not_exported_predictions_only": 0 if include_predictions else predictions_only,
        # Adding or removing images saves the version; renaming is blocked once exported.
        "changed_since_export": bool(
            exported and version.updated_at - version.exported_at > _EXPORT_SAVE_SLACK
        ),
    }

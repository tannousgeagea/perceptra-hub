"""
Give every annotation created without one (ingestion, auto-annotate, agent
inference jobs) its own annotation_uid, before the column becomes NOT NULL.
"""
import uuid

from django.db import migrations

BATCH = 2000


def backfill(apps, schema_editor):
    Annotation = apps.get_model("annotations", "Annotation")
    while True:
        rows = list(Annotation.objects.filter(annotation_uid__isnull=True).only("id")[:BATCH])
        if not rows:
            break
        for row in rows:
            row.annotation_uid = str(uuid.uuid4())
        Annotation.objects.bulk_update(rows, ["annotation_uid"])


class Migration(migrations.Migration):

    dependencies = [
        ('annotations', '0021_annotation_external_id_meta_imported'),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]

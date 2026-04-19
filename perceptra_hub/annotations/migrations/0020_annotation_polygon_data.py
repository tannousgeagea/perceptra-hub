from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("annotations", "0019_annotationaudit_iou"),
    ]

    operations = [
        migrations.AddField(
            model_name="annotation",
            name="polygon_data",
            field=models.JSONField(
                blank=True,
                null=True,
                help_text="Normalized polygon contour [[x,y], ...] from SAM mask. Complements the bbox in `data`.",
            ),
        ),
    ]

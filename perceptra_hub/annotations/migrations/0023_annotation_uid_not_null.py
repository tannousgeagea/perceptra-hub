from django.db import migrations, models

import annotations.models


class Migration(migrations.Migration):

    dependencies = [
        ('annotations', '0022_backfill_annotation_uid'),
    ]

    operations = [
        migrations.AlterField(
            model_name='annotation',
            name='annotation_uid',
            field=models.CharField(default=annotations.models.new_annotation_uid, max_length=100, unique=True),
        ),
    ]

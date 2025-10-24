from django.core.management.base import BaseCommand
from annotations.models import Annotation, AnnotationAudit

class Command(BaseCommand):
    help = "Audit existing annotations and assign TP, FP, FN statuses based on manual edits."

    def handle(self, *args, **kwargs):
        total = 0
        tp, fp, fn = 0, 0, 0

        for ann in Annotation.objects.all():
            original_data = ann.data
            original_class_id = ann.annotation_class_id

            if ann.annotation_source == 'prediction':
                if not ann.is_active:
                    status = 'FP'
                    was_edited = False
                    fp += 1
                elif ann.annotation_class_id != original_class_id or ann.data != original_data:
                    status = 'FP'
                    was_edited = True
                    fp += 1
                else:
                    status = 'TP'
                    was_edited = False
                    tp += 1

                AnnotationAudit.objects.update_or_create(
                    annotation=ann,
                    defaults={
                        "evaluation_status": status,
                        "was_edited": was_edited,
                    }
                )

            elif ann.annotation_source == 'manual':
                AnnotationAudit.objects.update_or_create(
                    annotation=ann,
                    defaults={
                        "evaluation_status": "FN",
                        "was_edited": False,
                    }
                )
                fn += 1

            total += 1

        self.stdout.write(self.style.SUCCESS(
            f"Audited {total} annotations: {tp} TP, {fp} FP, {fn} FN."
        ))

"""
Bring every job's status in line with its images (see jobs.lifecycle).

    python manage.py sync_job_statuses --dry-run      # show what would change
    python manage.py sync_job_statuses                # apply, no notifications
    python manage.py sync_job_statuses --project "Project Example"
"""
from collections import Counter

from django.core.management.base import BaseCommand

from jobs.lifecycle import derive_status, job_progress, sync_jobs
from jobs.models import Job, JobStatus


class Command(BaseCommand):
    help = "Set each job's status from its images' progress (no notifications are sent)."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='List the changes without saving them')
        parser.add_argument('--project', help='Only jobs of the project with this name')

    def handle(self, *args, dry_run=False, project=None, **options):
        jobs = Job.objects.filter(is_active=True).exclude(status=JobStatus.SLICED).select_related('project')
        if project:
            jobs = jobs.filter(project__name=project)

        moves = Counter()
        changed = []
        for job in jobs.order_by('project_id', 'id'):
            progress = job_progress(job.pk)
            status = derive_status(job, progress)
            if status == job.status:
                continue
            moves[(job.status, status)] += 1
            changed.append(job.pk)
            self.stdout.write(
                f"{job.project.name} / {job.name}: {job.status} → {status}  "
                f"({progress.done} reviewed, {progress.waiting} to review, {progress.pending} to do, of {progress.total})"
            )

        if not changed:
            self.stdout.write(self.style.SUCCESS('All job statuses are already up to date.'))
            return
        if dry_run:
            self.stdout.write(self.style.WARNING(f"Dry run: {len(changed)} job(s) would change."))
            return

        done = sync_jobs(changed, announce=False)
        summary = ', '.join(f"{a} → {b}: {n}" for (a, b), n in sorted(moves.items()))
        self.stdout.write(self.style.SUCCESS(f"Updated {len(done)} job(s). {summary}"))

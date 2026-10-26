"""
Deep links into a job: GET /projects/{pid}/jobs/{job_id}/images/{image_id}/position.

    DATABASE_ENGINE=django.db.backends.sqlite3 python manage.py test jobs
"""
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase

from images.models import Image
from jobs.models import Job, JobImage
from organizations.models import Organization
from projects.models import Project, ProjectImage, ProjectType, Visibility
from storage.models import StorageProfile


class JobImagePositionApiTest(TransactionTestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.dependencies import get_project_context
        from api.routers.projects.queries import project_job_image_position

        user = get_user_model().objects.create_user(email="u@example.com", password="x")
        self.org = Organization.objects.create(name="Org", slug="org", owner=user)
        profile = StorageProfile.objects.create(organization=self.org, name="p", backend="local",
                                                config={"base_path": "/tmp"})
        self.project = Project.objects.create(
            name="p", organization=self.org,
            project_type=ProjectType.objects.create(name="Object Detection"),
            visibility=Visibility.objects.create(name="Private"),
        )
        self.job = Job.objects.create(project=self.project, name="Job 1")
        # 25 unannotated images, then 3 annotated: tabs and pages differ
        for i in range(28):
            image = Image.objects.create(
                image_id=f"img-{i}", organization=self.org, storage_profile=profile, storage_key=f"k{i}.jpg",
                name=f"{i}", original_filename=f"{i}.jpg", file_format="jpg", file_size=1, width=1, height=1,
                image_name=f"{i}",
            )
            pi = ProjectImage.objects.create(project=self.project, image=image, annotated=i >= 25)
            JobImage.objects.create(job=self.job, project_image=pi)

        app = FastAPI()
        app.include_router(project_job_image_position.router, prefix="/api/v1")
        app.dependency_overrides[get_project_context] = lambda: SimpleNamespace(project=self.project)
        self.client = TestClient(app)

    def position(self, image_id, job_id=None, limit=20):
        return self.client.get(
            f"/api/v1/projects/{self.project.project_id}/jobs/{job_id or self.job.id}/images/{image_id}/position",
            params={"limit": limit},
        )

    def test_page_and_index_within_the_images_status_tab(self):
        body = self.position("img-22").json()
        self.assertEqual((body["status"], body["index"], body["page"], body["total"]), ("unannotated", 23, 2, 25))
        self.assertEqual(body["job_name"], "Job 1")

        annotated = self.position("img-26").json()
        self.assertEqual((annotated["status"], annotated["index"], annotated["page"], annotated["total"]),
                         ("annotated", 2, 1, 3))

    def test_image_not_in_job_or_unknown_job(self):
        self.assertEqual(self.position("nope").status_code, 404)
        self.assertEqual(self.position("img-1", job_id=999999).status_code, 404)

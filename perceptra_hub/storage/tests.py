"""
External (read-only) storage profiles, adapter caching and signed media URLs.

    DATABASE_ENGINE=django.db.backends.sqlite3 python manage.py test ingest storage
"""
import shutil
import tempfile
import time
from io import BytesIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase, TransactionTestCase, override_settings

from images.models import Image
from organizations.models import Organization
from perceptra_storage import ReadOnlyStorageAdapter, StoragePermissionError
from storage import signing
from storage.models import StorageProfile
from storage.services import (
    get_default_storage_profile,
    get_storage_adapter_for_profile,
    get_writable_storage_profile,
    invalidate_profile_adapter,
)


def _org(slug: str) -> Organization:
    owner = get_user_model().objects.create_user(email=f"{slug}@example.com", password="x")
    return Organization.objects.create(name=slug, slug=slug, owner=owner)


class ExternalProfileTest(TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, True)
        self.org = _org("org")
        self.managed = StorageProfile.objects.create(
            organization=self.org, name="managed", backend="local", is_default=True,
            config={"base_path": self.base},
        )
        self.external = StorageProfile.objects.create(
            organization=self.org, name="external", backend="local", access_mode="external_readonly",
            config={"base_path": self.base},
        )

    def test_external_profile_cannot_be_default(self):
        self.external.is_default = True
        with self.assertRaises(ValidationError):
            self.external.save()

    def test_writes_never_pick_an_external_profile(self):
        StorageProfile.objects.filter(pk=self.managed.pk).update(is_default=False)
        StorageProfile.objects.filter(pk=self.external.pk).update(is_default=True)  # a legacy row
        self.assertIsNone(get_default_storage_profile(self.org))
        self.assertIsNone(get_writable_storage_profile(self.org, preferred=self.external))
        self.assertEqual(get_writable_storage_profile(self.org, fallback_to_any=True), self.managed)

    def test_external_adapter_is_read_only_and_delete_only_unlinks(self):
        get_storage_adapter_for_profile(self.managed).upload_file(BytesIO(b"img"), "a.jpg")
        adapter = get_storage_adapter_for_profile(self.external)
        self.assertIsInstance(adapter, ReadOnlyStorageAdapter)
        with self.assertRaises(StoragePermissionError):
            adapter.delete_file("a.jpg")

        image = Image.objects.create(
            image_id="i1", organization=self.org, storage_profile=self.external, storage_key="a.jpg",
            name="a", original_filename="a.jpg", file_format="jpg", file_size=3, width=1, height=1, image_name="a",
        )
        self.assertFalse(image.delete_from_storage())
        self.assertEqual(adapter.download_file("a.jpg"), b"img")

    def test_adapter_cache_and_invalidation(self):
        first = get_storage_adapter_for_profile(self.managed)
        self.assertIs(get_storage_adapter_for_profile(self.managed), first)
        invalidate_profile_adapter(self.managed)
        second = get_storage_adapter_for_profile(self.managed)
        self.assertIsNot(second, first)

        self.managed.config = {**self.managed.config, "create_dirs": True}
        self.managed.save()  # updated_at changes: a new adapter
        self.assertIsNot(get_storage_adapter_for_profile(self.managed), second)


@override_settings(SECRET_ENCRYPTION_KEY="pV0x4cB5Y8mH3vO1nL9kQ2sW7tR6uE4iA0zX5cV8bN4=")
class CreateProfileWithCredentialsApiTest(TransactionTestCase):
    """POST /api/v1/storage/profiles-with-credentials accepts profile.access_mode."""

    def setUp(self):
        from types import SimpleNamespace

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.dependencies import get_request_context
        from api.routers.storage.queries import storage_with_credential

        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, True)
        self.org = _org("api-org")
        app = FastAPI()
        app.include_router(storage_with_credential.router, prefix="/api/v1")
        ctx = SimpleNamespace(organization=self.org, require_role=lambda *roles: None)
        app.dependency_overrides[get_request_context] = lambda: ctx
        self.client = TestClient(app)

    def post(self, **profile):
        return self.client.post("/api/v1/storage/profiles-with-credentials", json={
            "profile": {"name": "dropzone", "backend": "local", "config": {"base_path": self.base}, **profile},
            "credentials": {"type": "connection_string", "connection_string": ""},
            "test_before_save": True,
        })

    def test_external_readonly_is_stored_and_echoed(self):
        response = self.post(access_mode="external_readonly")
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["access_mode"], "external_readonly")
        profile = StorageProfile.objects.get(organization=self.org, name="dropzone")
        self.assertTrue(profile.is_external)

    def test_default_is_managed(self):
        response = self.post()
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["access_mode"], "managed")

    def test_external_default_is_rejected_before_any_secret_is_created(self):
        from storage.models import EncryptedSecret

        response = self.post(access_mode="external_readonly", is_default=True)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertFalse(EncryptedSecret.objects.filter(organization=self.org).exists())

    def test_unknown_access_mode_is_a_validation_error(self):
        self.assertEqual(self.post(access_mode="readonly").status_code, 422)


class SigningTest(TestCase):
    @override_settings(MEDIA_URL_SIGNING_KEY="k1")
    def test_sign_and_verify(self):
        query = dict(p.split("=", 1) for p in signing.signed_query("/api/v1/media/images/x", 60).split("&"))
        self.assertTrue(signing.verify("/api/v1/media/images/x", query["exp"], query["sig"]))
        self.assertFalse(signing.verify("/api/v1/media/images/y", query["exp"], query["sig"]))
        self.assertFalse(signing.verify("/api/v1/media/images/x", str(int(time.time()) - 1), query["sig"]))
        with override_settings(MEDIA_URL_SIGNING_KEY="k2"):
            self.assertFalse(signing.verify("/api/v1/media/images/x", query["exp"], query["sig"]))

    def test_image_falls_back_to_proxy_when_storage_cannot_presign(self):
        org = _org("org2")
        profile = StorageProfile(organization=org, name="ext-azure", backend="azure",
                                 access_mode="external_readonly", config={})
        image = Image(image_id="img-9", organization=org, storage_profile=profile, storage_key="k.jpg")
        with patch("storage.services.get_storage_adapter_for_profile") as get_adapter:
            get_adapter.return_value.can_presign = False
            url = image.get_download_url(expiration=120)
        self.assertTrue(url.startswith("/api/v1/media/images/img-9?exp="), url)
        self.assertIn("&sig=", url)

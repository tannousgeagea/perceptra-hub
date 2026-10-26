"""
External (read-only) storage profiles, adapter caching and signed media URLs.

    DATABASE_ENGINE=django.db.backends.sqlite3 python manage.py test ingest storage
"""
import shutil
import tempfile
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

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


class BrowseApiTest(TransactionTestCase):
    """GET /api/v1/storage/profiles/{id}/browse — the console's file picker."""

    def setUp(self):
        import os
        from types import SimpleNamespace

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.dependencies import get_request_context
        from api.routers.storage.queries import storage_browse

        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, True)
        for key in ("cam1/a.jpg", "cam1/b.jpg", "cam1/notes.txt", "cam2/c.png", "root.jpg"):
            os.makedirs(os.path.join(self.base, os.path.dirname(key)), exist_ok=True)
            with open(os.path.join(self.base, key), "wb") as fh:
                fh.write(b"x")
        self.org = _org("browse-org")
        self.profile = StorageProfile.objects.create(
            organization=self.org, name="ext", backend="local", access_mode="external_readonly",
            config={"base_path": self.base},
        )
        Image.objects.create(
            image_id="already", organization=self.org, storage_profile=self.profile, storage_key="cam1/a.jpg",
            name="a", original_filename="a.jpg", file_format="jpg", file_size=1, width=1, height=1, image_name="a",
        )
        app = FastAPI()
        app.include_router(storage_browse.router, prefix="/api/v1")
        ctx = SimpleNamespace(organization=self.org, api_key=None, has_role=lambda *roles: True)
        app.dependency_overrides[get_request_context] = lambda: ctx
        self.client = TestClient(app)
        self.url = f"/api/v1/storage/profiles/{self.profile.storage_profile_id}/browse"

    def test_root_lists_folders_and_images(self):
        body = self.client.get(self.url).json()
        self.assertEqual(body["folders"], ["cam1/", "cam2/"])
        self.assertEqual([f["key"] for f in body["files"]], ["root.jpg"])

    def test_folder_flags_registered_files_and_hides_non_images(self):
        body = self.client.get(self.url, params={"prefix": "cam1/"}).json()
        files = {f["key"]: f for f in body["files"]}
        self.assertEqual(sorted(files), ["cam1/a.jpg", "cam1/b.jpg"])
        self.assertTrue(files["cam1/a.jpg"]["registered"])
        self.assertEqual(files["cam1/a.jpg"]["image_id"], "already")
        self.assertFalse(files["cam1/b.jpg"]["registered"])

    def test_paging(self):
        first = self.client.get(self.url, params={"prefix": "cam1/", "page_size": 1, "images_only": False}).json()
        self.assertIsNotNone(first["next_token"])

    def test_credential_without_list_gets_a_helpful_403(self):
        from perceptra_storage import StoragePermissionError

        with patch("perceptra_storage.ReadOnlyStorageAdapter.list_page", side_effect=StoragePermissionError("no list")):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 403)
        self.assertIn("Paste the file keys", response.json()["detail"])


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

    def test_signed_url_is_stable_within_a_window_and_lasts_at_least_its_ttl(self):
        with patch("storage.signing.time.time", return_value=3600 * 10 + 5):
            first = signing.signed_query("/p", 3600)
        with patch("storage.signing.time.time", return_value=3600 * 11 - 1):
            same = signing.signed_query("/p", 3600)
        with patch("storage.signing.time.time", return_value=3600 * 11):
            later = signing.signed_query("/p", 3600)
        self.assertEqual(first, same)
        self.assertNotEqual(first, later)
        self.assertIn("exp=43200", first)  # minted at 10h00:05, valid until 12h00

    @override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}})
    def test_presigned_url_is_reused_within_a_window(self):
        from storage.services import cached_presigned_url

        profile = StorageProfile.objects.create(organization=_org("org3"), name="s3", backend="s3",
                                                config={"bucket_name": "b"})
        adapter = MagicMock()
        adapter.generate_presigned_url.side_effect = lambda key, expiration, method: MagicMock(
            url=f"https://s3/{key}?sig={adapter.generate_presigned_url.call_count}")
        with patch("storage.services.time.time", return_value=3600 * 10 + 5):
            first = cached_presigned_url(profile, adapter, "a.jpg", 3600)
            self.assertEqual(cached_presigned_url(profile, adapter, "a.jpg", 3600), first)
            other = cached_presigned_url(profile, adapter, "b.jpg", 3600)
        with patch("storage.services.time.time", return_value=3600 * 11 + 5):
            next_window = cached_presigned_url(profile, adapter, "a.jpg", 3600)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, next_window)
        self.assertEqual(adapter.generate_presigned_url.call_count, 3)
        adapter.generate_presigned_url.assert_called_with("a.jpg", expiration=7200, method="GET")


class MediaProxyTest(TransactionTestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.routers.images.queries import media

        org = _org("org-proxy")
        profile = StorageProfile.objects.create(organization=org, name="ext", backend="azure",
                                                access_mode="external_readonly",
                                                config={"account_name": "a", "container_name": "c"})
        self.data = bytes(range(256)) * 40  # 10 KB
        self.image = Image.objects.create(
            image_id="img-p", organization=org, storage_profile=profile, storage_key="a/b.jpg", name="b",
            original_filename="b.jpg", file_format="jpg", file_size=len(self.data), width=1, height=1,
            image_name="b", checksum="abc",
        )
        self.adapter = MagicMock()
        self.adapter.download_file.return_value = self.data
        self.adapter.download_range.side_effect = lambda key, offset, length: self.data[offset:offset + length]
        patcher = patch("storage.services.get_storage_adapter_for_profile", return_value=self.adapter)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(media.router, prefix="/api/v1")
        self.client = TestClient(app)
        self.media = media

    def url(self):
        return self.image.get_proxy_url(3600)

    def test_serves_the_file_with_cache_headers_and_answers_304_on_revalidation(self):
        response = self.client.get(self.url())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.data)
        self.assertEqual(response.headers["etag"], '"img-p-abc"')
        max_age = int(response.headers["cache-control"].split("max-age=")[1])
        self.assertGreaterEqual(max_age, 3600 - 5)

        self.adapter.download_file.reset_mock()
        again = self.client.get(self.url(), headers={"If-None-Match": '"img-p-abc"'})
        self.assertEqual(again.status_code, 304)
        self.adapter.download_file.assert_not_called()

    def test_large_files_are_streamed_in_ranged_reads(self):
        with patch.object(self.media, "PROXY_CHUNK_BYTES", 4096):
            response = self.client.get(self.url())
        self.assertEqual(response.content, self.data)
        self.adapter.download_file.assert_not_called()
        self.assertEqual(self.adapter.download_range.call_count, 3)  # 4096 + 4096 + 2048

    def test_bad_signature_is_rejected(self):
        self.assertEqual(self.client.get("/api/v1/media/images/img-p?exp=9999999999&sig=x").status_code, 403)

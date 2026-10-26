"""
Register-by-reference ingestion (ingest/services.py).

    DATABASE_ENGINE=django.db.backends.sqlite3 python manage.py test ingest storage
"""
import shutil
import tempfile
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from annotations.models import Annotation, AnnotationClass, AnnotationGroup
from images.models import Image, ImageTag
from ingest.services import IngestOptions, register_items, remove_items
from organizations.models import Organization
from projects.models import Project, ProjectImage, ProjectType, Visibility
from storage.models import StorageProfile

User = get_user_model()


class IngestFixture(TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, True)
        self.user = User.objects.create_user(email="producer@example.com", password="x")
        self.org = Organization.objects.create(name="Org", slug="org", owner=self.user)
        self.profile = StorageProfile.objects.create(
            organization=self.org, name="external", backend="local", access_mode="external_readonly",
            config={"base_path": self.base},
        )
        self.project = Project.objects.create(
            name="impurities", organization=self.org,
            project_type=ProjectType.objects.create(name="Object Detection"),
            visibility=Visibility.objects.create(name="Private"),
        )
        group = AnnotationGroup.objects.create(project=self.project, name="default")
        self.impurity = AnnotationClass.objects.create(annotation_group=group, class_id=0, name="impurity")

    def item(self, **overrides):
        item = {
            "external_id": "sample-1",
            "storage_profile_name": "external",
            "storage_key": "images/t1/2026/10/sample-1.jpg",
            "width": 1920, "height": 1080, "file_size": 1234, "checksum": "ab" * 32,
            "metadata": {"tenant": "t1", "frame_uid": "f-1"},
            "tags": ["tenant:t1", "delivery:D-7"],
            "annotations": [
                {"external_id": "alarm:e1", "class_name": "impurity", "bbox_xyxyn": [0.1, 0.1, 0.3, 0.3],
                 "source": "verified", "metadata": {"label_hint": "verified_positive"}},
                {"external_id": "edge_raw:f-1:abc", "class_name": "impurity", "bbox_xyxyn": [0.5, 0.5, 0.6, 0.7],
                 "source": "prediction", "confidence": 0.42},
            ],
        }
        item.update(overrides)
        return item

    def register(self, *items, **options):
        return register_items(self.org, self.user, list(items), project=self.project,
                              options=IngestOptions(**options), source_label="api:data_hub")


class RegisterTest(IngestFixture):
    def test_registers_by_reference_without_touching_storage(self):
        [result] = self.register(self.item())
        self.assertEqual(result["status"], "created", result)

        image = Image.objects.get(external_id="sample-1")
        self.assertEqual((image.storage_profile, image.storage_key), (self.profile, "images/t1/2026/10/sample-1.jpg"))
        self.assertEqual((image.width, image.height, image.file_format), (1920, 1080, "jpg"))
        self.assertEqual(image.source_of_origin, "api:data_hub")
        self.assertEqual(sorted(ImageTag.objects.filter(image=image).values_list("tag__name", flat=True)),
                         ["delivery:D-7", "tenant:t1"])

        pi = ProjectImage.objects.get(project=self.project, image=image)
        self.assertEqual((pi.job_assignment_status, pi.annotated, pi.status), ("excluded", True, "annotated"))

        verified = Annotation.objects.get(external_id="alarm:e1")
        self.assertEqual((verified.annotation_source, verified.reviewed), ("imported", True))
        prediction = Annotation.objects.get(external_id="edge_raw:f-1:abc")
        self.assertEqual((prediction.annotation_source, prediction.confidence, prediction.reviewed),
                         ("prediction", 0.42, False))

    def test_same_payload_is_unchanged_and_idempotent(self):
        self.register(self.item())
        [result] = self.register(self.item())
        self.assertEqual(result["status"], "unchanged", result)
        self.assertEqual(Image.objects.count(), 1)
        self.assertEqual(Annotation.objects.count(), 2)

    def test_new_revision_updates_in_place_and_withdraws_dropped_annotations(self):
        self.register(self.item())
        revised = self.item(tags=["tenant:t1"], annotations=[
            {"external_id": "alarm:e1", "class_name": "impurity", "bbox_xyxyn": [0.1, 0.1, 0.35, 0.3],
             "source": "verified"},
        ])
        [result] = self.register(revised)
        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["annotations"]["removed"], 1)
        self.assertEqual(Annotation.objects.get(external_id="alarm:e1").data, [0.1, 0.1, 0.35, 0.3])
        self.assertTrue(Annotation.objects.get(external_id="edge_raw:f-1:abc").is_deleted)
        self.assertEqual(list(ImageTag.objects.values_list("tag__name", flat=True)), ["tenant:t1"])

    def test_human_edits_always_win(self):
        self.register(self.item())
        edited = Annotation.objects.get(external_id="edge_raw:f-1:abc")
        edited.data = [0.5, 0.5, 0.65, 0.75]  # an annotator moved the box
        edited.save()

        moved = self.item()
        moved["annotations"][1]["bbox_xyxyn"] = [0.4, 0.4, 0.6, 0.7]
        [result] = self.register(moved)
        self.assertEqual([c["external_id"] for c in result["annotations"]["conflicts"]], ["edge_raw:f-1:abc"])
        self.assertEqual(Annotation.objects.get(external_id="edge_raw:f-1:abc").data, [0.5, 0.5, 0.65, 0.75])

    def test_unknown_class_is_reported_or_created(self):
        bad = self.item(annotations=[{"external_id": "a", "class_name": "mattress", "bbox_xyxyn": [0, 0, .1, .1]}])
        [result] = self.register(bad)
        self.assertEqual(result["status"], "created")
        self.assertIn("does not exist", result["annotations"]["errors"][0]["error"])

        [result] = self.register(bad, create_missing_classes=True)
        self.assertEqual(result["annotations"]["created"], 1)
        self.assertTrue(AnnotationClass.objects.filter(name="mattress", class_id=1).exists())

    def test_bad_item_does_not_sink_the_batch(self):
        results = self.register(self.item(width=0), self.item(external_id="sample-2", storage_key="k2.jpg"))
        self.assertEqual([r["status"] for r in results], ["error", "created"])
        self.assertIn("width and height", results[0]["error"])

    def test_storage_key_is_unique_per_profile_not_per_org(self):
        other = StorageProfile.objects.create(
            organization=self.org, name="external-2", backend="local", access_mode="external_readonly",
            config={"base_path": self.base},
        )
        a = self.register(self.item(annotations=None))[0]
        b = self.register(self.item(external_id="sample-x", storage_profile_name="external-2", annotations=None))[0]
        self.assertEqual((a["status"], b["status"]), ("created", "created"))
        self.assertEqual(Image.objects.filter(storage_key=self.item()["storage_key"]).count(), 2)
        self.assertEqual(Image.objects.get(external_id="sample-x").storage_profile, other)

    def test_mark_as_null_hard_negative(self):
        [result] = self.register(self.item(annotations=[], mark_as_null=True))
        pi = ProjectImage.objects.get(pk=result["project_image_id"])
        self.assertTrue(pi.marked_as_null)


class RemoveTest(IngestFixture):
    def test_remove_unlinks_but_never_touches_storage_or_human_work(self):
        self.register(self.item())
        human = Annotation.objects.get(external_id="alarm:e1")
        human.data = [0.12, 0.1, 0.3, 0.3]
        human.save()

        [result] = remove_items(self.org, ["sample-1"], project=self.project)
        self.assertEqual((result["status"], result["annotations_removed"]), ("removed", 1))
        self.assertFalse(ProjectImage.objects.get(image__external_id="sample-1").is_active)
        self.assertFalse(Annotation.objects.get(external_id="alarm:e1").is_deleted, "edited by a person")
        self.assertTrue(Image.objects.filter(external_id="sample-1").exists())

        # Re-registering brings the link back.
        self.register(self.item())
        self.assertTrue(ProjectImage.objects.get(image__external_id="sample-1").is_active)


class HeaderProbeTest(IngestFixture):
    """Registering without width/height reads only the image header."""

    def write_image(self, key, size=(640, 480), fmt="JPEG"):
        import os

        from PIL import Image as PILImage

        path = os.path.join(self.base, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        PILImage.new("RGB", size, (10, 20, 30)).save(path, format=fmt)
        return os.path.getsize(path)

    def unsized(self, key, **extra):
        return {"storage_profile_name": "external", "storage_key": key, "tags": ["console"], **extra}

    def test_size_is_probed_from_the_header(self):
        nbytes = self.write_image("cam1/a.jpg", (800, 600))
        self.write_image("cam1/b.png", (320, 200), "PNG")
        results = self.register(self.unsized("cam1/a.jpg"), self.unsized("cam1/b.png"))
        self.assertEqual([r["status"] for r in results], ["created", "created"], results)
        self.assertTrue(all(r.get("probed") for r in results))
        a = Image.objects.get(storage_key="cam1/a.jpg")
        self.assertEqual((a.width, a.height, a.file_size), (800, 600, nbytes))
        self.assertEqual((Image.objects.get(storage_key="cam1/b.png").width,), (320,))

    def test_registered_image_is_not_probed_again(self):
        self.write_image("cam1/a.jpg")
        self.register(self.unsized("cam1/a.jpg"))
        with patch("ingest.probe.probe_image") as probe:
            [result] = self.register(self.unsized("cam1/a.jpg"))
        probe.assert_not_called()
        self.assertEqual(result["status"], "unchanged")

    def test_large_exif_header_needs_the_second_read(self):
        from ingest.probe import READ_STEPS, probe_image
        from storage.services import get_storage_adapter_for_profile

        import os

        from PIL import Image as PILImage

        # Noise compresses badly: a file well over the first read.
        PILImage.frombytes("RGB", (1024, 768), os.urandom(1024 * 768 * 3)).save(
            os.path.join(self.base, "big.jpg"), format="JPEG", quality=95)
        adapter = get_storage_adapter_for_profile(self.profile)
        original = adapter.download_range
        calls = []

        def ranged(key, offset=0, length=65536):
            calls.append(length)
            if length == READ_STEPS[0]:  # a full-length first chunk the parser cannot size (huge EXIF)
                return b"\xff\xd8\xff\xe1" + b"\x00" * (length - 4)
            return original(key, offset, length)

        with patch.object(adapter, "download_range", side_effect=ranged):
            result = probe_image(adapter, "big.jpg")
        self.assertEqual((result.width, result.height), (1024, 768))
        self.assertEqual(calls, list(READ_STEPS))

    def test_unreadable_files_are_reported_per_item(self):
        import os

        with open(os.path.join(self.base, "notes.txt"), "w") as fh:
            fh.write("not an image")
        self.write_image("ok.jpg")
        results = self.register(self.unsized("notes.txt"), self.unsized("missing.jpg"), self.unsized("ok.jpg"))
        self.assertEqual([r["status"] for r in results], ["error", "error", "created"])
        self.assertIn("not a readable image", results[0]["error"])
        self.assertIn("cannot read", results[1]["error"])

"""Shared types and enums used across the SDK."""
from __future__ import annotations

from enum import StrEnum


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class AnnotationSource(StrEnum):
    MANUAL = "manual"
    PREDICTION = "prediction"


class ExportFormat(StrEnum):
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    TFRECORD = "tfrecord"


class StorageBackend(StrEnum):
    S3 = "s3"
    AZURE = "azure"
    MINIO = "minio"
    LOCAL = "local"


class ModelTask(StrEnum):
    OBJECT_DETECTION = "object-detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class ModelFramework(StrEnum):
    YOLO = "yolo"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
